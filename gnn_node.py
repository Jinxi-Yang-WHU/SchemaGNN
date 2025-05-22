import argparse
import random
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from model.model import Model, LightModel, SchemaGuidedLightModel
from model.text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from Relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph, make_schema_graph
from Relbench.modeling.nn import SchemaGCN
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from Relbench.loss.reconstruction_loss import reconstruction_loss 
from Relbench.loss.regularization_loss import regularization_loss

from logger import create_logger

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-event")
parser.add_argument("--task", type=str, default="user-attendance")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()

hyperparams = {
    "hyperparams": "best"
}
#8 97 132 2726 32749
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-f1 --task driver-position
#CUDA_VISIBLE_DEVICES=2 python gnn_node.py --dataset rel-f1 --task driver-top3
#CUDA_VISIBLE_DEVICES=2 python gnn_node.py --dataset rel-f1 --task driver-dnf
#python gnn_node.py --dataset rel-f1 --task driver-dnf
#python gnn_node.py --dataset rel-trial --task study-outcome --seed 8
#python gnn_node.py --dataset rel-amazon --task user-churn 
#python gnn_node.py --dataset rel-avito --task user-clicks
#python gnn_node.py --dataset rel-hm --task user-churn
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-trial --task study-outcome 
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-trial --task study-adverse 
#CUDA_VISIBLE_DEVICES=2 python gnn_node.py --dataset rel-avito --task user-clicks --seed 8
#CUDA_VISIBLE_DEVICES=3 python gnn_node.py --dataset rel-trial --task site-success
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-event --task user-attendance --seed 8
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-event --task user-ignore --seed 8
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-event --task user-ignore --seed 8
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-hm --task user-churn --seed 8
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-amazon --task user-churn --seed 8
#CUDA_VISIBLE_DEVICES=0 python gnn_node.py --dataset rel-stack --task user-engagement --seed 8

model_name = "GNN"
dataset_name = args.dataset
task_name = args.task
logger = create_logger("output", model_name, dataset_name, task_name,  hyperparams)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

print("Yep1")
stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
print("Yep2")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(), batch_size=256 #256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)
print(len(data.metadata()[1]))
clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
).to(device)

def get_trainable_parameter_size(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    try:  # 尝试获取第一个可训练参数
        param = next(trainable_params)
        param_size = 4  # 默认 float32
        if param.dtype == torch.float16:
            param_size = 2

        total_num_params = sum(p.numel() for p in trainable_params)  # 注意：这里需要重新生成迭代器
        trainable_params = filter(lambda p: p.requires_grad, model.parameters()) # 重新生成迭代器

        total_size_bytes = total_num_params * param_size
        total_size_mb = total_size_bytes / (1024 ** 2)
        return total_size_mb
    except StopIteration:  # 如果没有可训练参数
        return 0.0

print(get_trainable_parameter_size(model))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] <= best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")

logger.log.close()
sys.stdout = sys.stdout.terminal

#python gnn_node.py --dataset rel-f1 --task driver-position
#Best test metrics: {'r2': -0.06045781452517196, 'mae': 4.417125164333142, 'rmse': 5.365578513686385}
#Best test metrics: {'r2': -0.007677934599587211, 'mae': 4.297663814645064, 'rmse': 5.230349742880213}
#Best test metrics: {'r2': -0.0005923878811164851, 'mae': 4.299094204735337, 'rmse': 5.211928547261949}
#Best test metrics: {'r2': 0.02263148801838688, 'mae': 4.23385450404987, 'rmse': 5.151088689047588}
#Best test metrics: {'r2': 0.015586933038529205, 'mae': 4.275673540935182, 'rmse': 5.1696190464097596}

#Best Val metrics: {'r2': 0.2567687947789724, 'mae': 3.192961326438583, 'rmse': 3.996798534316677}
#Best Val metrics: {'r2': 0.2728768382872182, 'mae': 3.179708676029223, 'rmse': 3.9532499949177726}
#Best Val metrics: {'r2': 0.2682329946406081, 'mae': 3.1857370924455926, 'rmse': 3.9658538131801193}
#Best Val metrics: {'r2': 0.26732412443753184, 'mae': 3.195672416304778, 'rmse': 3.968315886373416}
#Best Val metrics: {'r2': 0.2563203201522871, 'mae': 3.2018055413194553, 'rmse': 3.998004210559275}


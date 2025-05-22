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
from Relbench.loss.combine_loss import AutomaticWeightedLoss

from logger import create_logger
import sklearn


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
parser.add_argument("--num_bases", type=int, default=6)
parser.add_argument("--weight1", type=float, default=1.0)
parser.add_argument("--weight2", type=float, default=1.0)
parser.add_argument("--weight3", type=float, default=0.3)
parser.add_argument("--gamma", type=float, default=2.0)
parser.add_argument("--stop", type=int, default=10)
parser.add_argument("--logdir", type=str, default="runs/seed_0")
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()

hyperparams = {
    "num_bases": args.num_bases,
    "w1": args.weight1,
    "w2": args.weight2,
    "w3": args.weight3,
}

model_name = "Schema_GNN"
dataset_name = args.dataset
task_name = args.task
logger = create_logger("output", model_name, dataset_name, task_name, hyperparams)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
        text_embedder=GloveTextEmbedding(device=device), batch_size=256 #256
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

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text_model = AutoModel.from_pretrained('bert-base-uncased').eval()

def embed_word(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[0][0]  # 获取 [CLS] token 的 embedding

Sch_data, index_to_table, table_to_index, adj_matrix = make_schema_graph(
    dataset.get_db(),
    word_embedder=embed_word,
)

def get_trainable_parameter_size(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    try: 
        param = next(trainable_params)
        param_size = 4  
        if param.dtype == torch.float16:
            param_size = 2

        total_num_params = sum(p.numel() for p in trainable_params) 
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        total_size_bytes = total_num_params * param_size
        total_size_mb = total_size_bytes / (1024 ** 2)
        return total_size_mb
    except StopIteration:  
        return 0.0


model = SchemaGuidedLightModel(
    data=data,
    col_stats_dict=col_stats_dict,
    in_channels=768,
    schema_hidden_channels=256,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    num_bases=args.num_bases,
    aggr=args.aggr,
    norm="batch_norm",
    num_schema_layers=1
).to(device)
Sch_data = Sch_data.to(device)
adj_matrix = adj_matrix.to(device)
print(get_trainable_parameter_size(model))
loss_history = {'task1': [], 'task2': []}
awl = AutomaticWeightedLoss(2)

def train( epoch ) -> float:
    model.train()
    epoch_loss1 = 0.0
    epoch_loss2 = 0.0
    if epoch <= args.stop:
        optimizer = optimizer1
    else:
        optimizer = optimizer2

    if epoch <= 2:
        weight1, weight2  = 1.0, 1.0
    else:
        w1 = np.exp(loss_history['task1'][-1] / loss_history['task1'][-2])
        w2 = np.exp(loss_history['task2'][-1] / loss_history['task2'][-2])

        sum_w = w1 + w2
        weight1 = w1 / sum_w
        weight2 = w2 / sum_w

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    stp = 0
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred, schema_node_features, ori_node_features, bases = model(
            batch,
            task.entity_table,
            Sch_data, 
            table_to_index
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss_rec = reconstruction_loss(adjacent_matrix=adj_matrix, original_node_features=ori_node_features, fined_node_features=schema_node_features, gamma=args.gamma)
        loss_tsk = loss_fn(pred.float(), batch[entity_table].y.float())
        loss_reg = regularization_loss(bases)
        if epoch <= args.stop:
            div1 = loss_history["task1"][0] if epoch > 1 else 1.0
            div2 = loss_history["task2"][0] if epoch > 1 else 1.0
            loss = args.weight1*loss_rec + args.weight2*loss_tsk + args.weight3*loss_reg 
        else:
            loss = weight2*loss_tsk + weight3*loss_reg
        loss.backward()
        optimizer.step()

        epoch_loss1 += loss_rec.item() * pred.size(0)
        epoch_loss2 += loss_tsk.item() * pred.size(0)

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    avg_loss1 = epoch_loss1 / count_accum
    avg_loss2 = epoch_loss2 / count_accum

    loss_history['task1'].append(avg_loss1)
    loss_history['task2'].append(avg_loss2)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred, schema_node_features, ori_node_features, bases  = model(
            batch,
            task.entity_table,
            Sch_data,
            table_to_index
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

params_to_optimize = []
params_to_freeze = []
for name, param in model.named_parameters():
    if "gnn.schema_gcn" in name:
        params_to_freeze.append(param)
    else:
        params_to_optimize.append(param)

optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer2 = torch.optim.Adam([
    {'params': params_to_optimize, 'lr': args.lr},  
    {'params': params_to_freeze, 'lr': 0.0} 
])

state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train( epoch )
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




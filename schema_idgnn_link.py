import argparse
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from model.model import Model, LightModel, SchemaGuidedLightModel
from model.text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from Relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph, make_schema_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.loader import LinkNeighborLoader
from Relbench.modeling.nn import SchemaGCN
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from Relbench.loss.reconstruction_loss import reconstruction_loss 
from Relbench.loss.regularization_loss import regularization_loss

from logger import create_logger

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_bases", type=int, default=6)
parser.add_argument("--weight1", type=float, default=1.0)
parser.add_argument("--weight2", type=float, default=1.0)
parser.add_argument("--weight3", type=float, default=0.3)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--stop", type=int, default=2)
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples")
)
args = parser.parse_args()

hyperparams = {
    "num_bases": args.num_bases,
    "w1": args.weight1,
    "w2": args.weight2,
    "w3": args.weight3,
}
model_name = "Schema_IDGNN"
dataset_name = args.dataset
task_name = args.task
logger = create_logger("output", model_name, dataset_name, task_name,  hyperparams)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
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
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)
    dst_nodes_dict[split] = table_input.dst_nodes
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
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

model = SchemaGuidedLightModel(
    data=data,
    col_stats_dict=col_stats_dict,
    in_channels=768,
    schema_hidden_channels=256,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    num_bases=args.num_bases,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
    num_schema_layers=1
).to(device)
Sch_data = Sch_data.to(device)
adj_matrix = adj_matrix.to(device)
print(get_trainable_parameter_size(model))
train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)

params_to_optimize = []
params_to_freeze = []
for name, param in model.named_parameters():
    if "gnn.schema_gcn" in name:
        params_to_freeze.append(param)
    else:
        params_to_optimize.append(param)

optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer2 = torch.optim.Adam([
    {'params': params_to_optimize, 'lr': args.lr},  # 可以为不同的组设置不同的学习率
    {'params': params_to_freeze, 'lr': 0.0} # 将不需要更新的参数学习率设置为0
])
loss_history = {'task1': [], 'task2': []}

# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}")

def train( epoch ) -> float:
    model.train()
    epoch_loss1 = 0.0
    epoch_loss2 = 0.0
    if epoch <= args.stop:
        optimizer = optimizer1
    else:
        optimizer = optimizer2

    if epoch <= 2:
        weight1, weight2 = 1.0, 1.0
    else:
        w1 = np.exp(loss_history['task1'][-1] - loss_history['task1'][-2])
        w2 = np.exp(loss_history['task2'][-1] - loss_history['task2'][-2])
        sum_w = w1 + w2
        weight1 = w1 / sum_w
        weight2 = w2 / sum_w

    if epoch <= args.stop:
        optimizer = optimizer1
    else:
        optimizer = optimizer2
    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        out, schema_node_features, ori_node_features, bases  = model.forward_dst_readout(
            batch, task.src_entity_table, task.dst_entity_table, Sch_data, table_to_index
        )
        out = out.flatten( )

        batch_size = batch[task.src_entity_table].batch_size

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target label
        target = torch.isin(
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        # Optimization
        optimizer.zero_grad()
        loss_tsk = F.binary_cross_entropy_with_logits(out, target)
        loss_rec = reconstruction_loss(adjacent_matrix=adj_matrix, original_node_features=ori_node_features, fined_node_features=schema_node_features, gamma=args.gamma)
        loss_reg = regularization_loss(bases)

        if epoch <= args.stop:
            loss = weight1*loss_rec + weight2*loss_tsk + args.weight3*loss_reg
        else:
            loss = args.weight2*loss_tsk + args.weight3*loss_reg
        
        loss.backward()
        optimizer.step()

        epoch_loss1 += loss_rec.item() * out.numel()
        epoch_loss2 += loss_tsk.item() * out.numel()

        loss_accum += float(loss) * out.numel()
        count_accum += out.numel()

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    avg_loss1 = epoch_loss1 / count_accum
    avg_loss2 = epoch_loss2 / count_accum
    loss_history['task1'].append(avg_loss1)
    loss_history['task2'].append(avg_loss2)

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list: list[Tensor] = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out, schema_node_features, ori_node_features, bases  = model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table, Sch_data, table_to_index
            )
        out = out.detach( )
        out = out.flatten( )

        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train( epoch )
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
# print(loss_history["task1"])
# print(loss_history["task2"])
#CUDA_VISIBLE_DEVICES=3 python schema_idgnn_link.py --dataset rel-trial --task condition-sponsor-run

from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import nn
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.nn import HeteroConv, LayerNorm, PositionalEncoding, SAGEConv, GCNConv
from torch_geometric.typing import EdgeType, NodeType
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn import Parameter

class HeteroEncoder(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int):
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(channels) for node_type in node_types}
        )
        self.lin_dict = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(channels, channels) for node_type in node_types}
        )

    def reset_parameters(self):
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
        self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class LightHeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: list,
        edge_types: list,
        channels: int,  # 统一的通道数
        num_bases: int,  # 参数基的数量
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()
        self.channels = channels  # 统一的通道数
        self.num_layers = num_layers

        # 创建参数基
        self.bases_list = nn.ParameterList( )

        # 创建组合系数
        self.edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(edge_types)}
        self.combination_coeffs = Parameter(torch.randn(len(edge_types), num_bases))

        # 创建卷积层和归一化层
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.bases_list.append(Parameter(torch.randn(num_bases, channels, channels)))
            conv_dict = dict()
            for edge_type in edge_types:
                #edge_type_str = "_".join(edge_type) 
                # 获取组合系数
                edge_idx = self.edge_type_to_idx[edge_type]
                coeffs = self.combination_coeffs[edge_idx]
                # 计算当前边类型的参数

                conv_dict[edge_type] = SAGEConv((channels, channels), channels, aggr=aggr, bias=True) # channels
                conv_dict[edge_type].lin_l.weight.requires_grad = False  # 设置 lin_l 权重不可学习
                if conv_dict[edge_type].lin_r : # 针对非对称聚合
                    conv_dict[edge_type].lin_r.weight.requires_grad = False  # 设置 lin_r 权重不可学习

            conv = HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)

            for _ in range(num_layers):
                norm_dict = torch.nn.ModuleDict()
                for node_type in node_types:
                    norm_dict[node_type] = LayerNorm(channels, mode="node")
                self.norms.append(norm_dict)


    def reset_parameters(self):
        xavier_uniform_(self.combination_coeffs)
        for bases in self.bases_list:
            xavier_uniform_(bases)
        for conv in self.convs:
            for module in conv.convs.values(): 
                module.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()


    def forward(self, x_dict, edge_index_dict):
        for i, (conv, norm_dict, bases) in enumerate(zip(self.convs, self.norms, self.bases_list)):
            # 更新每一层的参数
            conv_dict = conv.convs
            for edge_type in edge_index_dict.keys():
                #edge_type_str = "_".join(edge_type) 
                edge_idx = self.edge_type_to_idx[edge_type]
                coeffs = self.combination_coeffs[edge_idx]
                weights = torch.einsum("i,ijk->jk", coeffs, bases)
                conv_dict[edge_type].lin_l.weight.data = weights
                if conv_dict[edge_type].lin_r:
                    conv_dict[edge_type].lin_r.weight.data = weights

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict 


class SchemaGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super().__init__()
        self.pre_process = nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_process(x)
        ori_x = x
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return x, ori_x

class DynamicCoeffHeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: list,
        edge_types: list,
        in_channels: int,
        schema_hidden_channels: int,
        channels: int,
        num_bases: int,
        num_schema_layers: int = 1,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

        # SchemaGCN
        self.schema_gcn = SchemaGCN(in_channels, schema_hidden_channels, num_layers=num_schema_layers)
        self.coeff_linear = nn.Linear(2 * schema_hidden_channels, num_bases)

        # 卷积层和归一化层
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.bases_list = nn.ParameterList() # 为每一层创建独立的参数基

        for _ in range(num_layers):
            # 生成 combination_coeffs 的线性层

            # 参数基
            self.bases_list.append(Parameter(torch.randn(num_bases, channels, channels)))

            conv_dict = dict()
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv((channels, channels), channels, aggr=aggr, bias=True)
                conv_dict[edge_type].lin_l.weight.requires_grad = False  # 设置 lin_l 权重不可学习
                if conv_dict[edge_type].lin_r : # 针对非对称聚合
                    conv_dict[edge_type].lin_r.weight.requires_grad = False  # 设置 lin_r 权重不可学习

            conv = HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)

            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)
            self.edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(edge_types)}


    def reset_parameters(self):
        xavier_uniform_(self.coeff_linear.weight) # 只需初始化一次coeff_linear
        for bases in self.bases_list:
            xavier_uniform_(bases)
        for conv in self.convs:
            for module in conv.convs.values():
                module.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(self, x_dict, edge_index_dict, schema_data, table_to_index):
        schema_node_features, ori_node_features = self.schema_gcn(schema_data)

        for i, (conv, norm_dict, bases) in enumerate(zip(self.convs, self.norms, self.bases_list)):
            conv_dict = conv.convs
            for edge_type in edge_index_dict.keys():
                src_type, _, dst_type = edge_type
                src_schema_features = schema_node_features[table_to_index[src_type]]
                dst_schema_features = schema_node_features[table_to_index[dst_type]]

                combined_schema_features = torch.cat([src_schema_features, dst_schema_features], dim=0)
                coeffs = self.coeff_linear(combined_schema_features)  # 使用每一层的线性层

                weights = torch.einsum("i,ijk->jk", coeffs, bases)  # 使用每一层的bases
                conv_dict[edge_type].lin_l.weight.data = weights
                if conv_dict[edge_type].lin_r:
                    conv_dict[edge_type].lin_r.weight.data = weights

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict, schema_node_features, ori_node_features, self.bases_list

# class DynamicCoeffHeteroGraphSAGE(torch.nn.Module):
#     def __init__(
#         self,
#         node_types: list,
#         edge_types: list,
#         in_channels: int,
#         schema_hidden_channels: int,
#         channels: int,
#         num_bases: int,
#         num_schema_layers: int = 1,
#         aggr: str = "mean",
#         num_layers: int = 2,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.num_layers = num_layers

#         # SchemaGCN
#         self.schema_gcn = SchemaGCN(in_channels, schema_hidden_channels, num_layers=num_schema_layers)

#         # 生成 combination_coeffs 的线性层
#         self.coeff_linear = nn.Linear(2 * schema_hidden_channels, num_bases)

#         # 参数基
#         self.bases = Parameter(torch.randn(num_bases, channels, channels))
#         self.edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(edge_types)}


#         # 卷积层和归一化层
#         self.convs = torch.nn.ModuleList()
#         self.norms = torch.nn.ModuleList()

#         for _ in range(num_layers):
#             conv_dict = dict()
#             for edge_type in edge_types:
#                 conv_dict[edge_type] = SAGEConv((channels, channels), channels, aggr=aggr, bias=True)
#                 conv_dict[edge_type].lin_l.weight.requires_grad = False  # 设置 lin_l 权重不可学习
#                 if conv_dict[edge_type].lin_r : # 针对非对称聚合
#                     conv_dict[edge_type].lin_r.weight.requires_grad = False  # 设置 lin_r 权重不可学习

#             conv = HeteroConv(conv_dict, aggr="sum")
#             self.convs.append(conv)

#         for _ in range(num_layers):
#             norm_dict = torch.nn.ModuleDict()
#             for node_type in node_types:
#                 norm_dict[node_type] = LayerNorm(channels, mode="node")
#             self.norms.append(norm_dict)

#     def reset_parameters(self):
#         xavier_uniform_(self.bases)
#         xavier_uniform_(self.coeff_linear.weight)
#         for conv in self.convs:
#             for module in conv.convs.values():
#                 module.reset_parameters()
#         for norm_dict in self.norms:
#             for norm in norm_dict.values():
#                 norm.reset_parameters()

#     def forward(self, x_dict, edge_index_dict, schema_data, table_to_index):
#         schema_node_features, ori_node_features = self.schema_gcn(schema_data)

#         for conv, norm_dict in zip(self.convs, self.norms):
#             conv_dict = conv.convs
#             for edge_type in edge_index_dict.keys():
#                 src_type, _, dst_type = edge_type
#                 src_schema_features = schema_node_features[table_to_index[src_type]]
#                 dst_schema_features = schema_node_features[table_to_index[dst_type]]

#                 combined_schema_features = torch.cat([src_schema_features, dst_schema_features], dim=0)
#                 coeffs = self.coeff_linear(combined_schema_features)

#                 weights = torch.einsum("i,ijk->jk", coeffs, self.bases)
#                 conv_dict[edge_type].lin_l.weight.data = weights
#                 if conv_dict[edge_type].lin_r:
#                     conv_dict[edge_type].lin_r.weight.data = weights

#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
#             x_dict = {key: x.relu() for key, x in x_dict.items()}

#         return x_dict, schema_node_features, ori_node_features, self.bases
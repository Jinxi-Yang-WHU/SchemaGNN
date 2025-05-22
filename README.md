### SchemaGNN: Schema Graph-Guided Graph Neural Network for Relational Deep Learning

----
# Overview
Relational data is the backbone of countless real-world applications, and predicting future trends or behaviors from this data is crucial. Relational Deep Learning (RDL) has emerged as a powerful approach, often transforming relational data into graphs and applying Graph Neural Networks (GNNs).

However, existing RDL methods face two significant challenges:

1.  **High Memory Costs:** They typically learn distinct, large weight matrices for each type of relationship (edge type), leading to substantial memory overhead, especially with complex schemas.
2.  **Ignoring Schema Structure:** They often overlook the database schema itself – the blueprint that defines valid entity types and how they can relate. This schema graph contains vital structural regularities and domain knowledge that could guide the learning process, improve generalization, and enhance robustness.

**🚀 Introducing Schema-GNN!**

Schema-GNN addresses these limitations head-on by:

*   🧠 **Efficient Parameterization with Parameter Basis:** Instead of learning huge, independent weight matrices for each edge type, Schema-GNN defines a shared `Parameter Basis` (a set of smaller basis matrices). Each edge-specific weight matrix is then efficiently decomposed into a linear combination of these basis matrices. This drastically reduces the number of trainable parameters and memory requirements.
*   🔗 **Leveraging the Schema Graph for Guidance:** We explicitly use the schema graph! Schema-GNN learns embeddings from the schema graph's nodes. These schema embeddings then intelligently inform how the shared `Parameter Basis` elements are combined to form the specific weights for each edge type. This injects critical structural priors directly into the model.
*   💡 **Enhanced Expressiveness via Regularization:** A novel regularization loss is introduced that encourages dissimilarity between the basis matrices in our `Parameter Basis`. This promotes more expressive representations and helps mitigate overfitting.

By intelligently integrating schema knowledge and employing an efficient parameter-sharing mechanism, Schema-GNN achieves superior performance and robustness compared to state-of-the-art RDL methods on various relational learning tasks.

**This repository contains the official implementation for our paper, "Schema-GNN: Schema Graph-Guided Graph Neural Networks for Relational Deep Learning."**

# Design of SchemaGNN
![Overall framework of Schema Graph-Guided Graph Neural Network](/schema-gnn.jpg)
There are three main components of SchemaGNN: **Extraction of the schema graph information**, **Parameter Basis**, and **Regularization**
# Installation
Our project is based on the Relational Deep Learning Benchmark(RelBench), so you need to first install RelBench using ```pip```
```
pip install relbench
```
Additionally, our project requires [PyTorch](https://pytorch.org), [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [Pytorch Frame](https://github.com/pyg-team/pytorch-frame). Please install these dependencies manually or do:
```
pip install relbench[full]
```
# Run the Code
There are three kinds of models in this project: Relational Deep Learning(RDL), GNN with Parameter Basis but without the guidance of the schema graph(We call it LightModel for simplicity), SchemaGNN.

To run RDL, you can use the command ```python gnn_node.py``` for the node-level tasks and ```python gnn_link.py``` for the edge-level tasks.

To run LightModel, you can use the command ```python light_gnn_node.py``` for the node-level tasks and ```python light_gnn_link.py``` for the edge-level tasks.

To run SchemaGNN, you can use the command ```python schema_gnn_node.py``` for the node-level tasks and ```python schema_gnn_link.py``` for the edge-level tasks.

You can configure the ```--dataset``` and ```--task``` parameters to specify which dataset and task the model should be trained and evaluated on. For example, to train and test the SchemaGNN on the ```driver-position``` task from the ```rel-f1``` dataset, run the following script:
```
python schema_gnn_node.py --dataset rel-f1 --task driver-position
```
You can also configure the ```--num_bases``` parameter to set the number of weight matrices used in the Parameter Basis. For example, to train and test SchemaGNN on the ```driver-position``` task from the ```rel-f1``` dataset with 6 weight matrices in the Parameter Basis, run the following script:
```
python schema_gnn_node.py --dataset rel-f1 --task driver-position --num_bases 6
```
You can configure the ```--weight1```, ```--weight2```, and ```--weight3``` parameters to adjust the linear coefficients for the reconstruction loss $\mathcal{L}{rec}$, the task-specific loss $\mathcal{L}{tsk}$, and the regularization loss $\mathcal{L}{reg}$, respectively. For example, to train and test SchemaGNN on the ```driver-position``` task from the ```rel-f1``` dataset using 6 weight matrices in the Parameter Basis, with the overall loss defined as $1.0 \cdot \mathcal{L}{rec} + 1.0 \cdot \mathcal{L}{tsk} + 0.3 \cdot \mathcal{L}{reg}$, execute the following command:
```
python schema_gnn_node.py --dataset rel-f1 --task driver-position --num_bases 6 --weight1 1.0 --weight2 1.0 --weight3 0.3
```

## Acknowledgments
This project is based on [relbench](https://github.com/snap-stanford/relbench). Many thanks to the authors for their excellent work.

# Cite SchemaGNN

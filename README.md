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
# Package Usage

## Acknowledgments
This project is based on [relbench](https://github.com/snap-stanford/relbench). Many thanks to the authors for their excellent work.

# Cite SchemaGNN

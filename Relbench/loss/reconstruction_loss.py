import torch
from torch import tensor
def node_reconstruction_loss(original_node_features, fined_node_features, gamma):
    dot_products = torch.sum(original_node_features * fined_node_features, dim=1)
    norm1 = torch.norm(original_node_features, dim=1)
    norm2 = torch.norm(fined_node_features, dim=1)
    cosine_similarities = dot_products / (norm1 * norm2)
    cosine_similarities = torch.nan_to_num(cosine_similarities, nan=0.0)
    cosine_distances = 1.0 - cosine_similarities
    gamma_power = cosine_distances ** gamma
    mean_gamma_power = torch.mean(gamma_power)

    return mean_gamma_power

def reconstruction_loss(
    adjacent_matrix,
    original_node_features,
    fined_node_features,
    gamma = 1.0
):
    adjacent_matrix = adjacent_matrix + torch.eye( adjacent_matrix.size(0) ).to("cuda")
    features_norms = torch.norm(fined_node_features, dim=1, keepdim=True)
    fined_node_features = fined_node_features / features_norms
    reconstruction_adjacent_matrix = fined_node_features@fined_node_features.T
    # print("1:")
    # print(reconstruction_adjacent_matrix)
    # reconstruction_adjacent_matrix = torch.sigmoid(reconstruction_adjacent_matrix)
    # print("2:")
    # print(reconstruction_adjacent_matrix )
    edge_reconstruction_loss = torch.norm(reconstruction_adjacent_matrix-adjacent_matrix, p=2)**2
    return node_reconstruction_loss(original_node_features, fined_node_features, gamma) + edge_reconstruction_loss

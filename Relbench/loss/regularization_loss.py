import torch
from torch import tensor

# def regularization_loss( bases_list ):
#     total_reg_loss = 0
    
#     for bases in bases_list:
#         layer_reg_loss = 0
#         hadamard_products = bases[:-1] * bases[1:]  # (m-1, n, n)
#         elementwise_sums = torch.sum(hadamard_products, dim=(1, 2))
#         layer_reg_loss = torch.sum(torch.abs(elementwise_sums))
#         total_reg_loss += layer_reg_loss

#     return total_reg_loss

#     hadamard_products = bases[:-1] * bases[1:]  # (m-1, n, n)
#     elementwise_sums = torch.sum(hadamard_products, dim=(1, 2))
#     loss = torch.sum(torch.abs(elementwise_sums))
#     return loss

# def regularization_loss( bases_list ):
#     total_reg_loss = 0
#     for bases in bases_list:
#         gram_matrix = torch.einsum('bij,bkj->ik', bases, bases) # 计算Gram矩阵
#         layer_reg_loss = -torch.det(gram_matrix) # 使用负的行列式值
#         total_reg_loss += layer_reg_loss
#     return total_reg_loss

def regularization_loss(bases_list):
    total_reg_loss = 0
    for bases in bases_list:
        layer_reg_loss = 0
        for i in range(bases.shape[0]):
            for j in range(i + 1, bases.shape[0]):  # 遍历所有不同的矩阵对
                cos_sim = torch.cosine_similarity(bases[i].view(-1), bases[j].view(-1), dim=0)
                layer_reg_loss += cos_sim ** 2 # 使用负的余弦相似度
        total_reg_loss += layer_reg_loss
    return total_reg_loss

bases = torch.tensor([
    [
    [-1.0,-2.0,-3.0],
    [-4.0,-5.0,-6.0],
    [-7.0,-8.0,-9.0]
    ],
    [
    [1.0,2.0,4.0],
    [1.0,6.0,9.0],
    [3.0,5.0,7.0]
    ],
    [
    [2.0,7.0,5.0],
    [3.0,2.0,9.0],
    [4.0,9.0,6.0]
    ],
])

# print(regularization_loss(bases))
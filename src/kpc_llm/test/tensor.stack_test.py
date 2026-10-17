import torch

a = torch.tensor([[1, 2,4], [3, 4,6]], dtype=torch.float32)  # 形状: [2, 3]
b = torch.tensor([[5, 6,1], [7, 8,9]], dtype=torch.float32)   # 形状: [2, 3]

# res = torch.stack([a, b], dim=0)
# # 0位置维度前面多一个维度，多的维度和原维度一样
# print(res.shape)  # 输出: torch.Size([2, 2, 3])
# print(res)
# print("----------------------------------------------------")
# # 最后一个维度的前面增加个维度 加的维度和最后一个维度一样
# res = torch.stack([a, b], dim=-1)
# print(res.shape)  # 输出: torch.Size([2, 3, 2])
# print(res)
# print("----------------------------------------------------")
# # 倒数一个维度的前面增加个维度 加的维度和最后一个维度一样
# res = torch.stack([a, b], dim=-2)
# print(res.shape)  # 输出: torch.Size([2, 2, 3])
# print(res)
# print("----------------------------------------------------")
# # 默认是在最左边的维度0维度stack
# res = torch.stack([a, b])
# print(res.shape)  # 输出: torch.Size([2, 2, 3])
# print(res)

logits = a.softmax(dim=-1)
print(logits)
print(logits.shape)
print(logits.shape[0])
print(logits.size(0))
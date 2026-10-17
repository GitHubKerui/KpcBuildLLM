

from numpy import long
import torch

logits_1 = torch.tensor([[1,0],[0,1]],dtype=torch.float32)
logits_2 = torch.tensor([[1,0],[0,1],[0,1]],dtype=torch.float32)
logits_3 = torch.tensor([[1,0],[0,1],[0,1]],dtype=torch.float32)
logits_4 = torch.tensor([[1,0],[0,1],[0,1]],dtype=torch.float32)

target_1 = torch.tensor([0,1])
target_2 = torch.tensor([0,1,0])
target_3 = torch.tensor([0,1,-100])
target_4 = torch.tensor([0,1,-100])

loss1 = torch.nn.functional.cross_entropy(logits_1,target_1)
loss2 = torch.nn.functional.cross_entropy(logits_2,target_2)
loss3 = torch.nn.functional.cross_entropy(logits_3,target_3)
loss4 = torch.nn.functional.cross_entropy(logits_4,target_4)

""" -100 可以起到屏蔽作用，忽略某个位置的样本不参与loss计算，也就是这个样本不参与梯度更新。
    As we can see, the resulting loss on these 3 training examples is the same as the loss we calculated from the 2 training examples, 
    which means that the cross-entropy loss function ignored the training example with the -100 label
-   By default, PyTorch has the `cross_entropy(..., ignore_index=-100)` setting to ignore examples corresponding to the label -100
"""
print(f"loss1: {loss1}")
print(f"loss2: {loss2}")
print(f"loss3: {loss3}")
print(f"loss4: {loss4}")
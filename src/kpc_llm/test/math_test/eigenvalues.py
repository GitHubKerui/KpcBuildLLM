""" 特征值 和 特征向量 , 实对称矩阵, 特征值组成的对角矩阵，特征向量组成的正交矩阵，以及对角化和原实对称矩阵的对应关系"""

import numpy as np

# 1. 定义你的 4x4 实对称矩阵 A
A = np.array([
    [ 0,  1,  1, -1],
    [ 1,  0, -1,  1],
    [ 1, -1,  0,  1],
    [-1,  1,  1,  0]
], dtype=float)

# 2. 调用专门针对实对称矩阵的 eigh 函数
# 它会自动完成：求特征值 -> 特征向量正交化 -> 特征向量单位化
eigenvalues, eigenvectors = np.linalg.eigh(A) 

print("=== 1. 特征值 ===")
print(np.round(eigenvalues, 4)) 

print("\n=== 2. 标准正交矩阵 P (每一列是一个标准正交特征向量) ===")
print(np.round(eigenvectors, 4))
print("\n=== 2.1 标准正交矩阵 P^T == P^-1 ===")
print(np.round(eigenvectors.T , 4))


print("\n=== 3. 验证对角化结果 (P^T * A * P) ===A的对角矩阵")
A_diag = eigenvectors.T @ A @ eigenvectors
print(np.round(A_diag, 4))
print("\n=== 3.1 验证对角化结果 A === (P * A的对角矩阵 * P^T) ")
A_diag_reconstructed = eigenvectors @ A_diag @ eigenvectors.T
print(np.round(A_diag_reconstructed, 4))

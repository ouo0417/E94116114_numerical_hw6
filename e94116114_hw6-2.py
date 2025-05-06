import numpy as np

# 給定的矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

n = A.shape[0]

# Step 1: LU 分解（Doolittle 法：L 對角線為 1）
L = np.zeros((n, n))
U = np.zeros((n, n))

for i in range(n):
    L[i][i] = 1
    for j in range(i, n):
        U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
    for j in range(i+1, n):
        L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]

# Step 2: 解 Ax = I → 解 4 組 Ly = e_i, Ux = y
inv_A = np.zeros((n, n))
I = np.eye(n)

for col in range(n):
    # 解 Ly = e_i
    y = np.zeros(n)
    for i in range(n):
        y[i] = I[i, col] - sum(L[i][j] * y[j] for j in range(i))

    # 解 Ux = y
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    # 填入反矩陣的第 col 列
    inv_A[:, col] = x

# 顯示結果
np.set_printoptions(precision=5, suppress=True)
print("A 的反矩陣為：")
print(inv_A)

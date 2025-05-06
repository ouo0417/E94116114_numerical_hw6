import numpy as np

A = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)

b = np.array([2, 3, 4, 1], dtype=float)
n = len(b)

# Crout 分解
L = np.zeros((n, n))
U = np.identity(n)

for j in range(n):
    for i in range(j, n):
        L[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))
    for i in range(j+1, n):
        U[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(j))) / L[j][j]

# Forward substitution: Ly = b
y = np.zeros(n)
for i in range(n):
    y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

# Backward substitution: Ux = y
x = np.zeros(n)
for i in reversed(range(n)):
    x[i] = y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))

print("解 x =", x)

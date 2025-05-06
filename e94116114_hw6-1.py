import numpy as np

# 擴展係數矩陣 [A | b]
Ab = np.array([
    [1.19, 2.11, -100, 1, 1.12],
    [14.2, -0.112, 12.2, -1, 3.44],
    [0, 100, -99.9, 1, 2.15],
    [15.3, 0.110, -13.1, -1, 4.16]
], dtype=float)

n = len(Ab)

# Gaussian elimination with partial pivoting
for k in range(n):
    # Pivoting: 找目前欄的最大值那列，跟第 k 列交換
    max_row = max(range(k, n), key=lambda i: abs(Ab[i][k]))
    if k != max_row:
        Ab[[k, max_row]] = Ab[[max_row, k]]
    
    # 消去：讓第 k 列以下的第 k 欄都變成 0
    for i in range(k + 1, n):
        factor = Ab[i][k] / Ab[k][k]
        Ab[i, k:] -= factor * Ab[k, k:]

# Back substitution
x = np.zeros(n)
for i in reversed(range(n)):
    x[i] = (Ab[i][-1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i][i]

# 輸出答案
for i, val in enumerate(x, 1):
    print(f"x{i} = {val:.6f}")

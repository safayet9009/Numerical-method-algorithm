import numpy as np

# Coefficient matrix (a)
A = np.array([[3, 2],
              [2, 1]])

# Constant matrix (b)
B = np.array([60, 40])

# Determinant of A
det_A = np.linalg.det(A)

# Replace columns for x and y
A_x = np.array([[60, 2],
                [40, 1]])
A_y = np.array([[3, 60],
                [2, 40]])

# Determinants for x and y
det_x = np.linalg.det(A_x)
det_y = np.linalg.det(A_y)

# Cramer's Rule formulas
x = det_x / det_A
y = det_y / det_A

print("Units of Product A =", round(x, 2))
print("Units of Product B =", round(y, 2))

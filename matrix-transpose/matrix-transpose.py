import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    matrix = np.array(A).T
    return matrix

A = [
    [1, 2, 3],
    [4, 5, 6]
]
result = matrix_transpose(A)
print(result)
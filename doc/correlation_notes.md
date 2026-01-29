# Creating a Correlation Mask

**Problem:** You want to find highly correlated feature pairs in a correlation matrix.

**Solution:** Create a mask to exclude the diagonal and the upper triangle.

## Example
```
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_corr = corr_matrix.where(upper_triangle)
```

This code extracts just the upper triangle of a correlation matrix, which is useful because correlation matrices are symmetric (the correlation of A with B equals B with A).

**Line 1: Creating the mask**
```python
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
```

Breaking it down:
- `np.ones(corr_matrix.shape)` creates a matrix of 1s with the same shape as your correlation matrix
- `np.triu(..., k=1)` keeps only the upper triangle, setting everything else to 0. The `k=1` means start one diagonal above the main diagonal (excluding the diagonal itself, which is always 1.0 in a correlation matrix)
- `.astype(bool)` converts it to True/False

For a 4x4 matrix, the mask looks like:
```
[[False,  True,  True,  True],
 [False, False,  True,  True],
 [False, False, False,  True],
 [False, False, False, False]]
```

**Line 2: Applying the mask**
```python
upper_corr = corr_matrix.where(upper_triangle)
```

`.where()` keeps values where the mask is `True` and replaces everything else with `NaN`:
```
        A     B     C     D
A     NaN  0.85  0.32 -0.12
B     NaN   NaN  0.67  0.45
C     NaN   NaN   NaN -0.28
D     NaN   NaN   NaN   NaN
```

**Why do this?**

When finding highly correlated feature pairs, you don't want to count each pair twice (A-B and B-A) or include the diagonal (A-A = 1.0). This gives you each unique pair exactly once.
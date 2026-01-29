

# Python Notes




<details>
  <summary>Showing a subset of df.describe</summary>

## Showing a subset of df.describe

> using pandas and numpy
> ```python
> import numpy as np
> import pandas as pd
> ```
 
### One Liner
```python

# describe only integer columns, show only the min and max, sort by max
print(df.describe(include=[np.integer]).loc[['min','max']].T.sort_values(by='max'))
```

- Output
```text
                  min        max
credit.policy     0.0        1.0
not.fully.paid    0.0        1.0
pub.rec           0.0        5.0
delinq.2yrs       0.0       13.0
inq.last.6mths    0.0       33.0
fico            612.0      827.0
revol.bal         0.0  1207359.0
```

### Create a dataframe with min and max values for integer columns
```python
# use a data frame, preserving the index
int_columns = df.select_dtypes(include=[np.integer]).columns.tolist()

result = pd.DataFrame({
    'min': df[int_columns].min(),
    'max': df[int_columns].max()
})
print(result.sort_values(by='max'))
```

### Select by data types and iterate over columns
```python
int_columns = df.select_dtypes(include=[np.integer]).columns.tolist()

for col in int_columns:
    print(f"{col:<20} {df[col].min():>15} {df[col].max():>15}")
```

- Output
```text
['credit.policy', 'fico', 'revol.bal', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']
credit.policy                      0               1
fico                             612             827
revol.bal                          0         1207359
inq.last.6mths                     0              33
delinq.2yrs                        0              13
pub.rec                            0               5
not.fully.paid                     0               1
```

### Create a dataframe with a sequential index
```python
int_columns = df.select_dtypes(include=[np.integer]).columns.tolist()

result = pd.DataFrame({
    'Column': int_columns,
    'Min': df[int_columns].min().values,
    'Max': df[int_columns].max().values
})
print(result)
```
- Output
```text
           Column  Min      Max
0   credit.policy    0        1
1            fico  612      827
2       revol.bal    0  1207359
3  inq.last.6mths    0       33
4     delinq.2yrs    0       13
5         pub.rec    0        5
6  not.fully.paid    0        1
```
</details>
import pandas as pd

df1 = pd.DataFrame({
    "k": [1, 1, 2, 3],
    "k0": [0, 0, 0, 0],
    "v1": ["a", "b", "b", "c"]})
df2 = pd.DataFrame({
    "k": [1, 1, 2, 3, 3],
    "k0": [0, 1, 0, 0, 0],
    "v1": [10, 11, 20, 30, 31]})
print(df1.columns)
result = pd.merge(df1, df2, on=["k", "k0"], how="inner")
print(result)
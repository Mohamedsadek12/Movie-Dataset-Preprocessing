import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('movies.csv')

print(f"The Shape of the Dataset is:\n{df.shape}")
print("----------------------------------------------------------------")
print(f"First 5 Rawes is:\n{df.head()}")
print("----------------------------------------------------------------")
print(f"Column Data types is:\n{df.dtypes}")
print("----------------------------------------------------------------")
print(f"Numerical Columns Data types:\n{df.select_dtypes(include=[np.number]).columns}")
print(f"Categorical Columns Data types:\n{df.select_dtypes(exclude=[np.number]).columns}")

print("----------------------------------------------------------------")
print(f"Missing Values:\n{df.isnull().sum()}")
print("----------------------------------------------------------------")
print(df.describe())
print("----------------------------------------------------------------")


















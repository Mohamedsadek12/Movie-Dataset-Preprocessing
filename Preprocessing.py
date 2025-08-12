import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("movies.csv")

#--------------------
# 1- Data Information
#--------------------

print(f"Dataset Shape:\n{df.shape}")
print("-----------------------------------------------------------------")
print(f"First 5 Rows:\n{df.head()}")
print("-----------------------------------------------------------------")
print(f"Data Types:\n{df.dtypes}")
print("-----------------------------------------------------------------")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Numeric columns:\n{numerical_cols}")
print(f"Categorical columns:\n{categorical_cols}")
print("-----------------------------------------------------------------")
print(f"Missing Values:\n{df.isna().sum()}")
print("-----------------------------------------------------------------")
print(f"duplicated Values:\n{df.duplicated().sum()}")
print("-----------------------------------------------------------------")
print(f"Dataset statistic Information:\n{df.describe()}")
print("-----------------------------------------------------------------")

#-----------------
# 2- Preprocessing
#-----------------

df["Worldwide Gross"] = pd.to_numeric(df["Worldwide Gross"].replace('[$]', '', regex=True).str.strip()).astype(float)
print(df.dtypes)

fill_with_median = ["Worldwide Gross", "Profitability"]
fill_with_mean = ["Rotten Tomatoes %", "Audience score %"]
fill_with_mode = ["Genre", "Lead Studio", "Year"]

# Fill with median
for col in fill_with_median:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill with mean
for col in fill_with_mean:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Fill with mode
for col in fill_with_mode:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

print(df.isna().sum())

#---------------------
# 3- Handle Duplicates
#---------------------

df = df.drop_duplicates(subset=["Film"], keep="first")

print(f"duplicated Values:\n{df.duplicated().sum()}")

# --- Feature Engineering ---
df["Decade"] = (df["Year"] // 10 * 10).astype(str) + "s"
df["Is_Independent"] = df["Lead Studio"].str.lower().eq("independent")
df["Critic_Audience_Gap"] = df["Audience score %"] - df["Rotten Tomatoes %"]
df["ROI_Category"] = df["Profitability"].apply(lambda x: "Low" if (3.0 <= x < 6.0) else("Medium" if (6.0 <= x < 9.0) else "High"))

print(df["Decade"])
print(df["Is_Independent"])
print(df["Critic_Audience_Gap"])
print(df["ROI_Category"])


# --- Outlier Detection ---
def detect_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return column[(column < lower) | (column > upper)]

profit_outliers = detect_iqr(df["Profitability"])
gross_outliers = detect_iqr(df["Worldwide Gross"])

print(f"\nNumber of Profitability Outliers: {len(profit_outliers)}\nand they are:\n{profit_outliers}")
print(f"Number of Worldwide Gross Outliers: {len(gross_outliers)}\nand they are:\n{gross_outliers}")

# Fix Data Types

df["Year"] = df["Year"].astype(int)
print(df.dtypes)
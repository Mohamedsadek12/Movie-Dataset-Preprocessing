import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("movies.csv")

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
print(f"Dataset statistic Information:\n{df.describe()}")
print("-----------------------------------------------------------------")

#-----------------
# 2- Preprocessing
#-----------------


df["Worldwide Gross"] = pd.to_numeric(df["Worldwide Gross"].replace('[$]', '', regex=True).str.strip(),errors='coerce')

fill_with_median = ["Worldwide Gross", "Profitability"]
fill_with_mean = ["Rotten Tomatoes %", "Audience score %"]
fill_with_mode = ["Genre", "Lead Studio", "Year"]

print(df)
print(df.dtypes)

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


# # Special case: Worldwide Gross might be string like "$1,234.5"
# if df["Worldwide Gross"].dtype == object:
#     df["Worldwide Gross"] = (
#         df["Worldwide Gross"]
#         .replace('[\$,]', '', regex=True)
#         .astype(float)
#     )
#
# # --- Handle Duplicates ---
# df = df.drop_duplicates(subset=["Film"], keep="first")
#
# # --- Feature Engineering ---
# df["Decade"] = (df["Year"] // 10 * 10).astype(str) + "s"
# df["Is_Independent"] = df["Lead Studio"].str.lower().eq("independent")
# df["Critic_Audience_Gap"] = df["Audience score %"] - df["Rotten Tomatoes %"]
#
# # ROI Category
# def roi_category(p):
#     if p >= 5:
#         return "High"
#     elif p >= 2:
#         return "Medium"
#     else:
#         return "Low"
# df["ROI_Category"] = df["Profitability"].apply(roi_category)
#
# # --- Encode Categorical Variables ---
# from sklearn.preprocessing import LabelEncoder
# le_genre = LabelEncoder()
# df["Genre_Label"] = le_genre.fit_transform(df["Genre"])
#
# # One-hot encode top 5 studios
# top_studios = df["Lead Studio"].value_counts().nlargest(5).index
# df = pd.get_dummies(df, columns=["Lead Studio"], prefix="Studio", dummy_na=False)
#
# # --- Outlier Detection ---
# def detect_outliers_iqr(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     return series[(series < lower) | (series > upper)]
#
# profit_outliers = detect_outliers_iqr(df["Profitability"])
# gross_outliers = detect_outliers_iqr(df["Worldwide Gross"])
#
# print(f"\nProfitability Outliers: {len(profit_outliers)}")
# print(f"Worldwide Gross Outliers: {len(gross_outliers)}")

# # =========================
# # 3️⃣ EDA & Plots
# # =========================
#
# sns.set(style="whitegrid", palette="muted")
#
# # --- Basic Distributions ---
# plt.figure(figsize=(8,4))
# sns.countplot(x="Genre", data=df, order=df["Genre"].value_counts().index)
# plt.xticks(rotation=45)
# plt.title("Movies per Genre")
# plt.show()
#
# plt.figure(figsize=(8,4))
# sns.countplot(x="Decade", data=df)
# plt.title("Movies per Decade")
# plt.show()
#
# # --- Profitability Analysis ---
# plt.figure(figsize=(8,4))
# sns.barplot(x="Genre", y="Profitability", data=df, estimator=np.mean, order=df["Genre"].value_counts().index)
# plt.xticks(rotation=45)
# plt.title("Average Profitability by Genre")
# plt.show()
#
# # --- Ratings Analysis ---
# plt.figure(figsize=(8,4))
# sns.scatterplot(x="Rotten Tomatoes %", y="Audience score %", hue="Genre", data=df)
# plt.title("Audience vs Critic Ratings")
# plt.show()
#
# # --- Revenue Analysis ---
# top_grossing = df.nlargest(10, "Worldwide Gross")[["Film", "Worldwide Gross"]]
# print("\n=== Top 10 Highest Grossing Movies ===")
# print(top_grossing)
#
# plt.figure(figsize=(8,4))
# sns.barplot(x="Worldwide Gross", y="Film", data=top_grossing)
# plt.title("Top 10 Grossing Movies")
# plt.show()
#
# # --- Relationships ---
# plt.figure(figsize=(8,4))
# sns.scatterplot(x="Audience score %", y="Profitability", data=df)
# plt.title("Profitability vs Audience Score")
# plt.show()
#
# plt.figure(figsize=(6,5))
# sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()

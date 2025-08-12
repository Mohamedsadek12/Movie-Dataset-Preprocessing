import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('movies.csv')

#--------------------
# 1- Data Information
#--------------------

separate = "-"*80

print(f"Dataset Shape:\n{df.shape}")
print(separate)
print(f"First 5 Rows:\n{df.head()}")
print(separate)
print(f"Data Types:\n{df.dtypes}")
print(separate)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Numeric columns:\n{numerical_cols}")
print(f"Categorical columns:\n{categorical_cols}")
print(separate)
print(f"Missing Values:\n{df.isna().sum()}")
print(separate)
print(f"Dataset statistic Information:\n{df.describe()}")
print(separate)

#-----------------------
# 2- Preprocessing Steps
#-----------------------

df['Genre'] = df['Genre'].fillna('Unknown')
df['Lead Studio'] = df['Lead Studio'].fillna('Unknown')
df["Year"] = df["Year"].fillna(df["Year"].mode()[0])

df["Worldwide Gross"] = pd.to_numeric(df["Worldwide Gross"].replace('[$]', '', regex=True).str.strip()).astype(float)

fill_with_median = ["Worldwide Gross", "Profitability"]
fill_with_mean = ["Rotten Tomatoes %", "Audience score %"]
# Fill with median
for col in fill_with_median:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill with mean
for col in fill_with_mean:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())



df['Genre'] = df['Genre'].replace({'Comdy': 'Comedy', 'Romence': 'Romance', 'romance': 'Romance'})
df['Year'] = df['Year'].astype(int)

# Handle Duplicates

print("Number of duplicate 'Film' entries:", df['Film'].duplicated().sum())
df.drop_duplicates(subset='Film', keep='first', inplace=True)
print("Shape of the dataset after removing duplicates:", df.shape)
print(separate)


# Feature Engineering
df["Decade"] = (df["Year"] // 10 * 10).astype(str) + "s"
df["Is_Independent"] = df["Lead Studio"].str.lower().eq("independent")
df["Critic_Audience_Gap"] = df["Audience score %"] - df["Rotten Tomatoes %"]
df["ROI_Category"] = df["Profitability"].apply(lambda x: "Low" if (3.0 <= x < 6.0) else("Medium" if (6.0 <= x < 9.0) else "High"))

# Outlier Detection
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

# Final check of the preprocessed data
print("\nPreprocessed data info:")
print(df.info())
print(separate)

# 4- EDA Focus Areas: Visualizations and Tables

sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-pastel')

# A. Basic Distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribution of Key Numeric Variables', fontsize=18, color='white')
sns.histplot(df['Audience score %'], bins=15, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Audience Score % Distribution', color='white')
sns.histplot(df['Rotten Tomatoes %'], bins=15, kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Rotten Tomatoes % Distribution', color='white')
sns.histplot(df['Profitability'], bins=20, kde=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Profitability Distribution', color='white')
sns.histplot(df['Worldwide Gross'], bins=20, kde=True, ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Worldwide Gross Distribution', color='white')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('numeric_distribution_histograms.png')
plt.close()

# B. Profitability Analysis
avg_profitability_by_genre = df.groupby('Genre')['Profitability'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 7))
sns.barplot(x=avg_profitability_by_genre.index, y=avg_profitability_by_genre.values, hue=avg_profitability_by_genre.index, palette='viridis', legend=False)
plt.title('Average Profitability by Genre', color='white', fontsize=16)
plt.xticks(rotation=45, ha='right', color='white')
plt.ylabel('Average Profitability', color='white')
plt.tight_layout()
plt.savefig('avg_profitability_by_genre.png')
plt.close()

avg_profitability_by_studio = df.groupby('Lead Studio')['Profitability'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 7))
sns.barplot(x=avg_profitability_by_studio.index, y=avg_profitability_by_studio.values, hue=avg_profitability_by_studio.index, palette='plasma', legend=False)
plt.title('Top 10 Lead Studios by Average Profitability', color='white', fontsize=16)
plt.xticks(rotation=45, ha='right', color='white')
plt.ylabel('Average Profitability', color='white')
plt.tight_layout()
plt.savefig('avg_profitability_by_studio.png')
plt.close()

profitability_by_year = df.groupby('Year')['Profitability'].mean()
plt.figure(figsize=(12, 7))
sns.lineplot(x=profitability_by_year.index, y=profitability_by_year.values, marker='o', color='lightgreen')
plt.title('Average Profitability Trend Over Years', color='white', fontsize=16)
plt.xticks(profitability_by_year.index, color='white')
plt.ylabel('Average Profitability', color='white')
plt.tight_layout()
plt.savefig('profitability_trend_over_years.png')
plt.close()

#  C. Ratings Analysis
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Audience score %', y='Rotten Tomatoes %', data=df)
plt.title('Audience vs. Critic Ratings', color='white', fontsize=16)
plt.xlabel('Audience score %', color='white')
plt.ylabel('Rotten Tomatoes %', color='white')
plt.tight_layout()
plt.savefig('audience_vs_critic_ratings.png')
plt.close()

avg_ratings_by_genre = df.groupby('Genre')[['Audience score %', 'Rotten Tomatoes %']].mean().sort_values(by='Audience score %', ascending=False)
avg_ratings_by_genre.plot(kind='bar', figsize=(12, 7), color=['skyblue', 'salmon'])
plt.title('Average Audience vs. Critic Ratings by Genre', color='white', fontsize=16)
plt.xticks(rotation=45, ha='right', color='white')
plt.ylabel('Average Rating %', color='white')
plt.tight_layout()
plt.savefig('avg_ratings_by_genre.png')
plt.close()

avg_ratings_by_studio = df.groupby('Lead Studio')[['Audience score %', 'Rotten Tomatoes %']].mean().sort_values(by='Audience score %', ascending=False).head(10)
avg_ratings_by_studio.plot(kind='bar', figsize=(12, 7), color=['skyblue', 'salmon'])
plt.title('Top 10 Lead Studios by Average Ratings', color='white', fontsize=16)
plt.xticks(rotation=45, ha='right', color='white')
plt.ylabel('Average Rating %', color='white')
plt.tight_layout()
plt.savefig('avg_ratings_by_studio.png')
plt.close()

# # D. Revenue Analysis
# avg_gross_by_genre = df.groupby('Genre')['Worldwide Gross'].mean().sort_values(ascending=False)
# plt.figure(figsize=(12, 7))
# sns.barplot(x=avg_gross_by_genre.index, y=avg_gross_by_genre.values, hue=avg_gross_by_genre.index, palette='coolwarm', legend=False)
# plt.title('Average Worldwide Gross by Genre', color='white', fontsize=16)
# plt.xticks(rotation=45, ha='right', color='white')
# plt.ylabel('Average Worldwide Gross (in millions USD)', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# avg_gross_by_studio = df.groupby('Lead Studio')['Worldwide Gross'].mean().sort_values(ascending=False).head(10)
# plt.figure(figsize=(12, 7))
# sns.barplot(x=avg_gross_by_studio.index, y=avg_gross_by_studio.values, hue=avg_gross_by_studio.index, palette='cubehelix', legend=False)
# plt.title('Top 10 Lead Studios by Average Worldwide Gross', color='white', fontsize=16)
# plt.xticks(rotation=45, ha='right', color='white')
# plt.ylabel('Average Worldwide Gross (in millions USD)', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# avg_gross_by_year = df.groupby('Year')['Worldwide Gross'].mean()
# plt.figure(figsize=(12, 7))
# sns.lineplot(x=avg_gross_by_year.index, y=avg_gross_by_year.values, marker='o', color='gold')
# plt.title('Average Worldwide Gross Trend Over Years', color='white', fontsize=16)
# plt.xticks(avg_gross_by_year.index, color='white')
# plt.ylabel('Average Worldwide Gross (in millions USD)', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# # E. Relationship Analysis
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='Audience score %', y='Profitability', data=df)
# plt.title('Profitability vs. Audience Score %', color='white', fontsize=16)
# plt.xlabel('Audience score %', color='white')
# plt.ylabel('Profitability', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='Rotten Tomatoes %', y='Profitability', data=df)
# plt.title('Profitability vs. Rotten Tomatoes %', color='white', fontsize=16)
# plt.xlabel('Rotten Tomatoes %', color='white')
# plt.ylabel('Profitability', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# numeric_cols = ['Audience score %', 'Profitability', 'Rotten Tomatoes %', 'Worldwide Gross', 'Year', 'Critic-Audience Gap']
# correlation_matrix = df[numeric_cols].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix of Numeric Variables', color='white', fontsize=16)
# plt.xticks(rotation=45, ha='right', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# # F. Multi-Dimensional Insights
# top_studios = df['Lead Studio'].value_counts().head(5).index.tolist()
# top_genres = df['Genre'].value_counts().head(5).index.tolist()
# df_filtered = df[(df['Lead Studio'].isin(top_studios)) & (df['Genre'].isin(top_genres))]
# plt.figure(figsize=(15, 10))
# sns.barplot(x='Genre', y='Profitability', hue='Lead Studio', data=df_filtered, palette='muted')
# plt.title('Profitability by Genre and Top 5 Studios', color='white', fontsize=16)
# plt.xticks(rotation=45, ha='right', color='white')
# plt.ylabel('Average Profitability', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# avg_gap_by_genre = df.groupby('Genre')['Critic-Audience Gap'].mean().sort_values(ascending=False)
# plt.figure(figsize=(12, 7))
# sns.barplot(x=avg_gap_by_genre.index, y=avg_gap_by_genre.values, hue=avg_gap_by_genre.index, palette='RdYlGn', legend=False)
# plt.title('Average Audience-Critic Gap by Genre', color='white', fontsize=16)
# plt.axhline(0, color='white', linestyle='--')
# plt.xticks(rotation=45, ha='right', color='white')
# plt.ylabel('Average Gap (Audience - Critic)', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()
#
# avg_profitability_by_decade_genre = df.groupby(['Decade', 'Genre'])['Profitability'].mean().reset_index()
# plt.figure(figsize=(12, 7))
# sns.barplot(x='Decade', y='Profitability', hue='Genre', data=avg_profitability_by_decade_genre.sort_values(by='Profitability', ascending=False).head(10), palette='Paired')
# plt.title('Top Profitable Genres Over Decades', color='white', fontsize=16)
# plt.xticks(rotation=0, color='white')
# plt.ylabel('Average Profitability', color='white')
# plt.tight_layout()
# plt.show()
# plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 確認 output 資料夾存在，沒有就自動創
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 讀取資料
file_path = 'student-mat.csv'
df = pd.read_csv(file_path)

# 確認資料
print(df.head())

# --- 繪圖部分 ---

# 第一張圖：Walc vs G3 Scatter Plot by Sex
plt.figure(figsize=(8, 6))
colors = {'M': 'blue', 'F': 'red'}
markers = {'M': 'o', 'F': 's'}

for sex in df['sex'].unique():
    sub_df = df[df['sex'] == sex]
    plt.scatter(sub_df['Walc'], sub_df['G3'], 
                c=colors.get(sex, 'gray'), label=f'Sex: {sex}', 
                alpha=0.6, marker=markers.get(sex, 'x'))

    # 計算回歸線
    if len(sub_df) > 1:
        slope, intercept = np.polyfit(sub_df['Walc'], sub_df['G3'], 1)
        x_vals = np.array([sub_df['Walc'].min(), sub_df['Walc'].max()])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color=colors.get(sex, 'gray'), linestyle='--')

plt.title('Scatter Plot with Regression Lines: Walc vs G3 by Sex')
plt.xlabel('Weekend Alcohol Consumption (Walc)')
plt.ylabel('Final Grade (G3)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/sex_walc_g3_scatter_regression.png')
plt.close()

# 第二三張圖：分布圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(data=df, x='Walc', hue='sex', multiple='dodge', shrink=0.8, palette='Set1', bins=5, ax=axes[0])
axes[0].set_title('Distribution of Weekend Alcohol Consumption (Walc) by Sex')
axes[0].set_xlabel('Walc')
axes[0].set_ylabel('Count')
axes[0].grid(True)

sns.histplot(data=df, x='G3', hue='sex', multiple='dodge', shrink=0.8, palette='Set2', bins=20, ax=axes[1])
axes[1].set_title('Distribution of Final Grades (G3) by Sex')
axes[1].set_xlabel('G3')
axes[1].set_ylabel('Count')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f'{output_dir}/sex_walc_g3_distribution.png')
plt.close()

# 第四第五張圖：分男生女生的矩陣圖
for sex in df['sex'].unique():
    plt.figure(figsize=(5, 4))
    sub_df = df[df['sex'] == sex]
    corr = sub_df[['Walc', 'G3']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title(f'Correlation Matrix: Walc and G3 ({sex})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix_{sex}.png')
    plt.close()

print(f"✅ 圖片已經輸出到資料夾：{output_dir}/")

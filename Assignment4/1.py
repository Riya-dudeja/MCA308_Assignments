import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
df = pd.DataFrame(np.random.randint(1, 201, size=(100, 30)))

df = df.applymap(lambda x: np.nan if 10 <= x <= 60 else x)

print('Count of NAs in each row:')
print(df.isna().sum(axis=1))
print('\nCount of NAs in each column:')
print(df.isna().sum(axis=0))

df_filled = df.fillna(df.mean())

plt.figure(figsize=(35, 35))
sns.heatmap(df_filled, cmap='viridis')
plt.title('Heatmap of Dataset (NAs replaced with column mean)')
plt.show()

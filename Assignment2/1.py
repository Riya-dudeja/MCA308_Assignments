import pandas as pd
data = {
    "Tid": range(1, 11),
    "Refund": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
    "Taxable Income (K)": [125, 100, 70, 120, 95, 60, 220, 85, 75, 90],
    "Cheat": ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes"]
}
df = pd.DataFrame(data)
print(df)

rows = df.loc[[0, 4, 7, 8]]
print(rows)
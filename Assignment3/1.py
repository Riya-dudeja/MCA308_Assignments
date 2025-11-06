import pandas as pd

feature_types = {
    'Age': ('Continuous', 'Ratio'),
    'Education': ('Discrete', 'Ordinal'),
    'Occupation': ('Discrete', 'Nominal'),
    'Gender': ('Discrete', 'Nominal'),
    'MaritalStatus': ('Discrete', 'Nominal'),
    'HomeOwnerFlag': ('Discrete', 'Nominal'),
    'NumberCarsOwned': ('Discrete', 'Ratio'),
    'NumberChildrenAtHome': ('Discrete', 'Ratio'),
    'TotalChildren': ('Discrete', 'Ratio'),
    'YearlyIncome': ('Continuous', 'Ratio')
}
df = pd.read_csv("AWCustomers.csv")
df['BirthDate'] = pd.to_datetime(df['BirthDate'], dayfirst=True)
df['Age'] = pd.to_datetime('today').year - df["BirthDate"].dt.year

selected_features = list(feature_types.keys())
df_selected = df[selected_features]
print(df_selected)

print("| Attribute              | Data Type  | Subtype   |")
print("|------------------------|------------|-----------|")
for feature, (dtype, subtype) in feature_types.items():
    print(f"| {feature:22} | {dtype:10} | {subtype:9} |")

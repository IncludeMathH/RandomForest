import pandas as pd

df = pd.read_csv("source/wine.txt")
df = df[df['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)

index = [1, 3, 5]
df1 = df.loc[index]
for _, row in df1.iterrows():
    print(_, row)

print(df1.head())
print(df.loc[:3, 'label'])

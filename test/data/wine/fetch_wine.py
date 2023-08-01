import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
print("Data fetched")
target_feature = "class"

data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['class'] = wine.target

data_train, data_test = train_test_split(
    data, test_size=40, random_state=1, stratify=data[target_feature]
)

# Don't write out the row indices to the CSV.....
print("Saving to files")
data_train.to_parquet("./train/wine_train.parquet", index=False)
data_test.to_parquet("./test/wine_test.parquet", index=False)

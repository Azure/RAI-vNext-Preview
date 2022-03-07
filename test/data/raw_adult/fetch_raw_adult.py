import arff
from collections import OrderedDict
from contextlib import closing
import gzip
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import time

_categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def fetch_census_dataset():
    """Fetch the Adult Census Dataset.
    This uses a particular URL for the Adult Census dataset. The code
    is a simplified version of fetch_openml() in sklearn.
    The data are copied from:
    https://openml.org/data/v1/download/1595261.gz
    (as of 2021-03-31)
    """
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve

    filename = "1595261.gz"
    data_url = "https://rainotebookscdn.blob.core.windows.net/datasets/"

    remaining_attempts = 5
    sleep_duration = 10
    while remaining_attempts > 0:
        try:
            urlretrieve(data_url + filename, filename)

            http_stream = gzip.GzipFile(filename=filename, mode="rb")

            with closing(http_stream):

                def _stream_generator(response):
                    for line in response:
                        yield line.decode("utf-8")

                stream = _stream_generator(http_stream)
                data = arff.load(stream)
        except Exception as exc:  # noqa: B902
            remaining_attempts -= 1
            print(
                "Error downloading dataset from {} ({} attempt(s) remaining)".format(
                    data_url, remaining_attempts
                )
            )
            print(exc)
            time.sleep(sleep_duration)
            sleep_duration *= 2
            continue
        else:
            # dataset successfully downloaded
            break
    else:
        raise Exception("Could not retrieve dataset from {}.".format(data_url))

    attributes = OrderedDict(data["attributes"])
    arff_columns = list(attributes)

    raw_df = pd.DataFrame(data=data["data"], columns=arff_columns)

    target_column_name = "class"
    target = raw_df.pop(target_column_name)
    for col_name in _categorical_columns:
        dtype = pd.api.types.CategoricalDtype(attributes[col_name])
        raw_df[col_name] = raw_df[col_name].astype(dtype, copy=False)

    result = Bunch()
    result.data = raw_df
    result.target = (data.target == '>50K') * 1

    return result


def process_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Processes data to impute missing values
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="category")),
            ("cat", categorical_transformer, selector(dtype_include="category")),
        ]
    )

    train_array = preprocessor.fit_transform(train_df)
    # Transformer reorders the columns, and get_feature_names_out() not universally supported
    num_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = train_df.select_dtypes(exclude=np.number).columns.tolist()
    processed_train = pd.DataFrame(data=train_array, columns=num_cols+cat_cols)
    for c in train_df.columns:
        processed_train[c] = processed_train[c].astype(train_df[c].dtype)

    test_array = preprocessor.transform(test_df)
    processed_test = pd.DataFrame(data=test_array, columns=num_cols+cat_cols)
    for c in train_df.columns:
        processed_test[c] = processed_test[c].astype(train_df[c].dtype)

    return processed_train, processed_test


adult_census = fetch_census_dataset()

target_feature_name = "income"
full_data = adult_census.data.copy()
full_data[target_feature_name] = adult_census.target

print(full_data.columns)

data_train, data_test = train_test_split(
    full_data,
    test_size=1000,
    random_state=96132,
    stratify=full_data[target_feature_name],
)

# RAIInsights has problems with missing data, so impute for now
train_out, test_out = process_data(data_train, data_test)

# Don't write out the row indices to the Parquet.....
print("Saving to files")
train_out.to_parquet("raw_adult_train.parquet", index=False)
test_out.to_parquet("raw_adult_test.parquet", index=False)

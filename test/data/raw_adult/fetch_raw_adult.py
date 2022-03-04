import arff
from collections import OrderedDict
from contextlib import closing
import gzip
import pandas as pd
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
    result.target = target

    return result


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

# Don't write out the row indices to the Parquet.....
print("Saving to files")
data_train.to_parquet("raw_adult_train.parquet", index=False)
data_test.to_parquet("raw_adult_test.parquet", index=False)

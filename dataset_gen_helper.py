import sys
from pandas import read_csv, DataFrame
import datasets
from tqdm import tqdm
import dask.dataframe as dd


tqdm.pandas()

columns = []

# Merge all column entries as one single string using key-value pairs
def create_key_value_pairs_str(row):
    return ', '.join([f"{column}: {row[column]}" for column in columns])


def operation(df):
    global columns
    # Obtain the output label which shall be predicted by the LLM
    df = df.rename(columns={'Label': 'output'})

    # Remove the prediction labels from the data which will be encoded
    del df["Attack"]

    # Merge all remaining columns
    print("Merging all columns in parralel.")
    columns = df.columns.drop("output")
    df['input'] = df.apply(create_key_value_pairs_str, axis=1)

    return df[['input', 'output']]

# Helper Function for all datasets
def encode_dataset(dataset_name):
    print(f"Opening ./data_raw/{dataset_name}.csv")
    df = read_csv(f"./data_raw/{dataset_name}.csv")

    df = dd.from_pandas(df,npartitions=10)

    df.pipe(operation)

    return df


def save_to_arrow_disk(df, dataset_name):
    print(f"Saving processed dataset to disk: ./{dataset_name}/")
    df = df.compute()

    finetuning_dataset = datasets.Dataset.from_pandas(df)
    finetuning_dataset = finetuning_dataset.class_encode_column("output")
    finetuning_dataset = finetuning_dataset.train_test_split(test_size=0.3, seed=123, stratify_by_column="output")
    finetuning_dataset.save_to_disk(f"./{dataset_name}/")
    return finetuning_dataset


if __name__ == "__main__":
    DATASET_NAME = sys.argv[1]

    df = encode_dataset(DATASET_NAME)
    save_to_arrow_disk(df, DATASET_NAME)

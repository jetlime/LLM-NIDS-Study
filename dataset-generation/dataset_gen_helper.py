import sys
import datasets
import dask.dataframe as dd 
from dask.diagnostics import ProgressBar
from tqdm import tqdm

tqdm.pandas()

columns = []

# Merge all column entries as one single string using key-value pairs
def create_key_value_pairs_str(row):
    return ', '.join([f"{column}: {row[column]}" for column in columns])


def operation(df):
    global columns
    # Obtain the output label which shall be predicted by the LLM
    df = df.rename(columns={'Label': 'output'})

    # Merge all remaining columns
    print("Merging all columns in parralel.")
    columns = df.columns.drop("output")
    columns = columns.drop("Attack")
    df['input'] = df.apply(create_key_value_pairs_str, axis=1)

    return df[['input', 'output', 'Attack']]

# Helper Function for all datasets
def encode_dataset(DATASET_NAME):
    print(f"Opening ../data_raw/{DATASET_NAME}.csv")
    # df = read_csv(f"./data_raw/{DATASET_NAME}.csv")
    ProgressBar().register()
    df = dd.read_csv(f"../data_raw/{DATASET_NAME}.csv")
    # Both datasets shall only publish the testing set
    # if DATASET_NAME == "NF-CSE-CIC-IDS2018-v2" or DATASET_NAME == "NF-UQ-NIDS-v2":
    #     # FIXME: DASK ML does not currently support stratify option, PR is open for 4,5 years now.... 
    #     df, _ = train_test_split(df, test_size=0.95, random_state=42)

    # df = dd.from_pandas(df,npartitions=10)
    ProgressBar().register()
    df = df.pipe(operation)
    return df


def process_arrow_data(df, DATASET_NAME):
    df = df.compute()
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.class_encode_column("output")
    dataset = dataset.class_encode_column("Attack")
    # if DATASET_NAME == "NF-UNSW-NB15-v2":
    dataset = dataset.train_test_split(test_size=0.05, seed=123, stratify_by_column="Attack")
    dataset.save_to_disk(f"./{DATASET_NAME}/")

if __name__ == "__main__":
    DATASET_NAME = sys.argv[1]

    df = encode_dataset(DATASET_NAME)
    process_arrow_data(df, DATASET_NAME)

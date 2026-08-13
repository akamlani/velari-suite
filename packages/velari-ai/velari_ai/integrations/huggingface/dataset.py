from datasets import Dataset, DatasetDict

def load_from_dict(data_dict: dict) -> Dataset:
    """Load a HuggingFace Dataset from a dictionary."""
    return Dataset.from_dict(data_dict)

def load_from_list(data_list: list) -> Dataset:
    """Load a HuggingFace Dataset from a list of dictionaries."""
    return Dataset.from_list(data_list)

def generate_dataset_dict(train_data: list, test_data: list) -> DatasetDict:
    """Generate a HuggingFace DatasetDict from training and testing data."""
    return DatasetDict({
        "train": load_from_list(train_data),
        "test":  load_from_list(test_data)
    })

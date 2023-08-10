import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, Optional, List


def create_train_test_split(
    features: np.ndarray,
    stratify: Union[np.ndarray, List],
    labels: Optional[Union[np.ndarray, List]] = None,
    test_size: float = 0.2,
    random_state: int = 10,
) -> Union[
    Tuple[np.ndarray, np.ndarray], 
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Create a train-test split for features and labels.

    Args:
        features: Input feature data.
        stratify: Stratification target to ensure balanced classes.
        labels: Target labels (optional).
        test_size: Proportion of data to allocate for the test set.
        random_state: Random seed for reproducibility.

    Returns:
        If labels are None, returns a tuple of train and test feature arrays.
        If labels are provided, returns a tuple of train feature, train label, test feature, and test label arrays.
    """
    train_features = None
    train_labels = None
    test_features = None
    test_labels = None

    if labels is None:
        train_features, test_features = train_test_split(
            features, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return np.array(train_features), np.array(test_features)
    else:
        train_features, train_labels, test_features, test_labels = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        return (
            np.array(train_features),
            np.array(train_labels),
            np.array(test_features),
            np.array(test_labels),
        )


def pickle_python_object(obj: any, save_path: str) -> bool:
    """
    Pickle a Python object and save it to a file.

    Args:
        obj: Python object to pickle.
        save_path: Filepath to save the pickled object.

    Returns:
        True if the pickling and saving process was successful.
    """
    with open(save_path, "wb") as file:
        pickle.dump(obj, file)
        return True


def unpickle_python_object(pickle_filepath: str) -> any:
    """
    Unpickle a Python object from a pickle file.

    Args:
        pickle_filepath: Filepath of the pickled object.

    Returns:
        The unpickled Python object.
    """
    with open(pickle_filepath, "rb") as file:
        return pickle.load(file)

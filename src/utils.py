import os
import sys
import pickle

from src.exception import CustomException


# ================= SAVE OBJECT =================
def save_object(file_path: str, obj) -> None:
    """
    Save Python object (like model or preprocessor) using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)

        # Create directory if not exists
        os.makedirs(dir_path, exist_ok=True)

        # Save object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ================= LOAD OBJECT =================
def load_object(file_path: str):
    """
    Load saved Python object from pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)






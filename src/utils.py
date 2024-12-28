import pandas as pd
import numpy as np
import os

def load_data(path: str, sheet_name) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)
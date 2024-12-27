import pandas as pd
import numpy as np
import os

def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)
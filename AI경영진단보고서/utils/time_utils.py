from datetime import datetime
import pandas as pd

def get_access_time():
    """Return the current access time in a consistent datetime format."""
    return pd.to_datetime(datetime.now())

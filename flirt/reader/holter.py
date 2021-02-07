from datetime import datetime

import pandas as pd
from ishneholterlib import Holter

__all__ = [
    'get_starttime_from_holter',
    'read_holter_ibi_file_into_df',
]


def get_starttime_from_holter(holter_path: str) -> datetime:
    """
    Reads the start time of a holter file

    Parameters
    ----------
    holter_path : str
        path to the holter file

    Returns
    -------
    datetime
        the start datetime of the Holter file
    """
    holter = Holter(holter_path)
    return datetime.combine(holter.record_date, holter.start_time)


def read_holter_ibi_file_into_df(start_time: datetime, ibi_file: str) -> pd.DataFrame:
    """
    Reads a Holter IBI file (from Lifecard CF) into a pandas DataFrame

    Parameters
    ----------
    start_time : datetime
        starttime of the file (should be retrieved from Holter file)
    ibi_file : str
        path to the IBI file

    Returns
    -------
    pd.DataFrame
        DataFrame containing the IBIs
    """
    ibi = pd.read_csv(ibi_file, skiprows=1, names=['ibi', 'type'], delim_whitespace=True)

    ibi.index = pd.to_datetime((ibi['ibi'] * 1000 + start_time.timestamp() * 1000).map(int), utc=True, unit='ms')
    ibi.index = ibi.index.tz_convert(tz=start_time.tzinfo)

    ibi.drop(ibi[ibi['type'] != 'N'].index, inplace=True)
    ibi['ibi_diff'] = ((ibi['ibi'] - ibi['ibi'].shift(1)) * 1000)
    ibi.dropna(subset=['ibi_diff'], inplace=True)
    ibi['ibi'] = ibi['ibi_diff'].map(int)
    ibi.drop(columns=['ibi_diff', 'type'], inplace=True)

    ibi.index.name = 'datetime'
    return ibi

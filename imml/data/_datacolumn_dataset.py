import numpy as np
import pandas as pd
from typing import Union, Any
from torch.utils.data import Dataset

from ._dataframe_dataset import DataFrameDataset

class DataColumnDataset(Dataset):
    
    def __init__(
        self,
        frameset:DataFrameDataset,
        columns:Union[str,list[str]],
        remove_na:bool=True,
        unique:bool=True
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        self._dataset = frameset.df[columns].values.flatten()
        self._remove_na = remove_na
        self._unique = unique
        if remove_na:
            self._dataset = self._dataset[~pd.isna(self._dataset)]
        if unique:
            self._dataset = np.unique(self._dataset)
        self._column = columns
        self._parent_name = frameset.config['name']
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __repr__(self) -> str:
        _status = []
        if self._remove_na:
            _status.append('NA Removed')
        if self._unique:
            _status.append('Duplicate Dropped')
        _status = '; '.join(_status)
        _columns = ', '.join(self._column)
        return f'Column {_columns} of {self._parent_name} ({_status})'

    def __getitem__(self, key) -> Any:
        return self._dataset[key]
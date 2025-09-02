
import os
import json
import pandas as pd
from typing import Union, Optional, Tuple
from torch.utils.data import Dataset

from ._data_obj import DataObject


class DataFrameDataset(Dataset):
    '''Universal DataFrame Dataset
    
    Parameters
    ----------
    dataframe : DataFrame | str
        A dataframe or a path to a acceptable dataset path.
        If it is a dataframe, it should provided with
        configuration dictionary.
    config : dict | `None` = `None`
        A configuration dictionary for the dataframe.
    
    Configurations
    --------------
    A configuration dictionary for the dataframe.
    - index_col : int | str | `None` = `None`  
        The index column of the dataframe.
    - sep : str | `None` = `None`  
        The separater for the dataframe.
    - file_name : str = `data.csv`  
        The dataframe file name.
    - data_columns : List[str] | `None` = `None`  
        The data columns. If it is None, then all columns.
    - label_columns : List[str] | `None` = `None`  
        The label columns. If it is None, then all columns.
    '''
    def __init__(
        self,
        dataframe:Union[pd.DataFrame, str],
        config:Optional[dict]=None
    ) -> None:
        self._df, self._config = \
            self.__load_dataframe(
                dataframe=dataframe,
                config=config,
            )
            
    def __len__(self) -> int:
        return len(self._df)
    
    def __repr__(self) -> str:
        return f'{self.config["name"]} ({len(self)} samples)'
    
    def __getitem__(self, key:int) -> DataObject:
        _obj = self._df.iloc[key].to_dict()
        return DataObject(**_obj)
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @property
    def config(self) -> dict:
        return self._config
        
    def __load_dataframe(
        self,
        dataframe:Union[pd.DataFrame, str],
        config:Optional[dict]=None
    ) -> Tuple[pd.DataFrame, dict]:
        '''(Private) Load and format dataframe with configuration dict.
        '''
        if isinstance(dataframe, pd.DataFrame):
            # Load with dataframe and config pair
            _df = dataframe
            _config = config
        else:
            # Load from a dataset path
            _df, _config = \
                self.__read_dataset(
                    path = dataframe
                )
        return _df, _config
    
    default_config = \
        dict(
            name='DataFrame Dataset',
            index_col=None,
            sep=',',
            file_name='data.csv',
            data_columns=None,
            label_columns=None,
            desc='',
            seed='42',
        )
    def __read_dataset(
        self,
        path:str
    ) -> Tuple[pd.DataFrame, dict]:
        '''(Private) Load dataset config and dataframe from a given path.
        '''
        # Load configuration file
        _config_path = os.path.join(path, 'config.json')
        with open(_config_path, 'r') as _config_file:
            _config = json.load(_config_file)
        # Fill missing keys with default values
        for key, default_val in self.default_config.items():
            if key not in _config:
                _config[key] = default_val
        # Load dataframe by configuration
        _df_path = os.path.join(path, _config['file_name'])
        _df = \
            pd.read_csv(
                filepath_or_buffer=_df_path,
                sep=_config['sep'],
                index_col=_config['index_col'],
            )
        return _df, _config
    
    def save(
        self,
        path:str,
        name:Optional[str]=None,
        file_name:Optional[str]=None,
        desc:Optional[str]=None,
    ) -> None:
        _config = self._config
        if name is not None:
            _config['name'] = name
        if file_name is not None:
            _config['file_name'] = file_name
        if desc is not None:
            _config['desc'] = desc
        
        os.makedirs(path, exist_ok=True)
        _config_path = os.path.join(path, 'config.json')
        with open(_config_path, 'w') as _config_file:
            json.dump(_config, _config_file, indent=2)
        
        _df_path = os.path.join(path, file_name)
        self._df.to_csv(_df_path, sep=_config['sep'], index=False)
  
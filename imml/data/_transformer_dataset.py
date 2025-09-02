from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Union, Optional, Any, Callable

from ._datacolumn_dataset import DataColumnDataset
from ._dataframe_dataset import DataFrameDataset

class TransformerDataset(Dataset):
    
    def __init__(
        self,
        dataset:Union[DataFrameDataset, DataColumnDataset],
        columns:Optional[Union[list[str], str]]=None,
        mapping:dict={},
        processor:Callable=lambda x:x,
        cache:bool=True,
    ) -> None:
        self.dataset = dataset
        if columns is None:
            columns = self.dataset.df.columns.values
        self.columns = columns
        if isinstance(columns, str):
            columns = [columns]
        self.mapping = {col: mapping.get(col, col) for col in columns}
        self.processor = processor
        
        if cache:
            self.caches = [self.processor(_sample) for _sample in tqdm(self.dataset, desc=f'caching {self.dataset}')]
        else:
            self.caches = None
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __repr__(self) -> str:
        return f'TransformerDataset: {", ".join(self.columns)}'

    def __getitem__(self, key) -> Any:
        if self.caches is not None:
            return self.caches[key]
        _sample = self.dataset[key]
        return self.processor(_sample)
        # if isinstance(self.dataset, DataFrameDataset):
        #     _processed_sample = {}
        #     for _col in self.columns:
        #         _data = _sample[_col]
        #         if _col in self.processors:
        #             _data = self.processors[_col](_data)
        #             _data = {f'{_col}_{k}':v for k,v in _data.items()}
        #         else:
        #             _data = {_col:_data}
        #         _processed_sample.update(_data)
        #     return _processed_sample
        # else:
        #     if self.columns in self.processors:
        #         _sample = self.processors[self.columns](_sample)
        #     return {self.mapping.get(k,k):v for k, v in _sample.items()}
    
        
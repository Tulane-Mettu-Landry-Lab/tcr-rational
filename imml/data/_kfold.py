import numpy as np
from typing import Optional
from ._dataframe_dataset import DataFrameDataset
from ._dataframe_index import DataFrameIndexDataset
from ._traintest_split import TrainTestSplit


class KFold(object):
    def __init__(
        self,
        dataset:DataFrameDataset,
        k:int=5,
        seed:Optional[int]=None
    ) -> None:
        if seed is None:
            seed = dataset.config['seed']
        self.seed = seed
        self.k = k
        self.dataset = dataset
        self.folds = \
            self.__build_folds(
                dataset = self.dataset,
                k = self.k,
                seed = self.seed
            )
    
    def __build_folds(
        self,
        dataset:DataFrameDataset,
        k:int=5,
        seed:int=42
    ) -> list[TrainTestSplit]:
        _indices = np.arange(len(dataset.df))
        np.random.seed(seed)
        np.random.shuffle(_indices)
        _folds = []
        _fold_size = len(dataset)//k
        for i, _s in enumerate(range(0, len(dataset), _fold_size)):
            if i >= k-1:
                _test_indices = _indices[_s:]
                _train_indices = _indices[:_s]
            else:
                _test_indices = _indices[_s:_s+_fold_size]
                _train_indices = np.concatenate([_indices[:_s], _indices[_s+_fold_size:]])
            _folds.append(
                TrainTestSplit(
                    trainset=DataFrameIndexDataset(dataset, _train_indices),
                    testset=DataFrameIndexDataset(dataset, _test_indices),
                )
            )
        return _folds
    
    def __repr__(self) -> str:
        return f'{self.k}-Fold Datasets of {self.dataset.config["name"]} (seed = {self.seed})'
    
    def __len__(self) -> int:
        return self.k
    
    def __getitem__(self, key) -> TrainTestSplit:
        return self.folds[key]
    
    def __iter__(self) -> None:
        self.__iter_idx = 0
        return self
    
    def __next__(self) -> TrainTestSplit:
        if self.__iter_idx >= len(self):
            raise StopIteration
        else:
            self.__iter_idx += 1
            return self[self.__iter_idx - 1]
import numpy as np
from typing import Optional
from ._dataframe_dataset import DataFrameDataset
from ._dataframe_index import DataFrameIndexDataset

class TrainTestSplit(object):
    
    def __init__(
        self,
        dataset:DataFrameDataset=None,
        test_ratio:float=0.2,
        seed:Optional[int]=None,
        trainset:DataFrameDataset=None,
        testset:DataFrameDataset=None,
    ) -> None:
        if dataset is None:
            self.train = trainset
            self.test = testset
            self.test_ratio = len(self.test) / (len(self.train) + len(self.test))
            self.seed = self.train.config['seed']
        else:
            self.test_ratio = test_ratio
            if seed is None:
                self.seed = dataset.config['seed']
            else:
                self.seed = seed
            np.random.seed(self.seed)
            _indices = np.arange(len(self.dataset.df))
            np.random.shuffle(_indices)
            self.train = DataFrameIndexDataset(dataset, index=_indices[:int(1-test_ratio)])
            self.test = DataFrameIndexDataset(dataset, index=_indices[int(1-test_ratio):])
    
    def __repr__(self) -> str:
        return f'Train Test Split (train={len(self.train)}, test={len(self.test)}, test_ratio={self.test_ratio:.2f}, seed={self.seed})'
    
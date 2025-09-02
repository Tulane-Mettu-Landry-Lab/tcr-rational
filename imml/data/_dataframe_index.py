from ._dataframe_dataset import DataFrameDataset
class DataFrameIndexDataset(DataFrameDataset):
    def __init__(
        self,
        dataset:DataFrameDataset,
        index:list[int]
    ) -> None:
        dataframe = dataset.df.iloc[index]
        config = dataset.config
        super().__init__(dataframe, config)
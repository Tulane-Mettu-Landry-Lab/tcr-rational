import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from imml.data import DataFrameDataset
from imml.configs import IMMLConfigurationGroup
from utils import TCRRepExperiment



output_dir='projects/representations/cdr3_only_binder'
config_path = 'projects/representations/configs/cdr3_only_binder.json'
tcr_dataset_path = 'data/TCRRNL'
immrep23_dataset_path = 'data/IMMREP23L'

if __name__ == '__main__':
    TCRRepExperiment(
        'TCRpMHC-MLM-CA-Tasks',
        configs=IMMLConfigurationGroup.from_config(config_path),
        output_dir=output_dir,
        log_level='error',
        overwrite_output_dir=True,
    )(
        tcrr=DataFrameDataset(tcr_dataset_path),
        immrep23=DataFrameDataset(immrep23_dataset_path),
    )
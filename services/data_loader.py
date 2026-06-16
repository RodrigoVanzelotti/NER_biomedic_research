from datasets import load_from_disk, load_dataset
from pathlib import Path

from services.config.config_service import ConfigService

config = ConfigService().get()

def load_data(save_to_disk_on_fetch: bool = True):
    dataset_folder_path = Path(config.data.data_folder) / 'biored'

    if dataset_folder_path.exists():
        print('Retrieving data from disk')
        dataset = load_from_disk(str(dataset_folder_path))

    else:
        print('No data on disk. Retrieving online...')
        dataset = load_dataset("bigbio/biored", trust_remote_code=True)
        if save_to_disk_on_fetch: 
            print('Saving data to disk...')
            dataset.save_to_disk(str(dataset_folder_path))
    
    return dataset

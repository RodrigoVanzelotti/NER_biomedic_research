from datasets import load_from_disk, load_dataset
from typing import Union
from dotenv import dotenv_values
from pathlib import Path


env_variables = dotenv_values(Path('config') / '.env')
dataset_folder_path = Path(env_variables['DATA_DATASET_FOLDER']) / 'biored'

def load_data(save_to_disk_on_fetch: bool = True):
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

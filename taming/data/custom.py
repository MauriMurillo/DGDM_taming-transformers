import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex



class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomClasegTrain(CustomBase):
    def __init__(self, siz, dataset_num, fold, dataset_name=None):
        super().__init__()
        PREPROCESSED_ROOT = os.getenv('PREPROCESSED_ROOT', None)
        dataset_path = f"{PREPROCESSED_ROOT}/Dataset_"
        if dataset_name is not None:
            dataset_path += f"{dataset_name}_"
        dataset_path += f"{dataset_num}/fold_{fold}/imagesTr/"
        files = os.listdir(dataset_path)
        self.data = NumpyPaths(paths=files, size=size, random_crop=False)


class CustomClasegTest(CustomBase):
    def __init__(self, size, dataset_num, fold, dataset_name=None):
        super().__init__()
        PREPROCESSED_ROOT = os.getenv('PREPROCESSED_ROOT', None)
        dataset_path = f"{PREPROCESSED_ROOT}/Dataset_"
        if dataset_name is not None:
            dataset_path += f"{dataset_name}_"
        dataset_path += f"{dataset_num}/fold_{fold}/imagesVal/" 
        files = os.listdir(dataset_path)
        self.data = NumpyPaths(paths=files, size=size, random_crop=False)
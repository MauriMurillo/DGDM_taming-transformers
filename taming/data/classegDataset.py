import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import glob
from taming.data.sflckr import SegmentationBase # for examples included in repo

# With semantic map and scene label
class ClassegDataset(Dataset):
    def __init__(self, size, dataset_num, fold, mode, dataset_name=None):
        super().__init__()
        split = self.get_split()
        self.mode = mode
        PREPROCESSED_ROOT = os.getenv('PREPROCESSED_ROOT', None)
        self.data_root = f"{PREPROCESSED_ROOT}/Dataset_"
        if dataset_name is not None:
            self.data_root += f"{dataset_name}_"
        self.data_root += f"{dataset_num}/fold_{fold}/{split}/{self.mode}Tr/"
        self.file_paths = glob.glob(f"{self.data_root}/*")
        self._length = len(self.file_paths)

        size = None if size is not None and size<=0 else size
        self.size = size

        if mode == "images":
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_LINEAR)
        else:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_NEAREST)
        
        self.preprocessor = albumentations.Compose([self.rescaler, albumentations.CenterCrop(height=self.size, width=self.size)])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = np.load(self.file_paths[i])
        item = {
            "image":self.preprocess_item(item)
        }
        return item
    
    def preprocess_item(self, item):
        item = np.transpose(item, (1,2,0))
        item = self.preprocessor(image=item)["image"]
        return item

class ClassegTrain(ClassegDataset):
    # default to random_crop=True
    def __init__(self, size, dataset_num, fold, mode, dataset_name=None):
        super().__init__(size, dataset_num, fold, mode, dataset_name)

    def get_split(self):
        return "train"

class ClassegVal(ClassegDataset):
    # default to random_crop=True
    def __init__(self, size, dataset_num, fold, mode, dataset_name=None):
        super().__init__(size, dataset_num, fold, mode, dataset_name)

    def get_split(self):
        return "val"

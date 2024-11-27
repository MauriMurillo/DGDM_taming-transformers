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
        self.data_root += f"{dataset_num}/fold_{fold}/{split}/"
        self.croppper = albumentations.CenterCrop(height=self.size, width=self.size)

        if mode in ["images", "concat"]:
            self.img_paths = glob.glob(f"{self.data_root}/imagesTr/*")
            self.img_rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_LINEAR)
            self.img_pre = albumentations.Compose([self.croppper, self.img_rescaler])
        if mode in ["labels", "concat"]:
            self.label_paths = glob.glob(f"{self.data_root}/labelsTr/*")    
            self.label_rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_NEAREST)
            self.label_pre = albumentations.Compose([self.croppper, self.label_rescaler])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        if self.mode == "labels":
            item = np.load(self.label_paths[i])
            item = {
                "image":self.label_pre(item)
            }
        elif self.mode == "images":    
            item = np.load(self.img_paths[i])
            item = {
                "image":self.img_pre(item)
            }
        elif self.mode == "concat":
            label = np.load(self.label_paths[i])
            label = self.mask_pre(label)
            img = np.load(self.img_paths[i])
            img = self.mask_pre(img)
            item = {
                "image": np.concatenate([img, label])
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

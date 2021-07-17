import torchvision
from PIL import Image

import numpy as np

from typing import Any, Tuple

import os
import os.path
from typing import Callable, Optional

from torchvision.datasets.utils import verify_str_arg
from datasets.datasets_utils import getItem
import pickle

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

def put_pickle(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickle(file_path):
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    return data

class SeniorAnn:
    

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, num_imgs_per_cat=None, training_mode='SSL',
           
    ) -> None:

        self.transform = transform
        self.target_transform = target_transform
        self.training_mode = training_mode
        import os
        cwd = os.getcwd()
        print('cwd', cwd)
        self.split = split
        
        
        X_train, X_valid, X_test, y_train, y_valid, y_test, classifiers = get_pickle(
            'SiT_labled.pickle')

        # https://stackoverflow.com/a/55272357/13286959
        y_train = y_train[:, 0] - 1
        y_test = y_test[:, 0] - 1
        y_valid = y_valid[:, 0] - 1
       

        X_unlabeled = get_pickle('SiT_unlabled.pickle')

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]

        if self.split == 'train':
            self.data, self.labels = X_train, y_train
            

        elif self.split == 'train+unlabeled':
            self.data, self.labels = X_train,  y_train
            
            unlabeled_data = X_unlabeled
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data = X_unlabeled
            self.labels = np.asarray([-1] * self.data.shape[0])

        else:  # self.split == 'test':
            self.data, self.labels = np.concatenate((X_test, X_valid)),  np.concatenate((y_test, y_valid))

        unique, counts = np.unique(self.labels, return_counts=True)
        print("Dataset at SeniorAnn Class:")
        print(type(self.data))  # <class 'numpy.ndarray'>
        print(self.data.dtype)  # uint8
        print(self.data.shape)  # (105000, 3, 96, 96)
        print("first one:")
        print(type(self.data[0]))  # <class 'numpy.ndarray'>
        print(self.data[0].dtype)  # uint8
        print("labels:")
        print(type(self.labels))  # <class 'numpy.ndarray'>
        print(self.labels.shape)
        print('label value count')
        print(unique, counts)
        print("---")

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], -1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        return getItem(img, target, self.transform, self.training_mode)

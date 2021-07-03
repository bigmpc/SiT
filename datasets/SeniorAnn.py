import torchvision
from PIL import Image

import numpy as np

from typing import Any, Tuple

import os
import os.path
from typing import Callable, Optional

from torchvision.datasets.utils import verify_str_arg
from datasets.datasets_utils import getItem


class SeniorAnn(torchvision.datasets.STL10):
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False, num_imgs_per_cat=None, training_mode='SSL',
            lb_data = [],
            X_unlabeled = [],
    ) -> None:

        self.transform = transform
        self.target_transform = target_transform
        self.training_mode = training_mode

        X_train, X_valid, X_test, y_train, y_valid, y_test, classifiers = lb_data


        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]

        if self.split == 'train':
            self.data, self.labels = X_train, y_train
            

        elif self.split == 'train+unlabeled':
            self.data, self.labels = X_train, y_train
            
            unlabeled_data, _ = X_unlabeled
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate(
                (self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == 'unlabeled':
            self.data, _ = X_unlabeled
            self.labels = np.asarray([-1] * self.data.shape[0])

        else:  # self.split == 'test':
            self.data, self.labels = X_test, y_test

       


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], -1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        return getItem(img, target, self.transform, self.training_mode)

#! /usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.datasets.utils import check_integrity
import gdown
from gdown.cached_download import assert_md5sum
import os
import tarfile

class CUB200(Dataset):
    url = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, mode="train", transform=None, download=False):
        self.root = os.path.join(root, "cub2011")
        os.makedirs(self.root, exist_ok=True)
        self.mode = mode

        assert self.mode in ["train", "test", "all"]

        if download:
            try:
                self.set_paths_and_labels()
            except:
                self.download_dataset()
                self.set_paths_and_labels()
        else:
            self.set_paths_and_labels()
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict

    def set_paths_and_labels(self):
        img_folder = os.path.join(self.root, "CUB_200_2011", "images")
        self.dataset = datasets.ImageFolder(img_folder)
        self.labels = np.array([b for (a, b) in self.dataset.imgs])
        assert len(np.unique(self.labels)) == 200
        assert self.__len__() == 11788

        if self.mode == "train":
            # nonzero returns a tuple (one element per dimension)
            indices = np.nonzero(self.labels < 100)[0]
        elif self.mode == "test":
            indices = np.nonzero(self.labels >= 100)[0]
        else: # all
            indices = np.arange(len(self.labels))
        
        self.dataset = Subset(self.dataset, indices)
        self.labels = self.labels[indices]

        assert len(self.dataset) == len(self.labels)


    def download_dataset(self):
        output_location = os.path.join(self.root, self.filename)
        if check_integrity(output_location, self.md5):
            print('Using downloaded and verified file: ' + output_location)
        else:
            gdown.download(self.url, output_location, quiet=False)
            assert_md5sum(output_location, self.md5)
        with tarfile.open(output_location, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

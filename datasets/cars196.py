#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio
from torchvision.datasets.utils import download_url
import os
import tarfile
import logging
from tqdm import tqdm

class Cars196(Dataset):
    ims_url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
    ims_filename = 'car_ims.tgz'
    ims_md5 = 'd5c8f0aa497503f355e17dc7886c3f14'

    annos_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
    annos_filename = 'cars_annos.mat'
    annos_md5 = 'b407c6086d669747186bd1d764ff9dbc'

    def __init__(self, root, mode="train", transform=None, download=False):
        self.root = os.path.join(root, "cars196")
        self.mode = mode

        assert self.mode in ["train", "test", "all"]

        if download:
            try:
                self.set_paths_and_labels(assert_files_exist=True)
            except:
                self.download_dataset()
                self.set_paths_and_labels()
        else:
            self.set_paths_and_labels()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict

    def load_labels(self):
        img_data = sio.loadmat(os.path.join(self.dataset_folder, "cars_annos.mat"))
        self.labels = np.array([i[0, 0]-1 for i in img_data["annotations"]["class"][0]])
        self.img_paths = [os.path.join(self.dataset_folder, i[0]) for i in img_data["annotations"]["relative_im_path"][0]]

        if self.mode == "train":
            r = list(range(98))
        elif self.mode == "test":
            r = list(range(98, 196))
        else: # all
            r = list(set(self.labels))

        self.img_paths = [self.img_paths[i] for i in range(len(self.img_paths)) if self.labels[i] in r]
        self.labels = [l for l in self.labels if l in r]

        self.class_names = [i[0] for i in img_data["class_names"][0]]

    def set_paths_and_labels(self, assert_files_exist=False):
        self.dataset_folder = self.root
        self.load_labels()
        # assert len(np.unique(self.labels)) == 196
        # assert self.__len__() == 16185
        if assert_files_exist:
            logging.info("Checking if dataset images exist")
            for x in tqdm(self.img_paths):
                assert os.path.isfile(x)

    def download_dataset(self):
        url_infos = [(self.ims_url, self.ims_filename, self.ims_md5), 
                    (self.annos_url, self.annos_filename, self.annos_md5)]
        for url, filename, md5 in url_infos:
            download_url(url, self.root, filename=filename, md5=md5)
        with tarfile.open(os.path.join(self.root, self.ims_filename), "r:gz") as tar:
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

    def filter_examples(self, c):
        self.img_paths = [self.img_paths[i] for i in range(len(self.img_paths)) if self.labels[i] == c]
        self.labels = np.array([l for l in self.labels if l == c])

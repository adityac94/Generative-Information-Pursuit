"""
File containing a Torch Dataset object for the CUB dataset.
"""

import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader

N_ATTRIBUTES = 312


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        pkl_file_paths,
        use_attr,
        no_img,
        uncertain_label,
        image_dir,
        n_class_attr,
        prune=False,
        transform=None,
        no_label=False,
    ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            with open(file_path, "rb") as f:
                self.data.extend(pickle.load(f))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.no_label = no_label
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.prune = prune
        self.pruned_attr = [
            1,
            4,
            6,
            7,
            10,
            14,
            15,
            20,
            21,
            23,
            25,
            29,
            30,
            35,
            36,
            38,
            40,
            44,
            45,
            50,
            51,
            53,
            54,
            56,
            57,
            59,
            63,
            64,
            69,
            70,
            72,
            75,
            80,
            84,
            90,
            91,
            93,
            101,
            106,
            110,
            111,
            116,
            117,
            119,
            125,
            126,
            131,
            132,
            134,
            145,
            149,
            151,
            152,
            153,
            157,
            158,
            163,
            164,
            168,
            172,
            178,
            179,
            181,
            183,
            187,
            188,
            193,
            194,
            196,
            198,
            202,
            203,
            208,
            209,
            211,
            212,
            213,
            218,
            220,
            221,
            225,
            235,
            236,
            238,
            239,
            240,
            242,
            243,
            244,
            249,
            253,
            254,
            259,
            260,
            262,
            268,
            274,
            277,
            283,
            289,
            292,
            293,
            294,
            298,
            299,
            304,
            305,
            308,
            310,
            311,
        ]

        with open("data/CUB/classes.txt", "r") as f:
            self.classes = []
            for line in f.read().splitlines()[1:]:
                name = line.split(".")[1].replace("_", " ")
                for i in range(len(name)):
                    if name[i] == " " and name[i + 1].islower():
                        name = name[:i] + "-" + name[i + 1 :]
                self.classes.append(name)

        with open("data/CUB/attributes.txt", "r") as f:
            self.attributes = [line.split()[1] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img = Image.open(img_data["img_path"]).convert("RGB")

        class_label = img_data["class_label"]
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data["uncertain_attribute_label"]
            else:
                attr_label = img_data["attribute_label"]

            attr_label = torch.tensor(attr_label).float()

            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    for index in range(N_ATTRIBUTES):
                        if img_data["uncertain_attribute_label"][index] != 0:
                            one_hot_attr_label[index][int(attr_label[index])] = 1
                        else:
                            one_hot_attr_label[index][2] = 1
                    one_hot_attr_label = torch.tensor(one_hot_attr_label).float()
                    return one_hot_attr_label, class_label
                else:
                    if self.prune:
                        attr_label = attr_label[self.pruned_attr]
                    return attr_label, class_label
            else:
                if self.no_label:
                    return img, attr_label
                else:
                    return img, class_label, attr_label
        else:
            return img, class_label
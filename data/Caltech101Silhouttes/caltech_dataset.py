import os

import torch.utils.data as data
import scipy.io
import torchvision


class CaltechSilhouttes(data.Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.matfile = scipy.io.loadmat(
            os.path.join(
                root_dir, "Caltech101Silhouttes/caltech101_silhouettes_28_split1.mat"
            )
        )
        self.transform = transform
        self.split = split
        self.classes = [name.item() for name in self.matfile["classnames"].squeeze()]

    def __len__(self):
        if self.split == "train":
            return self.matfile["train_data"].shape[0]
        if self.split == "val":
            return self.matfile["val_data"].shape[0]
        if self.split == "test":
            return self.matfile["test_data"].shape[0]

    def __getitem__(self, idx):
        if self.split == "train":
            image = (
                self.matfile["train_data"][idx].reshape((28, 28, 1)).transpose(1, 0, 2)
            )
            label = int(self.matfile["train_labels"][idx]) - 1

        if self.split == "val":
            image = (
                self.matfile["val_data"][idx].reshape((28, 28, 1)).transpose(1, 0, 2)
            )
            label = int(self.matfile["val_labels"][idx]) - 1

        if self.split == "test":
            image = (
                self.matfile["test_data"][idx].reshape((28, 28, 1)).transpose(1, 0, 2)
            )
            label = int(self.matfile["test_labels"][idx]) - 1

        image = image * 255  # Image.fromarray(np.uint8(cm.binary(image)*255))
        PIL_converter = torchvision.transforms.ToPILImage(mode="L")

        image = PIL_converter(image)

        if self.transform:
            image = self.transform(image)

        return image, label

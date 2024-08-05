"""
Simple convolutional conditional VAE architecture for MNIST.
"""

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import utils


class VAE(nn.Module):
    def __init__(self, num_classes=10, zdims=100):
        super().__init__()
        self.num_classes = num_classes
        self.zdims = zdims

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        self.category_layer = nn.Linear(self.num_classes, 256)

        self.fc1_1 = nn.Linear(256 + 256, 2048)  # mu layer
        self.fc2_1 = nn.Linear(2048, zdims)  # mu layer
        self.fc1_2 = nn.Linear(256 + 256, 2048)  # logvariance layer
        self.fc2_2 = nn.Linear(2048, zdims)  # logvariance layer
        # self.fc23 = nn.Linear(500, num_classes)  # categorical parameters layer

        self.bnorm_fc1_1 = nn.BatchNorm1d(2048)
        self.bnorm_fc1_2 = nn.BatchNorm1d(2048)

        # this last layer bottlenecks through zdims connecti
        # DECODER
        # from bottleneck to hidden
        self.fc3 = nn.Linear(zdims + num_classes, 500)
        self.bnorm5_1d = nn.BatchNorm1d(500)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(500, 784)

        self.fc3_deconv = nn.Linear(zdims + num_classes, 256)

        self.deconv1 = torch.nn.ConvTranspose2d(256, 256, 4)
        self.bnorm5 = nn.BatchNorm2d(256)
        self.deconv2 = torch.nn.ConvTranspose2d(256, 128, 3)
        self.bnorm6 = nn.BatchNorm2d(128)

        self.deconv3 = torch.nn.ConvTranspose2d(256, 128, 3)
        self.bnorm7 = nn.BatchNorm2d(128)
        self.deconv4 = torch.nn.ConvTranspose2d(128, 64, 3)
        self.bnorm8 = nn.BatchNorm2d(64)

        self.deconv5 = torch.nn.ConvTranspose2d(64, 32, 3)
        self.bnorm9 = nn.BatchNorm2d(64)

        self.decoded_image_pixels = torch.nn.ConvTranspose2d(32, 1, 3)

        self.bnorm10 = nn.BatchNorm2d(32)

        self.unmaxpool1 = torch.nn.ConvTranspose2d(256, 256, 2, 2)

        self.unmaxpool2 = torch.nn.ConvTranspose2d(64, 64, 2, 2)

        # activations
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)

    def category_input(self, category):
        category = F.one_hot(category, self.num_classes).float()
        return self.category_layer(category)

    def encode(self, x, category):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool(self.bnorm2(self.relu(self.conv2(x))))

        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool(self.bnorm4(self.relu(self.conv4(x))))

        # x = self.conv_drop(x)

        h1 = nn.AvgPool2d(x.size()[2:])(x).view(-1, 256)
        category = self.category_input(category)

        h1 = torch.cat([h1, category], dim=1)
        h_mean = self.relu(self.bnorm_fc1_1(self.fc1_1(h1)))
        h_var = self.relu(self.bnorm_fc1_2(self.fc1_2(h1)))

        return self.fc2_1(h_mean), self.fc2_2(h_var)

    def sample(self, mu, logvar):
        std_dev = (0.5 * logvar).exp()
        eps = torch.randn_like(std_dev)
        return mu + eps * std_dev

    def decode(self, z, category):
        category = F.one_hot(category, self.num_classes).float()
        x = self.relu(self.fc3_deconv(torch.cat([z, category], dim=1)))

        x = x.view(-1, 256, 1, 1)
        x = self.relu(self.deconv1(x))
        # x = self.conv_drop(x)
        x = self.relu(self.bnorm5(self.unmaxpool1(x)))

        x = self.relu(self.bnorm6(self.deconv2(x)))
        # x = self.relu(self.deconv3(x))
        # x = self.bnorm7(x)
        x = self.relu(self.bnorm8(self.deconv4(x)))
        x = self.relu(self.bnorm9(self.unmaxpool2(x)))

        x = self.relu(self.bnorm10(self.deconv5(x)))
        return self.decoded_image_pixels(x)

    def generate(self, category):
        z_shape = (len(category), self.zdims)
        z = self.sample(
            torch.zeros(z_shape, device=category.device),
            torch.ones(z_shape, device=category.device),
        )
        return self.decode(z, category), z

    def forward(self, x, category):
        mu, logvar = self.encode(x, category)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z, category)
        return x_hat, mu, logvar

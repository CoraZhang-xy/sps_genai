import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.deconv2(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.sigmoid(self.fc(self.flatten(x)))


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator(100)
        self.discriminator = Discriminator()


def get_model(model_name):
    if model_name.upper() == "GAN":
        return GAN()
    raise ValueError(f"Unknown model: {model_name}")

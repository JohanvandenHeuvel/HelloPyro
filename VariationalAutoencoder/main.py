"""
Going trough Variational Autoencoder example from Pyro
Try implementation on my own, only using the given example as a
check and for the theory.
"""
import json
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
from torchvision import datasets


def plot_image(image):
    """
    Simple function to plot an image. Not very useful in its current state expect that it is modular
    and can be easily be re-used or expanded upon.
    :param image:
    :return:
    """
    plt.imshow(image)
    plt.show()


def load_data(batch_size):
    """
    Simple function to load data. Not very useful in its current state expect that it is modular
    and can be easily be re-used or expanded upon.
    :param batch_size:
    :return:
    """
    transform = T.ToTensor()
    train = datasets.MNIST('train', train=True, download=True, transform=transform)
    test = datasets.MNIST('test', train=False, download=True, transform=transform)

    loader = lambda dataset, shuffle: torch.utils.data.DataLoader(dataset,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  num_workers=1)

    train_loader = loader(train, shuffle=True)
    test_loader = loader(test, shuffle=False)

    return train_loader, test_loader


class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # Linear transformations
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # Non-linear transformations
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        hidden = self.softplus(self.fc1(z))

        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img


class Encoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # Linear transformations
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # Non-linear transformations
        self.softplus = nn.Softplus()

    def forward(self, x):

        x = x.reshape(-1, 784)

        hidden = self.softplus(self.fc1(x))

        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))

        return z_loc, z_scale


class VAE(nn.Module):

    def __init__(self, z_dim=50, hidden_dim=400):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        self.z_dim = z_dim

    def model(self, x):
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            loc_img = self.decoder(z)

            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

            return loc_img

    def guide(self, x):
        pyro.module("encoder", self.encoder)

        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)

            pyro.sample("obs", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)

        z = dist.Normal(z_loc, z_scale).sample()

        loc_img = self.decoder(z)

        return loc_img


def train(svi, train_loader):
    epoch_loss = 0.

    for x, _ in train_loader:
        epoch_loss += svi.step(x)

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader):
    test_loss = 0.

    for x, _ in test_loader:
        test_loss += svi.evaluate_loss(x)

    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def main(args):

    batch_size = args['batch_size']
    train_loader, test_loader = load_data(batch_size)

    vae = VAE()
    optimized = Adam({"lr": 1.0e-3})
    svi = SVI(vae.model, vae.guide, optimized, loss=Trace_ELBO())

    NUM_EPOCHS = 100
    TEST_FREQUENCY = 5

    pyro.clear_param_store()

    train_elbo = []
    test_elbo = []

    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch {:03d}] average training loss: {:04f}".format(epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            total_epoch_loss_test = evaluate(svi, test_loader)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch {:03d}] average training loss: {:04f}".format(epoch, total_epoch_loss_test))


if __name__ == "__main__":

    with open("parameters.json", "r") as read_file:
        parameters = json.load(read_file)

    main(parameters)
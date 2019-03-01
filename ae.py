"""Autoencoders.

Matthew Alger
Research School of Astronomy and Astrophysics
The Australian National University
2019
"""

import torch, torch.nn as nn, numpy, matplotlib.pyplot as plt
from tqdm import tqdm as tqdm


def torchify(x):
    """Convert numpy array to torch vector."""
    x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def numpify(x):
    """Convert torch vector to numpy array."""
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.detach().numpy()
    return x


# Default loss function
mseloss = nn.MSELoss()


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, repr_size, z_draws=7, kl_weight=1):
        super().__init__()
        self.input_size = self.n = input_size
        self.repr_size = self.k = repr_size
        self.hidden_size = self.h = hidden_size
        
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.k, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.n),
        )
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        # Decode
        x = self.decoder(x)
        return x
    

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, repr_size, z_draws=7, kl_weight=1):
        super().__init__()
        self.input_size = self.n = input_size
        self.repr_size = self.k = repr_size
        self.hidden_size = self.h = hidden_size
        self.kl_weight = kl_weight
        self.z_draws = z_draws
        
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.k * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.k, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.n),
        )
        
    def forward(self, x):
        x, _ = self.forward_(x)
        return x
    
    def forward_(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(-1, 2, self.k)
        # Reparameterisation trick
        mu = x[:, 0]
        sigma = x[:, 1]
        sigma_ = sigma.mul(0.5).exp_()
        # eps is N x k x z_draws
        eps = torch.DoubleTensor(sigma_.size() + (self.z_draws,)).normal_()
        eps = torch.transpose(eps, 1, 2)
        eps = torch.autograd.Variable(eps)
        if torch.cuda.is_available():
            eps = eps.cuda()
        # x is N x z_draws x k
        x = eps.mul(sigma_[:, None, :]).add_(mu[:, None, :])
        # Decode
        x = x.view(-1, self.k)
        x = self.decoder(x)
        x = x.view(-1, self.z_draws, self.n)
        full_x = x
        x = x.mean(dim=1)
        return x, mu, sigma, full_x, eps
    
    def loss(self, recon_x, x, mu, logvar):
        BCE = mseloss(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return BCE + self.kl_weight * KLD, self.kl_weight * KLD.item(), BCE.item()

    
def train(model, optimiser, losses=None, mses=None, klds=None, n_epochs=500, do_plot=True):
    if not losses:
        losses = []
    if not mses:
        mses = []
    if not klds:
        klds = []
    assert len(losses) == len(mses) == len(klds)

    if do_plot:
        fig = plt.figure(figsize=(6, 4))
        plot_loss, = plt.plot(losses, label='total')
        plot_mse, = plt.plot(mses, label='mse')
        plot_kld, = plt.plot(klds, label='kld')
        plt.gca().set_yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')

    torch_dataset = torchify(dataset)
    shuffle = numpy.arange(len(torch_dataset))
    bar = tqdm(total=n_epochs)
    for epoch in range(n_epochs):
        model.train()
        numpy.random.shuffle(shuffle)
        optimiser.zero_grad()
        recon, mu, sigma, full_recon, draws = model.forward(torch_dataset[shuffle])
        loss, kld_loss, mse_loss = model.loss(recon, torch_dataset[shuffle], mu, sigma)
        loss.backward()
        loss_data = loss.item()
        losses.append(loss_data)
        mses.append(mse_loss)
        klds.append(kld_loss)
        optimiser.step()
        bar.update()
        bar.postfix = ' {:.6f}'.format(loss_data)
        if do_plot:
            plot_loss.set_data(range(len(losses)), numpy.array(losses))
            plot_mse.set_data(range(len(losses)), numpy.array(mses))
            plot_kld.set_data(range(len(losses)), numpy.array(klds))
            fig.gca().relim()
            fig.gca().autoscale_view()
            fig.canvas.draw()
#%%
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from utils import *
from dataset import *
import pytorch_lightning as pl

import torch.nn.init as init
from torch.autograd import Variable


class Reconstruction_Loss(nn.Module):
    
    def __init__(self):
        super(Reconstruction_Loss, self).__init__()

    def forward(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        else:
            recon_loss = None

        return recon_loss

class KL_Divergence(nn.Module):

    def __init__(self):
        super(KL_Divergence, self).__init__()

    def forward(self, mu, logvar):
        
        batch_size = mu.size(0)
        assert batch_size != 0
       
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_B(pl.LightningModule):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    #is there an args init?? for the pl mocule


    def __init__(self, z_dim=10, nc=1, epochs=10, batch_size=12):
        super().__init__()
        
        self.nc = nc
        self.z_dim = z_dim

        self.beta = 4          #'beta parameter for KL-term in original beta-VAE'
        self.decoder_dist = 'bernoulli'
        
    
        self.batch_size = batch_size
        self.epochs = 10
        self.max_iter = self.batch_size*self.epochs

        self.count = 0
        self.lr = 1e-4
        self.beta1 = 0.9       #, type=float, help='Adam optimizer beta1')
        self.beta2 = 0.999
        #self.gamma = 1000      #'gamma parameter for KL-term in understanding beta-VAE')
        #self.C_max = 25        #, type=float, help='capacity parameter(C) of bottleneck channel')
        #self.C_stop_iter=1e5   #, type=float, help='when to stop increasing the capacity')

        self.C_max = torch.FloatTensor([25])
        self.C_stop_iter=torch.FloatTensor([0.1*self.max_iter])
        self.gamma = torch.FloatTensor([1000])

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        #self.weight_init()

    
    def training_step(self, batch, batch_idx):
        
        #we are not going to be using y(lablels) in the autoencoder
        x, y = batch
        self.count += 1

        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        x_recon = self.decoder(z).view(x.size())
        
        recon_loss = Reconstruction_Loss()(x, x_recon, self.decoder_dist)
        total_kld, dim_wise_kld, mean_kld = KL_Divergence()(mu, logvar)

        #beta_vae_loss = recon_loss + self.beta*total_kld
        
        C = torch.clamp(self.C_max/self.C_stop_iter*self.count, torch.FloatTensor([0]), self.C_max)
        beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

        self.log("train_loss", beta_vae_loss)
        
        return beta_vae_loss
              
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        return optimizer                
    
    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps
    
    def forward(self):
        pass

    def forward(self, x):
        
        #we are not going to be using y(lablels) in the autoencoder
        
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        x_recon = self.decoder(z).view(x.size())
        
        return x_recon

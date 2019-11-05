#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from pushover import notify
from utils import makegif
from random import randint

# from IPython.display import Image
from IPython.core.display import Image, display

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[3]:


bs = 128


# In[4]:


# Load Data
# dataset = datasets.ImageFolder(root='./rollouts', transform=transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(), 
# ]))
dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
len(dataset), len(dataloader)


# In[5]:


# Fixed input for debugging
fixed_x, _ = next(iter(dataloader))
# print(len(fixed_x))
save_image(fixed_x, 'real_image.png')

Image('real_image.png')


# In[6]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# In[7]:


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 2, 2)


# In[8]:


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=512, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_sigma = nn.Linear(h_dim, z_dim)
        self.fc_z_to_hidden = nn.Linear(z_dim, h_dim)
        
        self.partial_decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = 1e-5 + torch.exp(0.5 * logvar)
        # esp = torch.randn(*mu.size()).cuda()
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc_mu(h), self.fc_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        h = self.fc_z_to_hidden(z)
        return self.partial_decoder(h)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


# In[9]:


image_channels = fixed_x.size(1)
# print(image_channels, fixed_x.shape)


# In[10]:


model = VAE(image_channels=image_channels).to(device)
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))


# In[11]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 


# In[12]:


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# In[13]:


# get_ipython().system('rm -rfr reconstructed')
# get_ipython().system('mkdir reconstructed')


# In[14]:


epochs = 7


# In[15]:


for epoch in range(epochs):
    for idx, (images, _) in enumerate(dataloader):
        recon_images, mu, logvar = model(images.to(device))
        loss, bce, kld = loss_fn(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.item() / bs, bce.item() / bs, kld.item() / bs)
        if idx and idx % 150 == 0:
            print(to_print)

# notify to android when finished training
# notify(to_print, priority=1)

# torch.save(vae.state_dict(), 'vae.torch')


# In[15]:


def compare(x):
    recon_x, _, _ = model(x)
    return torch.cat([x, recon_x])


# In[22]:


# sample = torch.randn(bs, 1024)
# compare_x = vae.decoder(sample)

# fixed_x, _ = next(iter(dataloader))
# fixed_x = fixed_x[:8]
for i in range(10):
    fixed_x = dataset[randint(1, 100)][0].unsqueeze(0)
    compare_x = compare(fixed_x.to(device))

    save_image(compare_x.data.cpu(), f'sample_image_{i}.png')
# display(Image('sample_image.png', width=700, unconfined=True))


# In[ ]:


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from torch.distributions.uniform import Uniform

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import time

from IPython.display import display, clear_output

import math

from tqdm import tqdm

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)

        gamma = self.embed(y)

        out = gamma.view(-1, self.num_out) * out
        return out
        
class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 2)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))


        return self.lin4(x)
    




class Diffusion(nn.Module):

    def __init__(self,steps,latent_size,num_var,deg,improver,improver2):
        super(Diffusion, self).__init__()

        self.num_timesteps = steps

        #self.dim = 2
        self.latent_size = latent_size

        self.improver = improver

        self.improver2 = improver2

        #self.time_dim = self.dim * 4

        #Parameterisation of the reverse process
        layers = []
        layers.append(nn.Linear(latent_size+1+num_var*deg,128))   #latent size,time,context,advantage
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,self.latent_size))

        self.outy = torch.nn.Sequential(*layers)

        #Parameterisation of the reverse process
        layers = []
        layers.append(nn.Linear(latent_size + 1,128))   #latent size,time,context,advantage
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,self.latent_size))

        #self.outy2 = torch.nn.Sequential(*layers)

        self.outy2 = ConditionalModel(n_steps=self.num_timesteps)


        #Embedding for the time index
        layers = []
        layers.append(nn.Linear(1,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,1))

        self.time_mlp = torch.nn.Sequential(*layers)

        #self.time_mlp = nn.Embedding(100, num_out)
        #self.time_mlp.weight.data.uniform_()

        #Context
        layers = []
        layers.append(nn.Linear(num_var*deg,128))   #Adding the context length
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,num_var*deg))

        self.context_embedding = torch.nn.Sequential(*layers)
        
        """
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        """
        

        self.loss = torch.nn.MSELoss()

        self.min = None
        self.max = None




    def generate_sample_good(self,num,context,advantage,device,gradient):

        #advantage = advantage
        z_total = None

        #Scheduling parameters
        alphas,alphas_cumprod,betas,alphas_cumprod_prev = self.generate_alpha(device = device)

        #Initial sample
        z = torch.normal(mean = 0, std = 1,size = (num,self.latent_size)).to(device)

        #Iterative generation of sample
        for i in reversed(range(1,self.num_timesteps+1)):

            #Create time as a tensor
            t = i*torch.ones(size = (z.shape[0],1)).to(device)


            t_embed = self.time_mlp(t)

            #t = t.type(torch.int64) - 1 #Used for indexing so subtraction occures here

            time_0 = (t > 1).int()
            m = time_0*torch.normal(mean = 0, std = 1,size = (num,self.latent_size)).to(device)



            #Noise prediction

            noise_prediction = self.predict_noise(z,t)


            t = t.type(torch.int64) - 1 #Used for indexing so subtraction occures here

            #Extracting 
            alphas_cumprod_t = self.extract(alphas_cumprod, t.squeeze(dim=1),z.shape)
            alphas_cumprod_prev_t = self.extract(alphas_cumprod_prev, t.squeeze(dim=1),z.shape)
            alphas_t = self.extract(alphas, t.squeeze(dim=1),z.shape)
            betas_t = self.extract(betas, t.squeeze(dim=1),z.shape)

            gradient_value = self.improver2.get_grad(z,context.to(device),device)

            #What step size are we going to use for this
            if gradient is not None:
                noise_prediction = noise_prediction - torch.sqrt(1-alphas_cumprod_t)*gradient*gradient_value

            #Updated Z
            z = torch.sqrt(alphas_cumprod_prev_t)*(torch.reciprocal(torch.sqrt(alphas_cumprod_t))*(z -torch.sqrt(1-alphas_cumprod_t)*noise_prediction)) + torch.sqrt(1-alphas_cumprod_prev_t)*noise_prediction

            #z = (1/torch.sqrt(alphas_t))*(z - ((1-alphas_t)/torch.sqrt(1-alphas_cumprod_t))*noise_prediction) + torch.sqrt(betas_t)*m

        z = self.scale_up(z)


        return z
    
    def linear_beta_schedule(self,timesteps):

        beta_start = 0.0001
        beta_end = 0.02

        return torch.linspace(beta_start, beta_end, timesteps)

    def generate_alpha(self,device = None):

        # define beta schedule (variance schedule)
        betas = self.linear_beta_schedule(timesteps=self.num_timesteps)

        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        if device is not None:
            return alphas.to(device),alphas_cumprod.to(device),betas.to(device),alphas_cumprod_prev.to(device)
        else:
            return alphas,alphas_cumprod,betas,alphas_cumprod_prev


    #Question of whether this is correct
    def extract(self,a, t, x_shape):

        batch_size = t.shape[0]
        out = a.gather(-1, t)
        fin  =out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


        return fin

    def generate_targets(self,x0,t,device):

        alphas,alphas_cumprod,betas,alphas_cumprod_prev = self.generate_alpha(device = device)
        alphas_cumprod_t = self.extract(alphas_cumprod.to(device), t.squeeze(dim=1) - 1,x0.shape)


        noise = torch.normal(mean = 0, std = 1,size = x0.shape).to(device)

        x_t = torch.sqrt(alphas_cumprod_t)*x0 + torch.sqrt(1-alphas_cumprod_t)*noise

        #alphas_cumprod_t = self.extract(alphas_cumprod.to(device), t.squeeze(dim=1) - 1,x0.shape)
        #betas_t = self.extract(betas.to(device), t.squeeze(dim=1) - 1,x0.shape)

        return noise,x_t
    
    def predict_value(self,x):
        x = self.scale_down(x)
        return self.improver.predict_value(x)
    
    def predict_best(self,x):

        x = self.scale_down(x)

        logit = self.improver2.predict_value(x)

        m = nn.Sigmoid()

        prob = m(logit)

        return prob
    
    def scale_down(self,z):
        z = 2*(z- self.min)/(self.max-self.min) - 1

        return z
    
    def scale_up(self,z):
        z = (((z + 1)/2)*((self.max-self.min))) + self.min

        return z
        



    def predict_noise(self,z,t):

        #t_embed = self.time_mlp(t.float())

        #first we need to scale this down
        #z = self.scale_down(z)
        
        noise_prediction = self.outy2(z,t.long() - 1)

        return noise_prediction
    



    def calculate_loss(self,x0,context,advantage):

        x0 = self.scale_down(x0)

        device = x0.device

        #Sampling the time step to update
        #t = self.time_dist.sample(sample_shape = (x0.shape[0],1))

        t = torch.randint(1,self.num_timesteps + 1, (x0.shape[0],1)).to(device)

        #t_embed = self.time_mlp(t.float())

        t = t.type(torch.int64)

        #Apply noise to certain timeframes
        noise,data = self.generate_targets(x0,t,device)

        #Predict the noise we added from final value
        prediction = self.predict_noise(z = data, t = t)

        assert prediction.shape == noise.shape

        

        loss = self.loss(prediction,noise)

        #loss = torch.mean(loss,dim=1,keepdim=True)
        #advantage_weights = torch.exp(10*advantage)
        #assert loss.shape == advantage_weights.shape
        #scaled_loss = advantage_weights*loss
        #scaled_loss_mean = torch.mean(scaled_loss)

        return loss
        
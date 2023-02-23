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



class Diffusion(nn.Module):

    def __init__(self,steps,latent_size,num_var,deg):
        super(Diffusion, self).__init__()

        self.num_timesteps = steps

        #self.dim = 2
        self.latent_size = latent_size

        #self.time_dim = self.dim * 4

        #Parameterisation of the reverse process
        layers = []
        layers.append(nn.Linear(latent_size+1+num_var*deg + 1,128))   #latent size,time,context,advantage
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,self.latent_size))

        self.model = torch.nn.Sequential(*layers)

        #Embedding for the time index
        layers = []
        layers.append(nn.Linear(1,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128,1))

        self.time_mlp = torch.nn.Sequential(*layers)

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




    def generate_sample(self,num,context,advantage,device):

        advantage = advantage

        #Initial sample
        z = torch.normal(mean = 0, std = 1,size = (num,self.latent_size)).to(device)

        #Scheduling parameters
        alphas,alphas_cumprod,betas,alphas_cumprod_prev = self.generate_alpha()

        alphas = alphas.to(device)
        alphas_cumprod = alphas_cumprod.to(device)
        betas = betas.to(device)
        alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        

        for i in reversed(range(1,self.num_timesteps+1)):

            

            #Create t as a tensor
            t = i*torch.ones(size = (z.shape[0],1)).to(device)

            t_embed = self.time_mlp(t)


            

            
            t = t.type(torch.int64) - 1 #Used for indexing so subtraction occures here
        
            #Noise is zero at time step 0
            time_0 = (t > 1).int()
            m = time_0*torch.normal(mean = 0, std = 1,size = (num,self.latent_size)).to(device)

            #Noise prediction
            
            prediction = self.predict_noise(z,t_embed,context,advantage)
            


            alphas_t = self.extract(alphas, t.squeeze(dim=1),z.shape)
            alphas_cumprod_t = self.extract(alphas_cumprod, t.squeeze(dim=1),z.shape)
            alphas_cumprod_prev_t = self.extract(alphas_cumprod_prev, t.squeeze(dim=1),z.shape)
            betas_t = self.extract(betas, t.squeeze(dim=1),z.shape)


            posterior_variance = betas_t * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t)


            #Update z
            z = (1/torch.sqrt(alphas_t))*(z - ((1-alphas_t)/torch.sqrt(1-alphas_cumprod_t))*prediction) + torch.sqrt(betas_t)*m

            



        return z

    def linear_beta_schedule(self,timesteps):

        beta_start = 0.0001
        beta_end = 0.02

        return torch.linspace(beta_start, beta_end, timesteps)

    def generate_alpha(self):

        # define beta schedule (variance schedule)
        betas = self.linear_beta_schedule(timesteps=self.num_timesteps)

        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        return alphas,alphas_cumprod,betas,alphas_cumprod_prev


    def extract(self,a, t, x_shape):

        batch_size = t.shape[0]
        out = a.gather(-1, t)

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def generate_targets(self,x0,t,device):

        alphas,alphas_cumprod,betas,alphas_cumprod_prev = self.generate_alpha()

        noise = torch.normal(mean = 0, std = 1,size = x0.shape).to(device)



        alphas_t = self.extract(alphas.to(device), t.squeeze(dim=1) - 1,x0.shape)
        alphas_cumprod_t = self.extract(alphas_cumprod.to(device), t.squeeze(dim=1) - 1,x0.shape)
        betas_t = self.extract(betas.to(device), t.squeeze(dim=1) - 1,x0.shape)

        data = torch.sqrt(alphas_cumprod_t)*x0 + torch.sqrt(1-alphas_cumprod_t)*noise

        return noise,data

    def predict_noise(self,z,t,context,advantage):
        device = z.device
        context = self.context_embedding(context.to(device)[:,0:-1])

        advantage = advantage.masked_fill(advantage < 0,0)
        advantage = (advantage.masked_fill(advantage > 0,1).to(device))



        x = torch.cat((z,t,context.to(device),advantage),dim=1)
        return self.model(x)

    def calculate_loss(self,x0,context,advantage):

        advantage = advantage

        device = x0.device

        #Sampling the time step to update
        #t = self.time_dist.sample(sample_shape = (x0.shape[0],1))

        t = torch.randint(1,self.num_timesteps + 1, (x0.shape[0],1)).to(device)



        t_embed = self.time_mlp(t.float())

        

        t = t.type(torch.int64)

        #Apply noise to certain timeframes
        noise,data = self.generate_targets(x0,t,device)

        #Predict the noise we added from final value
        prediction = self.predict_noise(data,t_embed,context,advantage)

        assert prediction.shape == noise.shape

        loss = self.loss(prediction,noise)

        return loss
        
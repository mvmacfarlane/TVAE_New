import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
from torch.distributions import MultivariateNormal
from torch.distributions.uniform import Uniform
from diffusion import Diffusion
from torch.distributions import Categorical
from improver import Improver


from utils import get_layers


class Encoder(nn.Module):
    def __init__(self,latent_size,num_var,deg,activation,embed_dim):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.num_var = num_var
        self.deg = deg
        self.embed_dim = embed_dim


        self.context_embedding = get_layers(input_dim = num_var*deg,output_dim = self.embed_dim,act_output = True,act_type = activation,num = 0)
        self.solution_embedding = get_layers(input_dim = num_var,output_dim = self.embed_dim,act_output = True,act_type = activation,num = 0)
        self.embedding = get_layers(input_dim = 2*self.embed_dim,output_dim = self.embed_dim,act_output = True,act_type = activation,num = 3)

        self.output_head_mu = nn.Linear(128,self.latent_size)
        self.output_head_sigma = nn.Linear(128,self.latent_size)


    def reparameterise(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,solution,context):

        solution_embedding = self.solution_embedding(solution)
        context_embedding = self.context_embedding(context[:,0:-1])

        hx = self.embedding(torch.cat((solution_embedding,context_embedding),dim=1))

        mu = self.output_head_mu(hx)
        log_sigma = self.output_head_sigma(hx)

        assert not torch.isnan(mu.sum())
        assert not torch.isnan(log_sigma.sum())

        Z = self.reparameterise(mu, log_sigma)

        normal = torch.distributions.Normal(mu, torch.exp(0.5*log_sigma))

        Z_logp = normal.log_prob(Z.double()).sum(dim=1)

        return Z, mu, log_sigma, Z_logp




class DiscreteDecoder(nn.Module):
    def __init__(self,no_latent = False,num_var = None,deg = None,latent_size = None,activation = None,embed_dim = None):
        super(DiscreteDecoder, self).__init__()

        self.output_dim = 40
        self.num_dim = num_var
        self.embed_dim = embed_dim
        self.no_latent = no_latent

        #Parameterised Components
        self.context_embedder = get_layers(input_dim = num_var*deg,output_dim = self.embed_dim,act_output = True,act_type = activation,num = 1)
        self.action_embedder =  get_layers(input_dim = 1,output_dim = self.embed_dim,act_output = True,act_type = activation,num = 1)
        self.latent_embedder = get_layers(input_dim = latent_size,output_dim = self.embed_dim,act_output = True,act_type = activation,num = 1)

        if no_latent:
            self.embedding = get_layers(self.embed_dim,self.embed_dim,act_output = True,act_type = activation)
        else:
            self.embedding = get_layers(2*self.embed_dim,self.embed_dim,act_output = True,act_type = activation,num=1)

        self.gru = nn.GRUCell(self.embed_dim,self.embed_dim)

        self.action_readout = get_layers(self.embed_dim,self.output_dim +1,act_output = False,act_type = activation,num = 1)

        #Output Distribution
        self.m = torch.nn.Softmax(dim=1)

    def readout(self,embedding,greedy,force,solution,latent,num):

        #Readout distribution over actions
        logits = self.action_readout(embedding)
        probs = self.m(logits)
        dist=Categorical(probs = probs)

        #Generate action
        if force:
            solution = self.cont_to_cat(solution)
            action = solution
        else:
            if greedy:
                action = torch.argmax(probs,dim=1)
            else:
                action = dist.sample()

        logp = dist.log_prob(action).unsqueeze(dim=1)

        action = self.cat_to_cont(action).unsqueeze(dim=1)

        return action,logp

    def cat_to_cont(self,action):
       return -1 + action*(2/self.output_dim)

    def cont_to_cat(self,action):
       return ((action + 1)*(self.output_dim/2)).int()


    def forward(self, context, solution, Z, teacher_forcing,greedy,flag = False):

        if self.no_latent:
            context_embedding = self.context_embedder(context[:,0:-1])
            hx = self.embedding(context_embedding)

        else:
            context_embedding = self.context_embedder(context[:,0:-1])
            latent_embedding = self.latent_embedder(Z)

            #Context and Solution Representation
            hx = self.embedding(torch.cat((context_embedding,latent_embedding),dim=1))


        actions = []
        logps = []

        for i in range(self.num_dim):

            #Choose the action 
            if solution is None:
                action,logp = self.readout(hx,greedy = greedy,force = teacher_forcing,solution = None,latent = True,num = i)
            else:
                action,logp = self.readout(hx,greedy = greedy,force = teacher_forcing,solution = solution[:,i],latent = True,num = i)

            actions.append(action)
            logps.append(logp)

            #Updating the hidden reprensentation to include what actions we have taken so far
            action_embedding = self.action_embedder(action)
            hx = self.gru(action_embedding,hx)

        actions.reverse()


        solution = torch.cat(tuple(actions),dim=1)
        logps = torch.cat(tuple(logps),dim=1)
        solution_logp = torch.sum(logps,dim=1).unsqueeze(dim=1)


        return None,solution, solution_logp,None,None












class VAE_Solver(nn.Module):
    def __init__(self, config,diffusion_steps,num_var,deg,latent_size,act_type,embed_dim):
        super(VAE_Solver, self).__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size = self.latent_size,num_var = num_var,deg = deg,activation=act_type,embed_dim=embed_dim)

        self.decoder = DiscreteDecoder(no_latent = False,num_var = num_var,deg = deg,latent_size = self.latent_size,activation=act_type,embed_dim=embed_dim)

        self.improver = Improver(latent_size = self.latent_size,embed_dim=embed_dim,activation=act_type,num_var = num_var,deg = deg)
        self.improver2 = Improver(latent_size = self.latent_size,embed_dim=embed_dim,activation=act_type,num_var = num_var,deg = deg)

        self.diffusion = Diffusion(steps = diffusion_steps,latent_size = self.latent_size,num_var = num_var,deg = deg,improver = self.improver,improver2 = self.improver2)

    #We dont even use the config yet,why is that
    def forward(self,context,solution,config,teacher_forcing = True,greedy = False):

        #Encoding Solution
        Z, mu, log_var,Z_logp = self.encoder(
            
            solution,
            context,
            
        )

        #Decoding Solution
        _, decoded_solution, tour_logp,_,_ = self.decoder(
            
            context = context,
            solution = solution,
            Z = Z,
            teacher_forcing = teacher_forcing,   
            greedy = greedy,        

        )


        return decoded_solution, tour_logp,mu, log_var, Z,Z_logp






class RL_Solver(nn.Module):
    def __init__(self, config,diffusion_steps,num_var,deg):
        super(RL_Solver, self).__init__()

        self.latent_size = 2

        self.decoder = DiscreteDecoder(no_latent = True,num_var = num_var,deg = deg,latent_size = self.latent_size)
        #self.decoder = DiscreteDecoder(no_latent = False,num_var = num_var,deg = deg,latent_size = self.latent_size,activation=act_type,embed_dim=embed_dim)

    #We dont even use the config yet,why is that
    def forward(self,context,solution,teacher_forcing = True,greedy = False):


        #Decoding Solution
        _, decoded_solution, tour_logp,_,_ = self.decoder(
            
            context = context,
            solution = solution,
            Z = torch.zeros(size = (context.shape[0],2)),  #Hardcoded for now
            teacher_forcing = teacher_forcing,   
            greedy = greedy,        

        )


        return decoded_solution, tour_logp


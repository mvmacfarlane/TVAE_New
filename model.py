import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
from torch.distributions import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch_truncnorm.TruncatedNormal import TruncatedNormal
from diffusion import Diffusion
from torch.distributions import Categorical





class Encoder(nn.Module):
    def __init__(self,latent_size,num_var,deg):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        self.num_var = num_var
        self.deg = deg

        layers = []
        layers.append(nn.Linear(num_var + num_var*deg,128))
        #layers.append(nn.Linear(2,128))
        layers.append(nn.ReLU())
        layer_num = 3
        for i in range(layer_num):
            layers.append(nn.Linear(128,128))
            layers.append(nn.ReLU())

        self.latent_embed = nn.Sequential(*layers)

        self.output_head_mu = nn.Linear(128,self.latent_size)
        self.output_head_sigma = nn.Linear(128,self.latent_size)

    def reparameterise(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,solution,context):

        solution = solution.float()

        input = torch.cat((solution,context[:,0:-1]),dim=1)


        hx = self.latent_embed(input)

        mu = self.output_head_mu(hx)
        log_sigma = self.output_head_sigma(hx)

        #print("Hello")
        #print(mu)


        assert not torch.isnan(mu.sum())
        assert not torch.isnan(log_sigma.sum())

        Z = self.reparameterise(mu, log_sigma)


        normal = torch.distributions.Normal(mu, torch.exp(log_sigma))

        #cov_mat = torch.diag_embed(torch.exp(log_sigma)).to(mu.device)
        #m = MultivariateNormal(mu.double(), cov_mat.double())

        Z_logp = normal.log_prob(Z.double()).sum(dim=1)


        return Z, mu, log_sigma, Z_logp

"""
class Decoder(nn.Module):
    def __init__(self,latent_size):
        super(Decoder, self).__init__()

        self.latent_size = latent_size

        layers = []
        layer_num = 3
        layers.append(nn.Linear(self.latent_size + 4,128))
        #layers.append(nn.Linear(self.latent_size,128))
        layers.append(nn.Tanh())
        for i in range(layer_num):
            layers.append(nn.Linear(128,128))
            layers.append(nn.Tanh())

        self.latent_embed = nn.Sequential(*layers)

        #Wait why are we modeling this decoder as a gaussian
        layers = []
        layers.append(nn.Linear(128,128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128,2))
        layers.append(nn.Tanh())
        self.output_head_mu = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(128,128))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(128,2))
        layers.append(nn.Tanh())
        self.output_head_sigma = nn.Sequential(*layers)
    

    def reparameterise(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    #How do we ensure that is is between certain values, well technically the 
    def forward(self, context, solution, Z, teacher_forcing,greedy,flag = False):

        input = torch.cat((Z,context[:,0:4]),dim=1)
        #input = Z

        hx = self.latent_embed(input)
        mu = self.output_head_mu(hx)
        log_sigma = self.output_head_sigma(hx)


        if not teacher_forcing:
            if not greedy:
                action = self.reparameterize_truncated_gaussian(mean = mu, log_std = log_sigma, lower = -1, upper = 1)
            else:
                action = mu
        else:
            action = solution

        #Needed to prevent bug not sure what for now
        action = torch.clip(action,-0.999,0.999)

        m = TruncatedNormal(
            
            loc = mu,
            scale = torch.exp(log_sigma),
            a = -1,
            b= 1,

        )
        
        tour_logp = m.log_prob(action.double())
        tour_logp = torch.sum(tour_logp,dim=1)


        return None,action, tour_logp,mu,log_sigma
"""

class DiscreteDecoder(nn.Module):
    def __init__(self,no_latent = False,num_var = None,deg = None,latent_size = None):
        super(DiscreteDecoder, self).__init__()


        self.output_dim = 40
        self.num_dim = num_var
        self.embed_dim = 128

        self.no_latent = no_latent


        #Parameterised Components
        self.context_embedder = self.get_layers(num_var*deg,self.embed_dim,act_output = True,num = 1)
        self.action_embedder =  self.get_layers(1,self.embed_dim,act_output = True,num = 1)


        self.latent_embedder = self.get_layers(latent_size,self.embed_dim,act_output = True,num = 1)

        if no_latent:
            self.embedding = self.get_layers(self.embed_dim,self.embed_dim,act_output = True)
        else:
            self.embedding = self.get_layers(self.embed_dim,self.embed_dim,act_output = True,num=1)

        self.gru = nn.GRUCell(self.embed_dim,self.embed_dim)

        self.action_readout = self.get_layers(self.embed_dim + 1,self.output_dim +1,act_output = False,num = 1)

        #Output Distribution
        self.m = torch.nn.Softmax(dim=1)


    def get_layers(self,input_dim,output_dim,act_output,num = 0):

        layers = []

        if num == 0:
            layers.append(nn.Linear(input_dim,output_dim))
        else:
            layers.append(nn.Linear(input_dim,128))
            layers.append(nn.Tanh())

            for i in range(num):
                layers.append(nn.Linear(128,128))
                layers.append(nn.Tanh())

                

            layers.append(nn.Linear(128,output_dim))


        

        if act_output:
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def readout(self,embedding,greedy,force,solution,latent,num):

        


        logits = self.action_readout(embedding)





        probs = self.m(logits)





        dist=Categorical(probs = probs)

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
            hx = self.embedding(latent_embedding)


        actions = []
        logps = []


        for i in range(self.num_dim):

            #appending the dimension that we are reading out
            hx = torch.cat((hx,i*torch.ones(size = (hx.shape[0],1))))



            #Choose the action 
            if solution is None:
                action,logp = self.readout(latent_embedding,greedy = greedy,force = teacher_forcing,solution = None,latent = True,num = i)
            else:
                action,logp = self.readout(latent_embedding,greedy = greedy,force = teacher_forcing,solution = solution[:,i],latent = True,num = i)

            actions.append(action)
            logps.append(logp)

            action_embedding = self.action_embedder(action)

            hx = self.gru(action_embedding,hx)


        solution = torch.cat(tuple(actions),dim=1)
        logps = torch.cat(tuple(logps),dim=1)
        solution_logp = torch.sum(logps,dim=1).unsqueeze(dim=1)


        return None,solution, solution_logp,None,None









#General function solver
#This can't be deterministic it has to be a function now
class Improver(nn.Module):
    def __init__(self,latent_size,num_var,deg):
        super(Improver, self).__init__()

        self.latent_size = latent_size
        self.output_dim = 40
        #self.num_dim = 2
        self.embed_dim = 128

        self.num_var = num_var
        self.deg = deg


        #Parameterised Components

        self.encoder1 =  self.get_layers(self.latent_size,self.embed_dim,act_output = True)
        self.encoder2 =  self.get_layers(1,self.embed_dim,act_output = True)
        self.encoder3 =  self.get_layers(self.deg*self.num_var,self.embed_dim,act_output = True)
        self.encoder4 =  self.get_layers(3*self.embed_dim,self.embed_dim,act_output = True)

        self.mu=  nn.Linear(self.embed_dim,latent_size)
        self.logvar=  nn.Linear(self.embed_dim,latent_size)


    def get_layers(self,input_dim,output_dim,act_output,num = 0):

        layers = []

        if num == 0:
            layers.append(nn.Linear(input_dim,output_dim))
        else:
            layers.append(nn.Linear(input_dim,128))
            layers.append(nn.Tanh())

            for i in range(num):
                layers.append(nn.Linear(128,128))
                layers.append(nn.Tanh())

            layers.append(nn.Linear(128,output_dim))


        

        if act_output:
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def reparameterise(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std





    def forward(self,current_latents,target_latents,difference,context):

        #print(current_latents.shape)
        #print(difference.shape)
        #print(context.shape)



        current_latent_encoded = self.encoder1(current_latents)
        difference_encoded = self.encoder2(difference.unsqueeze(dim=1))
        context_encoded = self.encoder3(context[:,0:-1])

        data = torch.cat((current_latent_encoded,difference_encoded,context_encoded),dim=1)
        data = self.encoder4(data)


        mu = self.mu(data)
        log_sigma = self.logvar(data)


        assert not torch.isnan(mu.sum())
        assert not torch.isnan(log_sigma.sum())


        normal = torch.distributions.Normal(mu, torch.exp(log_sigma))

        if target_latents is not None:
            # Compute the log probability of the sample
            Z_logp = normal.log_prob(target_latents).sum(dim=1).unsqueeze(dim=1)
        else:
            Z_logp = None

        action = normal.sample()

        return Z_logp,action




class VAE_Solver(nn.Module):
    def __init__(self, config,diffusion_steps,num_var,deg,latent_size):
        super(VAE_Solver, self).__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size = self.latent_size,num_var = num_var,deg = deg)
        #self.decoder = Decoder(latent_size = self.latent_size)

        self.decoder = DiscreteDecoder(no_latent = False,num_var = num_var,deg = deg,latent_size = self.latent_size)

        self.improve_solution = Improver(latent_size = self.latent_size,num_var = num_var,deg = deg)

        self.diffusion = Diffusion(steps = diffusion_steps,latent_size = self.latent_size,num_var = num_var,deg = deg)

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


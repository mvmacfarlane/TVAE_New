import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
from torch.distributions import MultivariateNormal
from torch.distributions.uniform import Uniform
from diffusion import Diffusion
from torch.distributions import Categorical
from utils import get_layers
from torch.autograd import Variable
import torch.nn.functional as F



    
#General function solver
#This can't be deterministic it has to be a function now
class Improver(nn.Module):
    def __init__(self,latent_size,embed_dim,activation,num_var,deg):
        super(Improver, self).__init__()

        self.latent_size = latent_size
        self.embed_dim = embed_dim
        self.activation = activation
        self.num_var = num_var
        self.deg = deg


        #Parameterised Components
        self.context_encoder =  get_layers(input_dim = self.num_var*self.deg,output_dim = self.embed_dim,act_output = True,act_type = self.activation,num = 0)

        self.encoder1 =  get_layers(input_dim = self.latent_size,output_dim = self.embed_dim,act_output = True,act_type = self.activation,num = 0)

        self.encoder2 =  get_layers(input_dim = 2*self.embed_dim,output_dim = self.embed_dim,act_output = True,act_type = self.activation,num = 0)

        self.improver =  get_layers(input_dim = self.embed_dim,output_dim = 1,act_output = False,act_type = self.activation,num = 1)


    #predict if a movement from one vector to another is an improvement or not
    def predict_value(self,x_location):



        location_encoded = self.encoder1(x_location)
        #target_encoded = self.encoder2(torch.cat((self.encoder1(x_target),context_embedding),dim=1))

        #change = target_encoded - location_encoded

        pred_improvement = self.improver(location_encoded)

        return pred_improvement
    
        #predict if a movement from one vector to another is an improvement or not
    def better_value(self,x_location,value):

        location_encoded = self.encoder1(x_location)

        pred_improvement = self.improver(location_encoded)

        return pred_improvement
    

    #predict if a movement from one vector to another is an improvement or not
    def centre_improvement(self,x_target,context):
        return self.predict_value(x_target)

    #calculate the gradient of the prediction based on where we are and take a step in that directionxx
    #Question of whether this is the correct way to calculate things
    def improve(self,x_location,context):

        #x_location.requires_grad=True

        #x_location = Variable(x_location.data, requires_grad=True)

        total = x_location.shape[0]
        gradients = None
        



        for i in range(total):

            a = x_location[i:i+1,:]
            a.requires_grad=True

            improvement_pred = self.predict_improvement(a.detach(),a,context[i:i+1,:])

            improvement_pred.backward()

            gradient = a.grad

            if gradients is None:
                gradients = gradient
            else:
                gradients = torch.cat((gradients,gradient),dim=0)

        #print(new_location.shape)
        #print(gradients.shape)


        new_location = x_location + 10*gradients

        return new_location
    

    #This needs changed to batch form
    def get_grad(self,x_location,context,device):

        #x_location.requires_grad=True

        x_location = Variable(x_location.data,requires_grad = True) 

        #
        improvement_pred = self.predict_value(x_location)

        grad = torch.autograd.grad([improvement_pred.sum()], [x_location])[0]

        return grad.to(device)




    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad






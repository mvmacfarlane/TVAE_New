import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.uniform import Uniform
import utils


#Just getting rid of the advatnage multiplication
def calculate_weighted_RC_loss(solution_logp,advantage,weighted,weighting_temp):

    if weighted:

        #advantage = torch.exp(weighting_temp*(advantage.unsqueeze(dim=1)))
        #advantage = torch.clip(advantage.unsqueeze(dim=1), min=1, max=None)
        advantage = advantage.unsqueeze(dim=1)
        #advantage = advantage.masked_fill(advantage < 0,0)
        #advantage = advantage.masked_fill(advantage > 0,1)
        #advantage = torch.ones(size = advantage.shape).to(advantage.device)
        #advantage_mask = (advantage.unsqueeze(dim=1) > 0).int()
        #print(solution_logp*advantage)

        assert advantage.shape == solution_logp.shape


        RL = -(solution_logp).sum()
    else:
        RL = -(solution_logp).sum()

    return RL

def calculate_change_loss(solution_logp):

    Loss = -(solution_logp).mean()

    return Loss




#Just getting rid of the advatnage multiplication
def calculate_RL_loss(solution_logp,advantage):

    advantage = advantage.unsqueeze(dim=1)

    assert advantage.shape == solution_logp.shape

    RL = -(solution_logp*advantage).sum()

    return RL

def calculate_KLD_loss(mean, log_var,advantage,weighted,weighting_temp):


    x = (1 + log_var - mean.pow(2) - log_var.exp())
    x = x
    KLD = -0.5 * torch.sum(x)

    return KLD









def calculate_Improvement_loss(solution_logp_improved,decoded_solution,decoded_improved_solution,problem,batch_contexts):

    #print(solution_logp_improved)


    original_solution_cost = problem.cost_func(decoded_solution,batch_contexts)
    improved_solution_cost = problem.cost_func(decoded_improved_solution,batch_contexts)

    #improvement = (improved_solution_cost - original_solution_cost).detach()
    improvement = (improved_solution_cost.unsqueeze(dim=1)).detach()
    #improvement_normalise = (improvement - torch.mean(improvement))/torch.std(improvement)

    #distance = torch.sqrt(torch.sum(torch.pow((mu - mu_improved),exponent = 2),dim=1))

    #print(improvement[0:10])
    #print(solution_logp_improved[0:10])



    assert solution_logp_improved.shape == improvement.shape

    
    RL_loss = -(solution_logp_improved*improvement)



    total_loss = (RL_loss).sum()
    #total_loss = (distance).sum()

    return total_loss,((improved_solution_cost - original_solution_cost).detach()).mean()




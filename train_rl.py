import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.uniform import Uniform
import utils
import loss_functions
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from solvers.poly import optimal_solution
from tqdm import tqdm
import random
from utils import evaluation_plotting
from copy import deepcopy
import time




def train(model,config,problem,run_id):

    model_start = deepcopy(model)

    #Loading model
    if config.model_path is not None:
        model.load_state_dict(torch.load(config.model_path)['parameters'])

    #Setting up tensorboard
    runs_path = os.path.join(config.output_path_fixed,"runs",str(config.exp_name))
    writer = SummaryWriter(runs_path)

    #Training parameters
    params = list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)




    for epoch_idx in range(1, config.nb_epochs + 1):

        print("Epoch:{}".format(epoch_idx))

        #Training
        model,_,tracking_losses,total_time,total_time_generate = train_epoch(
            
            model,
            model_start,
            config,
            epoch_idx,
            optimizer,
            problem,

        )


        #Logging Training Variables
        for key, value in tracking_losses.items():
            writer.add_scalar(key, sum(value)/len(value), epoch_idx)

        #Validation
        mean_optimality_gap_1,mean_optimality_gap_all,mean_optimality_gap_1_greedy,total_time_evaluation= evaluate_epoch(model,config,problem)

        #Logging Evaluation Variables
        writer.add_scalar("Mean Evaluation Optimality Gap Shots:1", mean_optimality_gap_1, epoch_idx)
        writer.add_scalar("Greedy Mean Evaluation Optimality Gap Shots:1", mean_optimality_gap_1_greedy, epoch_idx)
        writer.add_scalar("Mean Evaluation Optimality Gap Shots:{}".format(config.test_retry_num), mean_optimality_gap_all, epoch_idx)

        writer.add_scalar("Epoch Generation time", total_time_generate, epoch_idx)
        writer.add_scalar("Epoch Training time", total_time, epoch_idx)
        writer.add_scalar("Epoch Evaluation time", total_time, epoch_idx)


    writer.close()



def train_epoch(
    
    model,
    model_start,
    config,
    epoch_idx,
    optimizer,
    problem,

    ):


    model.train()
    model_start.train()

    #VAE losses
    loss_RL_values = []

    #Sampling New Problem instances to train on
    problem.generate_new_problems(
        size = config.epoch_size,
    )

    contexts = problem.get_context()

    #Each problem gets 2 attmepts (generalise to N attempts)
    print("Sampling Solutions")

    start = time.time()

    #We need to change this to sampling from our decoder on non greedy
    with torch.no_grad():
        solutions,advantages = generate_solutions(
            
            model = model,
            epoch_size = config.epoch_size,
            config  = config,
            random = True,
            contexts = contexts,
            problem = problem,
            solution_num = config.sample_num,   #Solutions to sample per problem
            epoch_idx = epoch_idx,

        )

    total_time_generate = time.time()  -start



    device = solutions.device

    #Rpeat contexts for the ammount of solutions we have
    contexts = contexts.repeat(config.sample_num,1)
    contexts = contexts.to(device)

    #Calculate how many batches we need to get through all the solutions
    batch_num = math.floor(solutions.shape[0]/config.batch_size)


    #Storing the latent encodings and decoded solutions
    decoder_solutions = None

    #Shuffleing data
    indexes = torch.randperm(solutions.shape[0])
    solutions = solutions[indexes]
    advantages = advantages[indexes]
    contexts = contexts[indexes]

    print("Updating")

    start = time.time()

    for k in range(1):

        #Diffusion losses
        loss_RL_values = []


        for i in tqdm(range(batch_num)): 

            batch_solutions = solutions[i*config.batch_size:(i+1)*config.batch_size]
            batch_advantages = advantages[i*config.batch_size:(i+1)*config.batch_size]
            batch_contexts = contexts[i*config.batch_size:(i+1)*config.batch_size]


            #Encoder Decoder of sampled solutions
            _,solution_logp = model(
                
                context = batch_contexts,
                solution = batch_solutions,
                teacher_forcing = True,
                greedy = False,
    
            )


            #Reconstruction Loss
            loss_RL = loss_functions.calculate_RL_loss(
                
                solution_logp,
                batch_advantages,
     
            )

            loss = loss_RL

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_RL_values.append(loss_RL.item())


        print("RL Loss:{} Loop:{}".format(sum(loss_RL_values)/len(loss_RL_values),k))

    
    tracking_losses = {

        "loss_RL":loss_RL_values,

    }

    total_time = time.time() - start


        
    return model,solutions,tracking_losses,total_time,total_time_generate




#Ok we should have a batch of testing contexts within the problem which stay constant!!!
def evaluate_epoch(model,config,problem):

    start = time.time()

    #Hardcoded remove this asap
    device = "cuda:0"

    #Getting testing variables
    testing_contexts = problem.get_testing_context().to(device)
    testing_optimal_solutions = problem.get_testing_solutions().to(device)
    testing_num = testing_contexts.shape[0]


    print("Evaluating")

    reward_scores = None
    optimality_scores = None
    optimality_scores_2 = None
    optimality_scores_improved = None
    latent_samples = None
    solutions_storage = []


    #Now just decode these solutions
    greedy_solutions,_ = model(
            
            solution = None,
            greedy = True,
            teacher_forcing = False,
            context = testing_contexts,
                
    )


    for i in tqdm(range(config.test_retry_num)):

        #Now just decode these solutions
        solutions,_ = model(
                
                solution = None,
                greedy = False,
                teacher_forcing = False,
                context = testing_contexts,
                    
        )



        solutions_storage.append(solutions)

        sampled_reward = problem.cost_func(solution = solutions,context = testing_contexts).unsqueeze(dim=1)


        optimal_reward = problem.cost_func(solution = testing_optimal_solutions,context = testing_contexts)
        sampled_reward = problem.cost_func(solution = solutions,context = testing_contexts)

        assert torch.min(optimal_reward).item() >= 0
        
        optimality_gap = (optimal_reward - sampled_reward)/optimal_reward





        if optimality_scores is None:
            optimality_scores = optimality_gap.unsqueeze(dim=1)
            reward_scores = sampled_reward
   

        else:
            optimality_scores  = torch.cat((optimality_scores,optimality_gap.unsqueeze(dim=1)),dim=1)
            reward_scores = torch.cat((reward_scores,sampled_reward),dim=0)


    sampled_greedy_reward = problem.cost_func(solution = solutions,context = testing_contexts)
    optimal_reward = problem.cost_func(solution = testing_optimal_solutions,context = testing_contexts)
    greedy_optimality_gap = (optimal_reward - sampled_reward)/optimal_reward




    mean_optimality_gap_1_greedy = torch.mean(greedy_optimality_gap)


    best_scores,_ = torch.min(optimality_scores,dim=1)

    #Average one  shot sample
    mean_optimality_gap_1 = torch.mean(optimality_scores[:,0])

    #Average 10 shot sample
    mean_optimality_gap_all = torch.mean(best_scores)

    total_time = (time.time() - start)/config.test_retry_num

    return mean_optimality_gap_1,mean_optimality_gap_all,mean_optimality_gap_1_greedy,total_time

def round_to_05(tensor):
    return (torch.round(tensor * 20) / 20).float()
#Need to check this is correct and not mixing rewards to wrong solutions
def generate_solutions(model,epoch_size,config,random,contexts,solution_num,problem,epoch_idx):


    solutions  = []
    latents = []

    #Get rid of this hardcoding
    batch_size = config.generate_solutions_batch_size
    num_batches = int(epoch_size/batch_size)



    for j in range(solution_num):

        solution_set = None

        for i in tqdm(range(num_batches)):

            
            context_batch = contexts[i*batch_size:(i+1)*batch_size]

            solution_batch,_ = model(
                solution = None,
                greedy = False,
                teacher_forcing = False,
                context = context_batch.to(config.device),
                
            )

            if solution_set is None:
                solution_set = solution_batch
            else:
                solution_set = torch.cat((solution_set,solution_batch),dim=0)

        solutions.append(solution_set)

    rewards = [problem.cost_func(x,contexts).unsqueeze(dim=1) for x in solutions]
    rewards = torch.cat(tuple(rewards),dim=1)



    if solution_num == 1:
        mean_rewards = 0*torch.mean(rewards,dim=1,keepdim=True)
    else:
        mean_rewards = torch.mean(rewards,dim=1,keepdim=True)

    advantages = rewards - mean_rewards

    #Convert solutions and advatnages into one long vecotr
    solutions_t = torch.cat(tuple(solutions),dim=0)

    #transpose to ensure we do each set of solutions at a time which is how the solutions are stored
    advantages_t = torch.transpose(advantages, 0, 1).reshape((-1,))

    return solutions_t.detach(),advantages_t.detach()



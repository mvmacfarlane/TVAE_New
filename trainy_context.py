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
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    params_imp = list(model.improve_solution.parameters())
    params_diff = list(model.diffusion.parameters())

    optimizer_imp = torch.optim.Adam(params_imp, lr=config.lr_imp)
    #optimizer_diff = torch.optim.Adam(params_diff, lr=config.lr_diff)
    optimizer = torch.optim.Adam([{'params':params}, {'params':params_diff},{'params':params_imp}], lr=1e-3)

    optimizer.param_groups[0]['lr'] = config.lr
    optimizer.param_groups[1]['lr'] = config.lr_diff
    optimizer.param_groups[2]['lr'] = config.lr_imp

    #optimizer = torch.optim.Adam([{'params':list(model.parameters()) }], lr=1e-3)





    for epoch_idx in range(1, config.nb_epochs + 1):

        print("Epoch:{}".format(epoch_idx))

        #Reseting the optimizers



        #Training
        model,_,tracking_losses,plotting_data_decoder,plotting_data_sampled_solutions,plotting_data_correct_decoding,total_time,total_time_generation = train_epoch(
            
            model,
            model_start,
            config,
            epoch_idx,
            optimizer,
            optimizer_imp,
            #optimizer_diff,
            problem,

        )



        #Validation
        #Not implemented yet
        mean_optimality_gap_1,mean_optimality_gap_all,mean_optimality_gap_1_improved,mean_optimality_gap_all_improved,plotting,total_time_evaluation= evaluate_epoch(model,config,problem)

        #model = deepcopy(model_start)

        print("Plotting Data")

        #We only plot if there are two variables as we cannot plot two dimensions
        #Investigate methods we can use to visualise dimensions however
        if config.num_var == 2:

            #Training Plotting
            evaluation_plotting(plotting_data_decoder,writer,epoch_idx,"Training Decoder Decoding",min_val = -1 , max_val = 3,plot_name = "Decoder Decoding")
            evaluation_plotting(plotting_data_correct_decoding,writer,epoch_idx,"Training Perfect Decoding",min_val = -1 , max_val = 3,plot_name = "Perfect Decoding",density = True,context = True)
            evaluation_plotting(plotting_data_sampled_solutions,writer,epoch_idx,"Training Sampled Solutions",min_val = -1 , max_val = 3,limit_min = -1,limit_max = 1,plot_name = "Sampled Solution Space",density = True)



            #Evaluation Plotting
            #evaluation_plotting(plotting_optimality,writer,epoch_idx,"Evaluation Optimality Gap",min_val = 0 , max_val = 4,plot_name = "Evaluation Optimality Gap",plot_improver = False,evaluation = True)
            evaluation_plotting(plotting,writer,epoch_idx,"Evaluation Reward",min_val = -1 , max_val = 3,plot_name = "Evaluation",density = True,plot_improver = True,evaluation = True)


        #Logging Training Variables
        for key, value in tracking_losses.items():
            writer.add_scalar(key, sum(value)/len(value), epoch_idx)


        writer.add_scalar("Epoch Training time", total_time, epoch_idx)
        writer.add_scalar("Epoch Generation time", total_time_generation, epoch_idx)
        writer.add_scalar("Epoch Evaluation time", total_time_evaluation, epoch_idx)

        writer.add_scalar("Mean Evaluation Optimality Gap Shots:1", mean_optimality_gap_1, epoch_idx)
        writer.add_scalar("Mean Evaluation Optimality Gap Shots:{}".format(config.test_retry_num), mean_optimality_gap_all, epoch_idx)
        writer.add_scalar("Mean Evaluation Optimality Gap Shots:1 Improved", mean_optimality_gap_1_improved, epoch_idx)
        writer.add_scalar("Mean Evaluation Optimality Gap Shots:{} Improved".format(config.test_retry_num), mean_optimality_gap_all_improved, epoch_idx)


        
    #This needs tested and changed to saving a model every epoch
    """
    model_data = {
        'parameters': model.state_dict(),
        'problem': config.problem,
        'training_epochs': epoch_idx,
        'model': "VAE_final"
    }

    torch.save(
        
        model_data,
        os.path.join(config.output_path,
        "models",
        "model_{0}_final.pt".format(run_id, epoch_idx)),
    )
    """

    writer.close()





def train_epoch(
    
    model,
    model_start,
    config,
    epoch_idx,
    optimizer,
    optimizer_imp,
    #optimizer_diff,
    problem,

    ):


    model.train()
    model_start.train()

    #VAE losses
    loss_RC_values = []
    loss_KLD_values = []
    loss_D_values = []
    loss_D_values = []
    change_errors_values = []

    

    #Improver Losses
    loss_Imp_values = []
    improvement_change_values = []

    #Sampling New Problem instances to train on
    problem.generate_new_problems(
        size = config.epoch_size,
    )

    

    contexts = problem.get_context()



    #Each problem gets 2 attmepts (generalise to N attempts)
    print("Sampling Solutions")

    start = time.time()

    solutions,solutions_target,advantages,latents,latent_target= generate_solutions(
        
        model = model,
        epoch_size = config.epoch_size,
        config  = config,
        random = True,
        contexts = contexts,
        problem = problem,
        solution_num = config.sample_num,   #Solutions to sample per problem
        epoch_idx = epoch_idx,

    )

    total_time_generation = time.time() - start



    device = solutions.device

    #Rpeat contexts for the ammount of solutions we have
    contexts = contexts.repeat(config.sample_num,1)
    contexts = contexts.to(device)

    #Calculate how many batches we need to get through all the solutions
    batch_num = math.floor(solutions.shape[0]/config.batch_size)


    #Storing the latent encodings and decoded solutions
    latent_encodings = None
    decoder_solutions = None

    #Shuffleing data
    indexes = torch.randperm(solutions.shape[0])
    solutions = solutions[indexes]
    advantages = advantages[indexes]
    contexts = contexts[indexes]
    latents = latents[indexes]

    solutions_target = solutions_target[indexes]
    latent_target = latent_target[indexes]


    print("Updating")

    start = time.time()

    for k in range(config.diffusion_loops):

        #Diffusion losses
        loss_D_values = []
        loss_RC_values = []
        loss_Imp_values = []

        for i in tqdm(range(batch_num)): 

            batch_solutions = solutions[i*config.batch_size:(i+1)*config.batch_size]
            batch_solutions_target = solutions_target[i*config.batch_size:(i+1)*config.batch_size]

            batch_advantages = advantages[i*config.batch_size:(i+1)*config.batch_size]
            
            batch_contexts = contexts[i*config.batch_size:(i+1)*config.batch_size]
            batch_latents = latents[i*config.batch_size:(i+1)*config.batch_size]
            batch_latents_target = latent_target[i*config.batch_size:(i+1)*config.batch_size]

            #batch_calculated_rewards = problem.cost_func(solution = batch_solutions,context = batch_contexts)


            #Encoder Decoder of sampled solutions
            _,solution_logp,mu, log_var,Z,_ = model(
                
                context = batch_contexts,
                solution = batch_solutions,
                config = config,
                teacher_forcing = True,
                greedy = False,
    
            )

            #print("Hello")

            #Storing the decoded solutions
            if k == config.diffusion_loops-1:
                if latent_encodings is None:
                    latent_encodings = Z
                else:
                    latent_encodings = torch.cat((latent_encodings,Z),dim = 0)


            #Greedy decoding on latent of encoded solutions
            _,decoded_solution,_,_,_ = model.decoder(
            
                context = batch_contexts,
                solution = None,
                Z = Z,
                teacher_forcing = False,
                greedy = True, 

            )

            #print("Hello 1")
            #Is this actually needed
            #Storing the decoded solutions
            if k == config.diffusion_loops-1:
                if decoder_solutions is None:
                    decoder_solutions = decoded_solution
                else:
                    decoder_solutions = torch.cat((decoder_solutions,decoded_solution),dim = 0)


            #Learning the reward structure of the latent space
            rew_1 = problem.cost_func(solution = batch_solutions,context = batch_contexts) 
            rew_2 = problem.cost_func(solution = batch_solutions_target,context = batch_contexts)

            reward_diff = rew_1 - rew_2
            #1 step improvement on latent vector

            Z_improved_logp,Z_improved= model.improve_solution(
                
                current_latents = batch_latents.detach(),
                target_latents = batch_latents_target.detach(),
                difference = reward_diff.detach(),
                context = batch_contexts,
   
            )


            #Deocding
            _,decoded_solution_change,_,_,_ = model.decoder(
            
                context = batch_contexts,
                solution = None,
                Z = Z_improved,
                teacher_forcing = False,
                greedy = True, 

            )

            reward_target_pred= problem.cost_func(solution = decoded_solution_change,context = batch_contexts)

            mean_change_squared_error = (torch.pow((reward_target_pred - rew_2),2)).mean()







            #Diffusion Loss
            loss_D = model.diffusion.calculate_loss(x0 = Z.detach(),context = batch_contexts,advantage = 0*batch_advantages.unsqueeze(dim=1))

            #Reconstruction Loss
            loss_RC = loss_functions.calculate_weighted_RC_loss(
                
                solution_logp,
                batch_advantages,
                config.weighting,
                config.weighting_temp,
                
                
            )

            #KL divergence Loss
            loss_KLD = loss_functions.calculate_KLD_loss(
                
                mu,
                log_var,
                batch_advantages,
                config.weighting,
                config.weighting_temp,

            ) 





            loss_Imp = loss_functions.calculate_change_loss(

                solution_logp = Z_improved_logp,

            )

            #print("Loss CHange:{}".format(loss_Imp))

            



            loss = config.KLD_weight*loss_KLD + loss_RC
            #loss_2 = loss_Imp
            #loss_3 = 

            #assert not torch.isnan(loss)
            #assert not torch.isnan(loss_2)

            #Update Improver Parameters
            
            #optimizer_imp.zero_grad()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            #if k >= 1:
            #optimizer_diff.step()

            #Update Improver Parameters
            #if k == config.diffusion_loops - 1:
            #optimizer_imp.zero_grad()

            #if epoch_idx <= 10:
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            #else:
            #pass
            #    optimizer.zero_grad()
            #    loss_2.backward(retain_graph = False)
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            #    optimizer_imp.step()


            #Update encoder decoder parameters


            #print(model.encoder.output_head_mu.weight.grad) 


            #if k == 0:
            
            
            
            

            #Storing loss terms - 1

            loss_RC_values.append(loss_RC.item())
            loss_KLD_values.append(loss_KLD.item())
            loss_Imp_values.append(loss_Imp.item())
            #improvement_change_values.append(error.item())
            loss_D_values.append(loss_D.item())


            change_errors_values.append(mean_change_squared_error.item())


            


        print("Diffusion Loss:{} Loop:{}".format(sum(loss_D_values)/len(loss_D_values),k))
        print("Reconstruction Loss:{} Loop:{}".format(sum(loss_RC_values)/len(loss_RC_values),k))
        print("Change Loss:{} Loop:{}".format(sum(loss_Imp_values)/len(loss_Imp_values),k))
        print("Mean Squared Change Error:{} Loop:{}".format(sum(change_errors_values)/len(change_errors_values),k))

    



    plotting_data = {

        "latent_encoding": latent_encodings,
        "rewards": problem.cost_func(decoder_solutions,contexts),
        "x coord":decoder_solutions[:,0],
        "y coord":decoder_solutions[:,1],

    }

    plotting_data_sup = {

        "latent_encoding": solutions,
        "rewards": problem.cost_func(solutions,contexts),
        "x coord":solutions[:,0],
        "y coord":solutions[:,1],
    }

    binary_contexts = contexts[:,4]


    plotting_data_perfect = {

        "latent_encoding": latent_encodings,
        "context":binary_contexts,
        "rewards": problem.cost_func(solutions,contexts),
        "x coord":solutions[:,0],
        "y coord":solutions[:,1],
    }


    tracking_losses = {

        "loss_RC":loss_RC_values,
        "loss_KLD":loss_KLD_values,
        "loss_Imp":loss_Imp_values,
        "loss_D":loss_D_values,
        "Change errors":change_errors_values,

    }

    total_time = time.time() - start
        

    return model,solutions,tracking_losses,plotting_data,plotting_data_sup,plotting_data_perfect,total_time,total_time_generation




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

    with torch.no_grad():

        for i in tqdm(range(config.test_retry_num)):


            #This is not correct!
            #latent_sample = utils.generate_gaussian_vectors(testing_num).to(device)
            latent_sample = model.diffusion.generate_sample(testing_num,testing_contexts,advantage = 0*torch.ones(size = (testing_contexts.shape[0],1)),device = config.device).detach()


            #Now just decode these solutions
            _,solutions, _,_,_ = model.decoder.forward(
                    
                    solution = None,
                    Z = latent_sample,
                    greedy = True,
                    teacher_forcing = False,
                    context = testing_contexts,
                        
            )

            solutions_storage.append(solutions)

            sampled_reward = problem.cost_func(solution = solutions,context = testing_contexts).unsqueeze(dim=1)


            #Z_improved,_,_ = model.improve_solution(latent_sample,sampled_reward,testing_contexts)

            _,Z_improved= model.improve_solution(
                
                current_latents = latent_sample.detach(),
                target_latents = None,
                difference = 1*torch.ones(size = ( latent_sample.shape[0],)).to(latent_sample.device),
                context = testing_contexts,
   
            )



            #Now just decode these solutions
            _,solutions_improved, _,_,_ = model.decoder(
                    
                    solution = None,
                    Z = Z_improved,
                    greedy = True,
                    teacher_forcing = False,
                    context = testing_contexts,
                        
            )


            optimal_reward = problem.cost_func(solution = testing_optimal_solutions,context = testing_contexts)
            sampled_reward = problem.cost_func(solution = solutions,context = testing_contexts)
            
            sampled_improved_reward = problem.cost_func(solution = solutions_improved,context = testing_contexts)

            optimality_gap = optimal_reward - sampled_reward
            optimality_gap_improved = optimal_reward - sampled_improved_reward


            if optimality_scores is None:
                optimality_scores = optimality_gap.unsqueeze(dim=1)
                optimality_scores_improved = optimality_gap_improved.unsqueeze(dim=1)

                reward_scores = sampled_reward
                latent_samples = latent_sample
                optimality_scores_2 = optimality_gap
                

            else:
                optimality_scores  = torch.cat((optimality_scores,optimality_gap.unsqueeze(dim=1)),dim=1)
                optimality_scores_improved  = torch.cat((optimality_scores_improved,optimality_gap_improved.unsqueeze(dim=1)),dim=1)

                reward_scores = torch.cat((reward_scores,sampled_reward),dim=0)
                latent_samples = torch.cat((latent_samples,latent_sample),dim=0)
                optimality_scores_2 = torch.cat((optimality_scores_2,optimality_gap),dim=0)

        


    best_scores,_ = torch.min(optimality_scores,dim=1)
    best_scores_improved,_ = torch.min(optimality_scores_improved,dim=1)


    #Average one  shot sample
    mean_optimality_gap_1 = torch.mean(optimality_scores[:,0])
    mean_optimality_gap_1_improved = torch.mean(optimality_scores_improved[:,0])

    #Average 10 shot sample
    mean_optimality_gap_all = torch.mean(best_scores)
    mean_optimality_gap_all_improved = torch.mean(best_scores_improved)

    solutions_total = torch.cat(tuple(solutions_storage),dim=0)



    binary_contexts = testing_contexts[:,4]
    binary_contexts = binary_contexts.repeat(config.test_retry_num,)

    plotting_data = {

        "latent_encoding" : latent_samples,
        "context":binary_contexts,
        "optimality" : torch.transpose(optimality_scores, 0, 1).reshape((-1,)),
        "rewards" : reward_scores,
        "improved_latent_encoding":Z_improved,
        "x coord":solutions_total[:,0],
        "y coord":solutions_total[:,1],

    }

    total_time = (time.time() - start)/config.test_retry_num




    return mean_optimality_gap_1,mean_optimality_gap_all,mean_optimality_gap_1_improved,mean_optimality_gap_all_improved,plotting_data,total_time

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
        latent_set = None

        for i in tqdm(range(num_batches)):

            
            context_batch = contexts[i*batch_size:(i+1)*batch_size]

            #We only need this if wee decode
            with torch.no_grad():
                latent_sample = model.diffusion.generate_sample(batch_size,context_batch,advantage = 0*torch.ones(size = (context_batch.shape[0],1)),device = config.device).detach()

            #if config.sample_uniform:
            if epoch_idx <= 10000:


                solution_batch = utils.generate_uniform_vectors(number = batch_size,num_var = config.num_var).to(config.device)

                #Dealing with the discrete optimisation problem
                solution_batch = round_to_05(solution_batch)  



            else:
                _, solution_batch, _,_,_ = model.decoder.forward(
                    Z = latent_sample,
                    solution = None,
                    greedy = False,
                    teacher_forcing = False,
                    context = context_batch.to(config.device),
                    
                )



            if solution_set is None:
                solution_set = solution_batch
                latent_set = latent_sample
            else:
                solution_set = torch.cat((solution_set,solution_batch),dim=0)
                latent_set = torch.cat((latent_set,latent_sample),dim=0)

        solutions.append(solution_set)
        latents.append(latent_set)

    

    rewards = [problem.cost_func(x,contexts).unsqueeze(dim=1) for x in solutions]
    rewards = torch.cat(tuple(rewards),dim=1)



    if solution_num == 1:
        mean_rewards = 0*torch.mean(rewards,dim=1,keepdim=True)
    else:
        mean_rewards = torch.mean(rewards,dim=1,keepdim=True)

    advantages = rewards - mean_rewards

    #Convert solutions and advatnages into one long vecotr
    solutions_t = torch.cat(tuple(solutions),dim=0)
    latents_t = torch.cat(tuple(latents),dim=0)

    #transpose to ensure we do each set of solutions at a time which is how the solutions are stored
    advantages_t = torch.transpose(advantages, 0, 1).reshape((-1,))

    solutions.reverse()
    latents.reverse()

    solutions_target_t = torch.cat(tuple(solutions),dim=0)
    latents_target_t = torch.cat(tuple(latents),dim=0)





    return solutions_t.detach(),solutions_target_t.detach(),advantages_t.detach(),latents_t.detach(),latents_target_t.detach()



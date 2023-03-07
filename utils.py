import numpy as np
import zipfile
import pickle
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
from torch.distributions import MultivariateNormal
from torch.distributions.uniform import Uniform
from diffusion import Diffusion
from torch.distributions import Categorical

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


    plt.gcf().set_size_inches(8, 4)   # adjust the figure size as desired
    plt.tight_layout()      

    plt.savefig("data/test.png")
"""
def read_instance_data_tsp(problem_size, nb_instances, instance_file, solution_file, offset=0):




    instances = []
    solutions = []
    with zipfile.ZipFile(instance_file) as instance_zip:
        with zipfile.ZipFile(solution_file) as solution_zip:
            instances_list = instance_zip.namelist()
            solutions_list = solution_zip.namelist()
            assert len(instances_list) == len(solutions_list)
            instances_list.sort()
            solutions_list.sort()
            i = offset
            while len(instances) < nb_instances:
                if instances_list[i].endswith('/'):
                    i += 1
                    continue

                #Read instance data
                f = instance_zip.open(instances_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                instance = np.zeros((problem_size, 2))
                ii = 0
                while not lines[ii].startswith("NODE_COORD_SECTION"):
                    ii += 1
                ii += 1
                header_lines = ii
                while ii < len(lines):
                    line = lines[ii]
                    if line == 'EOF':
                        break
                    line = line.replace('\t', " ").split(" ")
                    x = line[1]
                    y = line[2]
                    instance[ii-header_lines] = [x, y]
                    ii += 1

                instance = np.array(instance) / 1000000
                instances.append(instance)

                #Read solution data
                f = solution_zip.open(solutions_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                tour = [int(l) for ll in lines[1:] for l in ll.split(' ')]

                solutions.append(tour)
                i += 1

    return instances, solutions


def read_instance_data_cvrp(problem_size, nb_instances, instance_file, solution_file, offset=0):
    instances = []
    solutions = []
    with zipfile.ZipFile(instance_file) as instance_zip:
        with zipfile.ZipFile(solution_file) as solution_zip:
            instances_list = instance_zip.namelist()
            solutions_list = solution_zip.namelist()
            assert len(instances_list) == len(solutions_list)
            instances_list.sort()
            solutions_list.sort()
            i = offset
            while len(instances) < nb_instances:
                if instances_list[i].endswith('/'):
                    i += 1
                    continue

                #Read instance data
                f = instance_zip.open(instances_list[i], "r")
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                ii = 0

                while ii < len(lines):
                    line = lines[ii]
                    if line.startswith("DIMENSION"):
                        dimension = int(line.split(':')[1])
                    elif line.startswith("CAPACITY"):
                        capacity = int(line.split(':')[1])
                    elif line.startswith('NODE_COORD_SECTION'):
                        locations = np.loadtxt(lines[ii + 1:ii + 1 + dimension], dtype=float)
                        ii += dimension
                    elif line.startswith('DEMAND_SECTION'):
                        demand = np.loadtxt(lines[ii + 1:ii + 1 + dimension], dtype=float)
                        ii += dimension

                    ii += 1

                locations = locations[:, 1:] / 1000000
                demand = demand[:, 1:] / capacity
                loads = np.ones((len(locations), 1))
                instance = np.concatenate((locations, loads, demand), axis=1)
                instances.append(instance)

                #Read solution data
                f = solution_zip.open(solutions_list[i], "r")
                solution = []
                lines = [str(ll.strip(), 'utf-8') for ll in f]
                ii = 0
                while ii < len(lines):
                    line = lines[ii]
                    ii += 1
                    if not line.startswith("Route"):
                        continue
                    line = line.split(':')[1]
                    tour = [int(l) for l in line[1:].split(' ')]
                    solution.append(tour)


                solutions.append(solution)
                i += 1

    return instances, solutions

def read_instance_data(config):
    offset = max(config.network_validation_size, config.search_validation_size)

    if config.problem == "TSP":
        training_data = read_instance_data_tsp(config.problem_size, config.epoch_size, config.instances_path,
                                           config.solutions_path, offset)
        validation_data = read_instance_data_tsp(config.problem_size, offset, config.instances_path,
                                             config.solutions_path)
    elif config.problem == "CVRP":
        training_data = read_instance_data_cvrp(config.problem_size, config.epoch_size, config.instances_path,
                                               config.solutions_path, offset)
        validation_data = read_instance_data_cvrp(config.problem_size, offset, config.instances_path,
                                                 config.solutions_path)
    return training_data, validation_data



def read_instance_pkl(config):
    with open(config.instances_path, 'rb') as f:


        instances_data = pickle.load(f)

    if config.problem == "TSP":
        return instances_data
    elif config.problem == "CVRP":
        instances = []
        for instance in instances_data:
            instance_np = np.zeros((config.problem_size + 1, 4))
            instance_np[0, :2] = instance[0]  # depot location
            instance_np[1:, :2] = instance[1]  # customer locations
            instance_np[:, 2] = 1  # loads
            instance_np[1:, 3] = np.array(instance[2]) / instance[3]  # customer demands
            instance_np[0, 3] = 0  # depot demand
            instances.append(instance_np)
        return instances

"""

import torch
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.distributions.uniform import Uniform


"""
def evaluation_grapher(
    
    encoded_mu,
    uniform_latents,
    improved_latents,
    improved_gaussian_latents,
            
    original_solutions,
    decoded_original_solutions,
    decoded_uniform_solutions,

    writer,
    epoch_idx,
    problem,

    contexts,
    
):

    #Will need to investigate how I am going decide the ranges. Latent space will always be in similar range so this shouldnt be an issue anyway

    #x_lim = 1
    #y_lim = 1

    original_solution_cost = problem.cost_func(original_solutions,contexts).to("cpu")
    decoded_solution_cost = problem.cost_func(decoded_original_solutions,contexts).to("cpu")
    decoded_uniform_solution_cost = problem.cost_func(decoded_uniform_solutions,contexts).to("cpu")


    #Latent Spaces
    latent_space_perfect_decoding = torch.cat((encoded_mu.to("cpu"),original_solution_cost.unsqueeze(dim=1)),dim=1)
    latent_space_decoder_decoding = torch.cat((encoded_mu.to("cpu"),decoded_solution_cost.unsqueeze(dim=1) ),dim=1)
    uniform_latent_space_decocder_decoding = torch.cat((uniform_latents.to("cpu"),decoded_uniform_solution_cost.unsqueeze(dim=1)),dim=1)



    latent_spaces = [latent_space_perfect_decoding,latent_space_decoder_decoding,uniform_latent_space_decocder_decoding]
    latent_spaces = [pd.DataFrame(x.detach().numpy(),columns = ['x','y','ov']) for x in latent_spaces]
    latent_names = ["latent_space_perfect_decoding","latent_space_decoder_decoding","uniform_latent_space_decocder_decoding"]


    figs = []

    #Log latents to tensorboard
    for i,space in enumerate(latent_spaces):
        name = "Figure:{}".format(i)
        fig, ax = plt.subplots()

        x = sns.scatterplot(data=space, x="x", y="y", hue="ov",hue_norm = (problem.min,problem.max),palette = "icefire",s=20)

        ax.set_ylim(-6,6)
        ax.set_xlim(-6,6)

        ax.set_title(latent_names[i])

        figs.append(fig)

        plt.close(fig)
    

    #Creating a density plot of the points
    
    fig, ax = plt.subplots()
    plt.hist2d(data=latent_spaces[0], x="x", y="y", bins=(50, 50), cmap=plt.cm.jet)
    figs.append(fig)
    ax.set_title("Encoder Density")
    ax.set_ylim(-4,4)
    ax.set_xlim(-4,4)
    plt.close(fig)
    

    
    #First create the original latent space
    fig, ax = plt.subplots()
    x = sns.scatterplot(data=latent_spaces[1], x="x", y="y", hue="ov",hue_norm = (problem.min,problem.max),palette = "icefire",s=20)
    ax.set_ylim(-6,6)
    ax.set_xlim(-6,6)
    ax.set_title("Improvement")

    #Plotting Improvements
    coordinates_from = encoded_mu.tolist()
    coordinates_to = improved_latents.tolist()


    for i,j in zip(coordinates_from[0:10],coordinates_to[0:10]):

        ax.annotate(
            '', 
            xy=tuple(i),
            #xycoords='data',
            xytext=tuple(j),
            #textcoords='data',
            arrowprops=dict(arrowstyle= '<|-, head_width=1',color='black',lw=2,ls='-')
        )
    figs.append(fig)
    plt.close(fig)




    #First create the original latent space
    fig, ax = plt.subplots()
    x = sns.scatterplot(data=latent_spaces[2], x="x", y="y", hue="ov",hue_norm = (problem.min,problem.max),palette = "icefire",s=20)
    ax.set_ylim(-6,6)
    ax.set_xlim(-6,6)
    ax.set_title("Improvement Gaussian")

    #Plotting Improvements
    coordinates_from = uniform_latents.tolist()
    coordinates_to = improved_gaussian_latents.tolist()


    for i,j in zip(coordinates_from[0:10],coordinates_to[0:10]):

        ax.annotate(
            '', 
            xy=tuple(i),
            #xycoords='data',
            xytext=tuple(j),
            #textcoords='data',
            arrowprops=dict(arrowstyle= '<|-, head_width=1',color='black',lw=2,ls='-')
        )
    figs.append(fig)
    plt.close(fig) 





    writer.add_figure(name, figs, epoch_idx)
    writer.add_scalar("Mean Uniform Decoding Cost", torch.mean(decoded_uniform_solution_cost), epoch_idx)
    writer.add_scalar("Max Uniform Decoding Cost", torch.max(decoded_uniform_solution_cost), epoch_idx)
    writer.add_scalar("Min Uniform Decoding Cost", torch.min(decoded_uniform_solution_cost), epoch_idx)

    #writer.close()
"""
    

def generate_uniform_vectors(number,num_var):

    coordinates = Uniform(-1,1).sample((number,num_var)) 

    return coordinates

def generate_gaussian_vectors(number):


    coordinates = torch.normal(mean = 0,std = 1,size = (number,2))


    return coordinates


#This should be moved to utils
def evaluation_plotting(plotting_data,writer,epoch_idx,name,min_val,max_val,limit_min = None,limit_max = None,plot_name = "",density = False,plot_improver = False,evaluation = False,context = False):

    figs = []
    #names = []

    latent = plotting_data["latent_encoding"].to("cpu")

    #limit_max = 15
    #limit_min = -15

    
    if limit_max is None:
        limit_max = torch.max(latent).item()
    if limit_min is None:
        limit_min = torch.min(latent).item()
    

    #for data in ["x coord","y coord","decoding optimality"]:
    #    print(plotting_data[data].to("cpu").shape)
    #Not sure what these things are all for, do we need them all now
    if evaluation:
        eval = ["x coord","y coord","optimality","rewards","context","predicted_rewards"]
    elif context:
        eval = ["x coord","y coord","rewards","context"]
    else:
        eval = ["x coord","y coord","rewards"]


    #decoded_performance = plotting_data["decoding optimality"].to("cpu") ["x coord","y coord","decoding optimality"]:
    for lab in eval:


        decoded_performance = plotting_data[lab].to("cpu")

        latent_space_decoder_decoding = torch.cat((latent,decoded_performance.unsqueeze(dim=1) ),dim=1)
        latent_space_decoder_decoding = pd.DataFrame(latent_space_decoder_decoding.detach().numpy(),columns = ['x','y','ov'])


        if min_val is False:
            min_val = torch.min(decoded_performance).item()
            max_val = torch.max(decoded_performance).item()

        #Log latents sampling to tensorboard
        fig, ax = plt.subplots()
        x = sns.scatterplot(data=latent_space_decoder_decoding , x="x", y="y", hue="ov",hue_norm = (min_val,max_val),palette = "icefire",s=20)
        ax.set_ylim(limit_min,limit_max)
        ax.set_xlim(limit_min,limit_max)
        ax.set_title(lab)

        if plot_improver:
            improved_latents = plotting_data["improved_latent_encoding"].to("cpu")

            coordinates_from = latent.tolist()
            coordinates_to = improved_latents.tolist()

            for i,j in zip(coordinates_from[0:10],coordinates_to[0:10]):

                ax.annotate(
                    '', 
                    xy=tuple(i),
                    #xycoords='data',
                    xytext=tuple(j),
                    #textcoords='data',
                    arrowprops=dict(arrowstyle= '<|-, head_width=1',color='black',lw=2,ls='-')
                )


        #names.append(name)
        figs.append(fig)
        plt.close(fig)



    if density:
        fig, ax = plt.subplots()
        plt.hist2d(data=latent_space_decoder_decoding, x="x", y="y", bins=(50, 50), cmap=plt.cm.jet)
        ax.set_title("Encoder Density")
        ax.set_ylim(limit_min,limit_max)
        ax.set_xlim(limit_min,limit_max)

        #names.append(name + ":Density")
        figs.append(fig)
        plt.close(fig)

    writer.add_figure(name, figs, epoch_idx)
    


def get_layers(input_dim,output_dim,act_output,act_type,num = 0):

    layers = []

    if num == 0:
        layers.append(nn.Linear(input_dim,output_dim))
    else:
        layers.append(nn.Linear(input_dim,128))

        if act_type == "tanh":
            layers.append(nn.Tanh())
        elif act_type == "Relu":
            layers.append(nn.ReLU())
        else:
            raise Exception("Error")

        for i in range(num):
            layers.append(nn.Linear(128,128))

            if act_type == "tanh":
                layers.append(nn.Tanh())
            elif act_type == "Relu":
                layers.append(nn.ReLU())
            else:
                raise Exception("Error")


        layers.append(nn.Linear(128,output_dim))

    if act_output:
        if act_type == "tanh":
            layers.append(nn.Tanh())
        elif act_type == "Relu":
            layers.append(nn.ReLU())
        else:
            raise Exception("Error")

    return nn.Sequential(*layers)
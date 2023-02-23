import torch
from solvers.poly import optimal_solution
import pickle


class ContextProblem():

    def __init__(self,testing_num,variation,num_var,deg):
        
        self.context = None

        #Parameters that determine the solutions we want
        self.variation = variation
        self.num_var = num_var
        self.deg = deg
        self.testing_num = testing_num

        
        try:
            raise Exception("data not available")
            file = open('data/test/testing_data.pkl', 'rb')
            file = pickle.load(file)
            self.testing_context,self.testing_optimal_solutions,variation,num_var,deg = file

            if variation != self.variation or num_var != self.num_var:
                raise Exception("data not available")

        except:
            self.testing_context = self.sample_problems(size = testing_num)
            self.testing_optimal_solutions = optimal_solution(self.testing_context,num_var = self.num_var,deg = self.deg)

            

        data = [self.testing_context,self.testing_optimal_solutions,self.variation,self.num_var,self.deg]

        file = open('data/test/testing_data.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
    
    def sample_problems(self,size):
        if self.variation:
            #Just considering two different types of problems

            #return 2*torch.rand(size = (size,5))
            #return 2*torch.normal(mean = 0, std = 1,size = (size,(self.num_var*self.deg) + 1))
            single_problem_1 = torch.Tensor([-1,0,0,2,0])
            #single_problem_2 = torch.Tensor([1,0,0,2,1])
            #single_problem_3 = torch.Tensor([0,1,2,0,2])
            #single_problem_4 = torch.Tensor([0,-1,2,0,3])

            #single_problem_1 = torch.Tensor([2,-1,-1,-1,0])
            #single_problem_2 = torch.Tensor([1,-1,-1,-1,1])
            #single_problem_3 = torch.Tensor([-2,-1,-1,-1,2])

            return single_problem_1.repeat(size,1)
            """
            single_problem_4 = torch.Tensor([-1,-1,-1,-1,3])

            batch_1 = single_problem_1.repeat(int(size/4),1)
            batch_2 = single_problem_2.repeat(int(size/4),1)
            batch_3 = single_problem_3.repeat(int(size/4),1)
            batch_4 = single_problem_4.repeat(int(size/4),1)

            return torch.cat((batch_1,batch_2,batch_3,batch_4),dim=0)
            #return torch.cat((batch_2,batch_3),dim=0)
            return single_problem_3.repeat(size,1)
            """
        else:
            single_problem = torch.Tensor([1,0,0,2])
            return single_problem.repeat(size,1)

    def generate_new_problems(self,size):
        self.context = self.sample_problems(size)

    def get_context(self):
        if self.context is None:
            raise Exception("Problems have not been sampled yet")
        else:
            return self.context

    def get_testing_context(self):
        return self.testing_context

    def get_testing_solutions(self):
        return self.testing_optimal_solutions

    #Problem is you need to say what problems you are solving here
    #I think it might be better if we just take the contexts into the function here
    def cost_func(self,solution,context):

        reward = ((context[:,0:self.num_var*self.deg].to(solution.device)*self.create_poly_t(solution,self.deg)).sum(dim=1))

        return reward


    def create_poly_t(self,var,deg):

        total = var

        for i in range(2,deg+1):
            total = torch.cat((total,torch.pow(var,i)),dim=1)
            
        return total


        







class Problem():

    def __init__(self,type):

        self.type = type

        if self.type == "chess":
            self.min = 0
            self.max = 1
        elif self.type == "smooth_1":
            self.min = -2
            self.max = 3
        else:
            raise Exception("Loss function not available")

    def cost_func(self,solution):

        if self.type == "chess":
            return self.chess(solution)
        elif self.type == "smooth_1":
            return self.smooth_1(solution)
        else:
            raise Exception("Loss function not available")




    def chess(self,solution):

        x1 = solution[:,0]
        x2 = solution[:,1]

        
        a = x1 < -0.8
        b = torch.logical_and(-0.6 < x1,x1 < -0.4)
        c = torch.logical_and(-0.2 < x1,x1 < 0)
        d = torch.logical_and(0.2 < x1,x1 < 0.4)
        e = torch.logical_and(0.6 < x1,x1 < 0.8)

        x_condition = torch.logical_or(torch.logical_or(torch.logical_or(a,b),torch.logical_or(c,d)),e)

        a = x2 < -0.8
        b = torch.logical_and(-0.6 < x2,x2 < -0.4)
        c = torch.logical_and(-0.2 < x2,x2 < 0)
        d = torch.logical_and(0.2 < x2,x2 < 0.4)
        e = torch.logical_and(0.6 < x2,x2 < 0.8)

        y_condition = torch.logical_or(torch.logical_or(torch.logical_or(a,b),torch.logical_or(c,d)),e)


        total = torch.logical_and(x_condition,y_condition)


        reward = total.double()
        
        return reward


    def smooth_1(self,solution):

        x1 = solution[:,0]
        x2 = solution[:,1]

        reward = 2*x1 + x2*x2

        return reward
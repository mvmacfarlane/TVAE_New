import torch
from scipy.optimize import minimize, rosen, rosen_der
from tqdm import tqdm


def optimal_solution(context,num_var,deg):

    print("Finding Optimal Solutions")

    num_context = context.shape[0]


    solutions = torch.zeros(size = (context.shape[0],num_var))

    for i in tqdm(range(num_context)):

        #This needs to be created using a loop
        function = []

        def create_poly(var,deg):

            total = []

            for i in range(1,deg+1):
                for x in var:
                    total.append(x**i )

            return torch.Tensor(total)


        fun = lambda x: (-(context[i,0:num_var*deg]*create_poly(list(x),deg)).sum()).item()

        #Maybe I need to allow the bounds to hit the number
        #bnds = ((-1, 1),(-1, 1))

        bnds = tuple([(-1,1) for i in range(num_var)])
        initial = tuple([0]*num_var)


        res = minimize(fun, initial, method='nelder-mead', bounds=bnds)


        solutions[i,:] = torch.Tensor(res.x)

    return solutions
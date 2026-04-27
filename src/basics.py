import torch 

class KnotVector():
    def __init__(self,knots,
                 multiplicities = None,
                 degree         = 2,
                 open           = False):
        """Knot vector class."""
        self.device        = knots.device
        self.degree        = degree
        # Check whether vector consists of nondecreasing values
        assert (knots.sort().values == knots).all()
        # Individual knots without multiplicities
        self.knots = knots
        if multiplicities is None:
            self.multiplicities = torch.ones_like(knots, dtype = torch.int32)
        else:
            self.multiplicities = multiplicities
        if open == False: 
            self.multiplicities[[0,-1]]+=degree
        self.m = self.multiplicities.sum()-1  # {u_0, u_1, ... , u_m}
        self.n = self.m-self.degree-1  # {P_0, P_1, ... , P_n}
        self.open = open
        self.knot_indices = self.multiplicities.cumsum(dim=0)-1
        self.full_knots = torch.repeat_interleave(self.knots,self.multiplicities)
        # self.full_knots_pad = torch.cat((self.full_knots,self.full_knots[-1:]))
    def find_span_batch(self, u):
        return FindSpanBatch(self.degree,u,self.full_knots, self.open)

    def eval_u_batch(self,u):
        b_size = len(u)
        spans = self.find_span_batch(u) # <N_u>
        # print('debug1: ', spans)
        Ns = []
        Ns.append([torch.ones(u.shape, device = self.device)]) # [<B>]
        for p in range(1, self.degree+1):
            N_pad = [torch.zeros(b_size, device = self.device)] + Ns[-1] + [torch.zeros(b_size, device = self.device)] # [0.,1.,0.]
            N_new = []
            for j in range(p+1):
                ii = spans-p+j # <B> 
                # print('debug2', p,j,ii, self.full_knots.shape)
                N_new.append(Left(u,self.full_knots[ii],self.full_knots[ii+p])*N_pad[j] + \
                             Right(u,self.full_knots[ii+1],self.full_knots[ii+p+1])*N_pad[j+1])
                # N_new.append(Left(u,self.full_knots_pad[ii],self.full_knots_pad[ii+p])*N_pad[j] + \
                #              Right(u,self.full_knots_pad[ii+1],self.full_knots_pad[ii+p+1])*N_pad[j+1])
            Ns.append(N_new)
        return Ns #<B,>
    
    def get_basis_function(self,u):
        b_size = len(u)
        Ns = self.eval_u_batch(u)
        spans = self.find_span_batch(u) # <N_u>
        basis_functions = []
        for i in range(1, self.degree+2):
            # basises = torch.zeros((self.m-i+1+1, b_size), device = self.device)
            basises = torch.zeros((self.m-i+1, b_size), device = self.device)
            for j in range(i):
                for b in range(b_size):
                    basises[spans[b]-i+1+j,b]=Ns[i-1][j][b]
            basis_functions.append(basises)
        return basis_functions
    # def get_basis_function(self,u):
    #     b_size = len(u)
    #     Ns = self.eval_u_batch(u)
    #     spans = self.find_span_batch(u) # <N_u>
    #     basis_functions = []
    #     for i in range(1, self.degree+2):
    #         basises = torch.zeros((self.m-i+1+1, b_size), device = self.device)
    #         # basises = torch.zeros((self.m-i+1, b_size), device = self.device)
    #         for j in range(i):
    #             for b in range(b_size):
    #                 basises[spans[b]-i+1+j,b]=Ns[i-1][j][b]
    #         basis_functions.append(basises)
    #     return basis_functions

class PerodicKnotVector(KnotVector):
    def __init__(self,Nu,
                 degree         = 2,
                 ):
        du = 1/Nu
        self.extented_knots = torch.linspace(0.-degree*du,1.+degree*du,Nu+1+2*degree)
        self.original_knots = self.extented_knots[degree:-degree]
        super().__init__(self.extented_knots,
                        degree = degree,
                        multiplicities = None,
                        open           = True
                        )
        

def CreateKnotVector(sequence,p=1):
    return torch.cat((torch.repeat_interleave(sequence[0],p),sequence,torch.repeat_interleave(sequence[-1],p)))

def FindSpanBatchold(p,u,U):
    '''
    inputs:
        p, <0> order of spline
        u, <n> query u values 
        U, <m+1> knot vector 
    output:
        spans, <n> spans for the query values 
    '''
    return ((U[:-(p+1)]>u.unsqueeze(-1))==False).sum(-1)-1

def FindSpanBatch(p,u,U,open = False):
    '''
    inputs:
        p, <0> order of spline
        u, <n> query u values 
        U, <m+1> knot vector 
    output:
        spans, <n> spans for the query values 
    '''
    return ((U[:-(p+1)]>u.unsqueeze(-1))==False).sum(-1)-1

    # return torch.searchsorted(U[:-(p+1)],u,right=True)-1
    # if open == False:
    #     return torch.searchsorted(U[:-(p+1)],u,right=True)-1
    # else: 
    #     # return torch.searchsorted(U[:-(p+1)],u,right=True)-1
    #     return torch.searchsorted(U[:-1],u,right=True)-1

def Left(u, lb, ub, tor = 1e-6):
        return (u-lb)/(ub-lb+tor)

def Right(u, lb, ub, tor = 1e-6):
        return (ub-u)/(ub-lb+tor)


def FindSpan(p,u,U):
    '''
    Algorithm 2.1

    Input:
        p, the order of the curve
        u, the parametric variable
        U, the knot vector 

    Output:
        B, basis function B0_n,  B1_n, ... , Bn_n
    '''
    assert u <= U[-1], 'u should be smaller than the last element in the knot vector '
    assert u >=0. , 'u should be bigger than zero '
    # print('findspan:', u)
    n = len(U)-1-p-1 # len(U) = m+1, so here n = m+1-1-p-1 = m-p-1 
    if u==U[n+1]:
        return n
    low = p
    high = n+1
    mid = int((low+high)/2)
    while u<U[mid] or u>=U[mid+1]:
        # print('lowmidhigh:', low,mid,high)
        if u<U[mid]:
            high = mid
        else:
            low = mid 
        mid = int((low+high)/2)
    return mid 


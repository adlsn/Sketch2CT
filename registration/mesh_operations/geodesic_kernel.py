import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import geodesic_distance as gd



class GeodesicConv(MessagePassing):
    def __init__(self,kernel_width = 1.):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.kernel_width = Parameter(torch.Tensor(1)) 
        with torch.no_grad():
            self.kernel_width.fill_(kernel_width)
        # self.reset_parameters()

    def reset_parameters(self):
        self.kernel_width.reset_parameters()
        # self.bias.data.zero_()

    def forward(self, data, hop_size=1):
        edge_index = data.edge_index
        if hop_size>1:
            # extending the adj matrix 
            ## self loop must be added ahead 
            N = len(data.x)
            adj = torch.zeros(N, N, dtype=torch.long)
            adj[data.edge_index[0], data.edge_index[1]] = 1        
            new_adj = torch.matrix_power(adj, hop_size)
            new_adj = (new_adj > 0).to(torch.long)
            edge_index = torch.nonzero(new_adj)

        # calculate geodesic distance 
        print('calculate gd')
        norm = gd(data.pos,data.face,edge_index[0][0:5],edge_index[1][0:5])
        # Step 4-5: Start propagating messages.
        print('propagate')
        out = self.propagate(edge_index, x=data.x, norm=norm)
        return out

    def message(self, x_j, norm):
        
        # Step 4: Normalize node features.
        return self.gaussian_kernel(norm).view(-1, 1) * x_j
    
    def gaussian_kernel(self, x):
        return torch.exp(-0.5*(x**2)/self.kernel_width**2)


# class point_filter():
#     def __init__(self,kernel_width = 1.):
#         super().__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.kernel_width = Parameter(torch.Tensor(1)) 
#         with torch.no_grad():
#             self.kernel_width.fill_(kernel_width)
#         # self.reset_parameters()

#     def reset_parameters(self):
#         self.kernel_width.reset_parameters()
#         # self.bias.data.zero_()

#     def forward(self, data, hop_size=1):
#         edge_index = data.edge_index
#         if hop_size>1:
#             # extending the adj matrix 
#             ## self loop must be added ahead 
#             N = len(data.x)
#             adj = torch.zeros(N, N, dtype=torch.long)
#             adj[data.edge_index[0], data.edge_index[1]] = 1        
#             new_adj = torch.matrix_power(adj, hop_size)
#             new_adj = (new_adj > 0).to(torch.long)
#             edge_index = torch.nonzero(new_adj)

#         # calculate geodesic distance 
#         print('calculate gd')
#         norm = gd(data.pos,data.face,edge_index[0][0:5],edge_index[1][0:5])
#         # Step 4-5: Start propagating messages.
#         print('propagate')
#         out = self.propagate(edge_index, x=data.x, norm=norm)
#         return out

#     def message(self, x_j, norm):
        
#         # Step 4: Normalize node features.
#         return self.gaussian_kernel(norm).view(-1, 1) * x_j
    
#     def gaussian_kernel(self, x):
#         return torch.exp(-0.5*(x**2)/self.kernel_width**2)




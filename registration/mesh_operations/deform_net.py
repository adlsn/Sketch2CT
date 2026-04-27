
# import torch_geometric
from torch_geometric.nn.conv import GCNConv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn



class GCN_rez(nn.Module):
    def __init__(self, input_sizes):
        super().__init__()
        self.num_of_layers = len(input_sizes)-1
        self.sizes = input_sizes
        convs = []
        for i in range(self.num_of_layers):
            convs.append(GCNConv(self.sizes[i],self.sizes[i+1]))
        self.convs = torch.nn.ModuleList(convs)

    def forward(self, x, edge_index):
        x0 = torch.clone(x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

 
class GCN_convblk(nn.Module):
    def __init__(self,feature_dim, hidden_dim, coord_dim, num_blocks):
        super().__init__()
        self.gcn0 = GCNConv(feature_dim, hidden_dim)
        gcn_rezs = []
        for i in range(num_blocks):
            gcn_rezs.append(GCN_rez([hidden_dim,hidden_dim,hidden_dim]))
        self.gcn_rezs = torch.nn.ModuleList(gcn_rezs)
        self.gcn1 = GCNConv(hidden_dim,coord_dim)

    def forward(self, x, edge_index):
        x = self.gcn0(x,edge_index)
        x_cat = F.relu(x)
        for i, conv in enumerate(self.gcn_rezs):
            x_cat = conv(x_cat,edge_index)
            x_cat = F.relu(x_cat)
        x = self.gcn1(x_cat,edge_index)    

        return x,x_cat

class GCN_deform(nn.Module):
    def __init__(self,coord_dim=3, scale = 1):
        super().__init__()
        self.scale = scale
        self.gcn0 = GCNConv(coord_dim, 384)
        self.gcn_convblk0 = GCN_convblk(384, 288, coord_dim, 3)
        
        self.gcn1 = GCNConv(288, 144)
        self.gcn_convblk1 = GCN_convblk(144, 96, coord_dim, 3)
        
        self.gcn2 = GCNConv(96,64 )
        self.gcn_convblk2 = GCN_convblk(64, 32, coord_dim, 3)
        
    def forward(self, x, edge_index):
        # x0 = torch.clone(x)
        x = self.gcn0(x,edge_index)
        output1,output_cat = self.gcn_convblk0(x,edge_index)
        # output1 = output1/256+x
        output1 = output1

        output = self.gcn1(output_cat,edge_index)
        output2,output_cat = self.gcn_convblk1(output,edge_index)
        # output2 = output2/256+output1
        output2 = output2+output1

        output = self.gcn2(output_cat,edge_index)
        output3,output_cat = self.gcn_convblk2(output,edge_index)
        # output3 = output3/256+output2
        output3 = output3+output2

        return output1*self.scale, output2*self.scale, output3*self.scale
    # def reset_parameters(self):
    #     dic =self.state_dict()
    #     for k in dic:
    #         dic[k] *= 0
    #     Model.load_state_dict(dic)
    #     del(dic)
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(name,param)
        
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.fill_(0.)

        # for name, param in self.named_parameters():
        #     param.fill_(0.)



from torch_geometric.nn import ChebConv
from torch_scatter import scatter_add
def CalPointNormal(pos, face):
    assert pos.shape[-1] == 3,'points coordinates needs to be 3'
    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]
    face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

    idx = torch.cat([face[0], face[1], face[2]], dim=0)
    face_norm = face_norm.repeat(3, 1)

    norm = scatter_add(face_norm, idx, dim=0, dim_size=pos.size(0))
    norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]
    return norm 

class SGCN_deform(nn.Module):
    def __init__(self,input_dim = 3, output_dim = 1, hidden_dim = 32, order = 1, scale = 1., dt = 1.):
        super().__init__()
        self.scale = scale
        self.dt = dt
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.conv1 = ChebConv(hidden_dim, output_dim, order)
        self.conv2 = ChebConv(hidden_dim, output_dim, order)
        self.conv3 = ChebConv(hidden_dim, output_dim, order)

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.linear3 = nn.Linear(input_dim, output_dim)


    def forward(self, x, edge_index, faces):
        x0 = torch.clone(x)
        pos = x
        dposes = []
        dpos = torch.zeros(pos.shape, device = x.device)
        for conv in self.convs: 
            dpos_n = conv(pos, edge_index)
            normal = CalPointNormal(pos,faces.T)
            # print(pos.shape, dpos_n.view(-1,1).shape, normal.shape)
            print('dpos_n: ',dpos_n)
            dpos += dpos_n.view(-1,1)*normal*self.scale* self.dt
            dposes.append(dpos)
            pos = x0 + dpos
        # print('dposes:', dposes)
        return dposes
    
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(name,param)
        
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.fill_(0.)


class ChebconvRez(nn.Module):
    def __init__(self,input_dim = 32, output_dim = 32,hidden_dim = 32, order = 1):
        super().__init__()

        self.conv1 = ChebConv(input_dim, hidden_dim, order)
        self.conv2 = ChebConv(hidden_dim, output_dim, order)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x0 = torch.clone(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x+x0
    
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(name,param)
        
    # def reset_parameters(self):
    #     with torch.no_grad():
    #         for name, param in self.named_parameters():
    #             param.fill_(0.)


class SGCN_deform_s1(nn.Module):
    def __init__(self,input_dim = 3, output_dim = 1, hidden_dim = 32, order = 1, scale = 1.):
        super().__init__()
        self.scale = scale
        # self.dt = dt
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.GNN_blk1 = ChebconvRez(hidden_dim,hidden_dim,hidden_dim,order)
        self.GNN_blk2 = ChebconvRez(hidden_dim,hidden_dim,hidden_dim,order)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x0 = torch.clone(x)
        # print(x)
        x = F.relu(self.linear0(x))
        # print(x)
        x = self.GNN_blk1(x, edge_index)
        # print(x)
        x = F.relu(x)
        # print(x)
        x = self.GNN_blk2(x, edge_index)
        # print(x)
        x = F.relu(x)
        # print(x)
        x = self.linear1(x)
        # print(x)
        # x = (x-torch.mean(x))/torch.std(x)
        x = x*self.scale
        # print(x)
        # normal = CalPointNormal(x0,faces.T)
        # dx = x.view(-1,1)*normal*self.scale* self.dt
        return x
    
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(name,param.shape)
        
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.fill_(0.)



class SGCN_deform_s3(nn.Module):
    def __init__(self,input_dim = 3, output_dim = 1, hidden_dim = 32, order = 1, scale = 1.):
        super().__init__()
        self.scale = scale
        # self.dt = dt
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.GNN_blk1 = ChebconvRez(hidden_dim,hidden_dim,hidden_dim,order)
        self.GNN_blk2 = ChebconvRez(hidden_dim,hidden_dim,hidden_dim,order)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x0 = torch.clone(x)
        print(x)
        x = F.relu(self.linear0(x))
        print(x)
        x = self.GNN_blk1(x, edge_index)
        print(x)
        x = F.relu(x)
        print(x)
        x = self.GNN_blk2(x, edge_index)
        print(x)
        x = F.relu(x)
        print(x)
        x = self.linear1(x)
        print(x)
        # x = (x-torch.mean(x))/torch.std(x)
        x = x*self.scale
        print(x)
        # normal = CalPointNormal(x0,faces.T)
        # dx = x.view(-1,1)*normal*self.scale* self.dt
        return x
    
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(name,param)
        
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.fill_(0.)













import numpy as np
import torch 
import pyvista as pv
import os, shutil
from os.path import exists as ose
from os.path import join as osj
import torch 
from registration.mesh_operations.normalizer import *
from registration.mesh_operations.transforms import *
from registration.mesh_operations.interpolation import *
from registration.mesh_operations.mesh2graph import vtk2GraphVertex
from registration.mesh_operations.deform_net import *
import pytorch3d as p3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.io import load_ply, save_ply 
from pytorch3d.structures import Meshes
from torch_geometric.nn import fps, knn


def QuadCurve(spt,ept,param, res = 100):
    '''
    input:
        spt: start point 
        ept: end point
    output: 
        quadratice curve. 
    '''
    t = torch.linspace(0,1,res, device = spt.device)
    t_2, t_0 = t**2, torch.ones(res, device = spt.device)
    t_matrix = torch.cat((t_2,t,t_0)).reshape(3,-1)
    # param = torch.nn.Parameter(torch.zeros(3), device = device))
    poly_matrix = torch.hstack((param.unsqueeze(-1),
                                (ept-param-spt).unsqueeze(-1),
                                spt.unsqueeze(-1)))
    return (poly_matrix @ t_matrix).t()

def BatchDotProduct(a,b):
    return torch.abs(torch.sum(a*b, dim = -1)/(torch.norm(a,dim=-1)*torch.norm(b,dim=-1)))

def FitQuadCurve(centers, vectors,credibilities, device = None,res = 200,
            epoch = 100, w_chamfer =0.5 , w_credibility = 0.5, lr= 1e-3,save_interval=100,output_dir ='./'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    param = torch.nn.Parameter(torch.zeros(3, device = device))
    optimizer = torch.optim.Adam([param], lr=lr, weight_decay=1e-4)

    # normalize the curve 
    x_normalizer = Normalizer_ts(method = 'ms', dim=0)
    centers_norm = x_normalizer.fit_normalize(centers)

    losses_chamfer = []
    losses_credibility = []
    total_losses = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    for j in range(epoch):
        optimizer.zero_grad()
        CPs_norm = QuadCurve(centers_norm[0],centers_norm[-1],param, res = res)
        # calculate the chamfer loss 
        loss_chamfer, _= chamfer_distance(CPs_norm.unsqueeze(0), centers_norm.unsqueeze(0),
                                        single_directional = False)
        losses_chamfer.append(loss_chamfer.item())
        
        # calcualte the intersection of the curve with the circles 
        # remember, you need to project them back to demorned space!!!
        N_neighbors = 5
        CPs_denorm = x_normalizer.denormalize(CPs_norm)
        # print('debug:',CPs_norm[1:-1].shape,centers_norm.shape)
        # print('debug00:', CPs_norm)
        # print('debug000:', centers_norm[1:-1])
        assign_index = knn(CPs_denorm,centers[1:-1],N_neighbors) #<2, 5*N>
        # assign_index = knn(CPs_norm,centers_norm[1:-1],N_neighbors) #<2, 5*N>
        # print('debug0',assign_index, CPs_norm.shape, centers_norm.shape)
        # diff_vectors = (centers_norm[1:-1][assign_index[0]] - CPs_norm[assign_index[1]]).reshape(-1,N_neighbors,3)
        diff_vectors = (centers[1:-1][assign_index[0]] - CPs_denorm[assign_index[1]]).reshape(-1,N_neighbors,3)
        # print('debug1:',diff_vectors.shape, vectors[1:-1].unsqueeze(1).shape)

        min_index = BatchDotProduct(vectors[1:-1].unsqueeze(1),diff_vectors).min(1)[1]
        # min_index = torch.abs(torch.sum((vectors[1:-1].unsqueeze(1))*diff_vectors,dim = -1)/()).min(1)[1]
        # print('min index', min_index)
        # print('min index', min_index[1])
        # multi_index = torch.hstack((torch.arange(len(min_index), device = device).unsqueeze(1), min_index.unsqueeze(1)))

        # print(multi_index.shape, assign_index[1].reshape(-1,5).shape)
        # closest_ids = assign_index[1].reshape(-1,5)[multi_index]
        closest_ids = assign_index[1].reshape(-1,5)[torch.arange(len(min_index), device = device),min_index]
        closest_pts = CPs_denorm[closest_ids]
        # print('closest ids shape',closest_ids.shape)
        # print('closest ids',closest_ids)
        loss_credibility = torch.sum(credibilities[1:-1]*torch.norm(centers[1:-1]-closest_pts, dim = -1))
        # print('wtf?????',torch.norm(centers[1:-1]-CPs_denorm[closest_ids], dim = -1))
        losses_credibility.append(loss_credibility.item())

        loss = w_chamfer*loss_chamfer + w_credibility*loss_credibility
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_losses.append(loss.item())
        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, epoch,loss.item()))
            # temp_mesh = pv.PolyData(rotated_points.detach().cpu().numpy())
            # temp_mesh = pv.PolyData(x_normalizer.denormalize(CPs_norm).detach().cpu().numpy())
            temp_mesh = pv.PolyData(CPs_denorm.detach().cpu().numpy())
            temp_mesh.point_data['closest'] = 0.
            temp_mesh.point_data['closest'][closest_ids.detach().cpu().numpy()] = 1.
            temp_mesh.save(osj(output_dir, 'fit_curve_{:d}.vtp'.format(j)))

            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            er0 = torch.tensor(total_losses)
            er1 = torch.tensor(losses_chamfer)
            er2 = torch.tensor(losses_credibility)
            ax.plot(er0,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
            ax.plot(er1,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='L_chamfer')
            ax.plot(er2,color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='L_credibility')
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    return closest_pts.detach()
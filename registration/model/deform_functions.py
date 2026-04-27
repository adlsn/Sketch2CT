
from registration.modules.my_lddmm import LDDMM, MakeNormloss, MakeVarifoldloss, MakeVarifoldloss_NURBS
from registration.utilities.others import vtpface2graphface,graphface2vtpface
import os
from os.path import exists as ose
from os.path import join as osj
import shutil 
from registration.mesh_operations.barycentric import *
from registration.mesh_operations.normalizer import *
from registration.mesh_operations.transforms import *
from registration.mesh_operations.interpolation import *
from registration.mesh_operations.mesh2graph import vtk2GraphVertex
from registration.mesh_operations.deform_net import *
import glob
import torch 
import pytorch3d as p3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.structures import Pointclouds as PCs
from pytorch3d.io import load_ply, save_ply 
from pytorch3d.structures import Meshes
import matplotlib
import pylab 
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.geometries import createAortaSurface, createAortaSurface2vtp

# lddmm class
def DeformLDDMM(source_mesh, target_points, device=None, x_normalizer=None, output_dir ='./',
                fps_ratio=0.6, epoch=500, lr= 1e-3, save_interval=100, initial_normal_factor=0.1, number_of_time_pts = 20, T = 1/8, 
                 w_chamfer= 1.,w_edge=1.,w_normal = 0.01,w_laplacian = 0.1,w_landmark = 0.01,w_strain= 0.,  #0.1,#1,#0.01,#0.1,
                 single_direction= True, landmark = None):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    source_points = torch.tensor(source_mesh.points,dtype = torch.float32, device = device)
    target_points = target_points.to(device)
    if x_normalizer == None: 
        x_normalizer = Normalizer_ts(method = 'ms', dim=0)
        _= x_normalizer.fit_normalize(torch.vstack([source_points, target_points]))
    source_faces = torch.tensor(vtpface2graphface(source_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
    source_points_norm = x_normalizer.normalize(source_points)
    target_points_norm = x_normalizer.normalize(target_points)
    initial_verts =source_points_norm
    normals = CalPointNormal(initial_verts,source_faces.t())
    aera0 = ComputeArea(initial_verts, source_faces)
    # from torch_geometric.nn import fps, knn
    import torch_cluster.fps as fps
    torch.manual_seed(0)
    indices = fps(source_points_norm, ratio=fps_ratio) # 0.6
    # indices = torch.arange(len(source_points_norm)) # 0.6
    deform_verts = torch.nn.Parameter(initial_normal_factor*normals[indices])
    optimizer = torch.optim.Adam([deform_verts], lr=lr, weight_decay=1e-4)
    # Number of optimization steps
    Niter = epoch
    w_chamfer =w_chamfer#  1.#0.1
    w_edge = w_edge# 1. #1
    w_normal = w_normal# 0.01#0.01
    w_laplacian = w_laplacian# 0.1#0.1
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    strain_losses = []

    if landmark != None: 
        landmark_losses = []
        w_landmark = w_landmark
        landmark_ids,landmark_points,landmark_credibility = landmark
        landmark_weights = 1/landmark_credibility
        # landmark_weights = torch.tesnor(landmark_weights/np.sum(landmark_weights),dtype = torch.float32, device = device) 
        # landmark_points = torch.tesnor(landmark_points,dtype = torch.float32, device = device)
        landmark_weights = (landmark_weights/torch.sum(landmark_weights)).to(device)
        landmark_points =x_normalizer.normalize(landmark_points.to(device))


    total_losses = []

    # criterion = torch.nn.MSELoss()

    
    for j in range(Niter+1):
        optimizer.zero_grad()
        my_deform = LDDMM(source_points_norm[indices], 
                        # normals[indices]*deform_verts.unsqueeze(1), 
                        deform_verts, 
                        initial_verts,
                        kernel_width = 0.9, #0.3, #0.17,#0.3,
                        T = T , #1/16.,##1./16,#1./256,
                        number_of_time_pts = number_of_time_pts)
        my_deform.shoot()
        my_deform.flow()
        new_vertses = my_deform.template_pts_t[-1] 
        new_source_mesh = Meshes(verts=[new_vertses], faces=[source_faces])
        if j ==0: 
            save_ply(osj(output_dir, 'fit_mesh_init.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            # template_mesh.point_data['normal'] =normals.cpu().numpy()
            # template_mesh.save('./results/Jan12_test1/fit_mesh_init_with_normal.vtp')

        # # We sample 5k points from the surface of each mesh 
        # sample_target = sample_points_from_meshes(label_mesh, 10000)
        # sample_source = sample_points_from_meshes(new_source_mesh, 10000)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        # loss_chamfer, _ = chamfer_distance(sample_target, sample_source)
        # print('debug',new_vertses.shape, target_points.shape)
        # loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0))
        loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0),
                                        single_directional = single_direction) # true
        chamfer_losses.append(loss_chamfer.item())

        loss_edge = mesh_edge_loss(new_source_mesh)#; print('edge:',loss_edge)
        edge_losses.append(loss_edge.item())

        loss_normal = mesh_normal_consistency(new_source_mesh)#; print('normal:',loss_normal)
        normal_losses.append(loss_normal.item())

        loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
        laplacian_losses.append(loss_laplacian.item())

        if not w_strain == 0.:
            # print('debug: ',ComputeArea(new_vertses, source_faces).shape,aera0.shape )
            loss_strain = torch.mean((ComputeArea(new_vertses, source_faces)-aera0)**2)
            strain_losses.append(loss_strain.item())

        if landmark == None: 
            loss = w_chamfer*loss_chamfer + w_edge*loss_edge + w_normal*loss_normal+w_laplacian*loss_laplacian
        else: 
            loss_landmark =torch.mean((new_vertses[landmark_ids] - landmark_points)**2*landmark_weights.unsqueeze(-1))
            if j % save_interval ==50: # 100
                source_landmarks = pv.PolyData(new_vertses[landmark_ids].detach().cpu().numpy())
                target_landmarks = pv.PolyData(landmark_points.detach().cpu().numpy())
                source_landmarks.save(osj(output_dir,'source_landmarks_{:d}.vtp'.format(j)))
                target_landmarks.save(osj(output_dir,'target_landmarks_{:d}.vtp'.format(j)))
            landmark_losses.append(loss_landmark.item())
            loss = w_chamfer*loss_chamfer + w_edge*loss_edge + w_normal*loss_normal+w_laplacian*loss_laplacian+w_landmark*loss_landmark
            if not w_strain == 0.:
                loss += w_strain*loss_strain
        loss.backward()
        optimizer.step()
        total_losses.append(loss.item())
        # tqdm.write(str(loss.item()))

        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, Niter,loss.item()))
            save_ply(osj(output_dir,'fit_mesh_{:d}.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            
            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            er0 = torch.tensor(total_losses)
            er1 = torch.tensor(chamfer_losses)
            er2 = torch.tensor(edge_losses)
            er3 = torch.tensor(normal_losses)
            er4 = torch.tensor(laplacian_losses)

            ax.plot(er0,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
            ax.plot(er1,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='L_chamfer')
            ax.plot(er2,color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='L_edge')
            ax.plot(er3,color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='L_normal')
            ax.plot(er4,color = 'C3',linestyle='solid',linewidth=lw,alpha=1,label='L_lap')

            if landmark != None: 
                er5 = torch.tensor(laplacian_losses)
                ax.plot(er5,color = 'C4',linestyle='solid',linewidth=lw,alpha=1,label='L_landmark')
            
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    deformed_mesh = pv.PolyData(x_normalizer.denormalize(new_source_mesh.verts_packed()).detach().cpu().numpy(),
                                faces = source_mesh.faces)
    animation_mesh = []
    print('saving animations')
    for i, template_pts in enumerate(my_deform.template_pts_t):
        temp_mesh = pv.PolyData(x_normalizer.denormalize(template_pts).detach().cpu().numpy(),
                            faces = source_mesh.faces)
        if landmark != None:
            temp_mesh.point_data['landmark'] = 0.
            temp_mesh.point_data['landmark'][landmark_ids] = 1.
        temp_mesh.save(osj(output_dir,'animation','mesh{:d}.vtp'.format(i)))
        animation_mesh.append(temp_mesh)

        temp_mesh = pv.PolyData(x_normalizer.denormalize(my_deform.control_pts_t[i]).detach().cpu().numpy())
        temp_mesh.point_data['mu'] = my_deform.momenta_t[i].detach().cpu().numpy()
        temp_mesh.save(osj(output_dir,'animation','control_points_{:d}.vtp'.format(i)))
    torch.save(torch.stack(my_deform.template_pts_t),osj(output_dir,'lddmm_control_points.pt'))
    print('saving animations done')
    return deformed_mesh,animation_mesh


def DeformLDDMM_landmark_chamfer(source_mesh, target_points, device=None, x_normalizer=None, output_dir ='./',
                                fps_ratio=0.6, epoch=500, lr= 1e-3, save_interval=100, initial_normal_factor=0.1, number_of_time_pts = 20, T = 1/8, 
                                w_chamfer= 1.,w_edge=1.,w_normal = 0.01,w_laplacian = 0.1,w_landmark = 0.01,w_strain= 0.,  #0.1,#1,#0.01,#0.1,
                                single_direction= True, landmark = None):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    source_points = torch.tensor(source_mesh.points,dtype = torch.float32, device = device)
    target_points = target_points.to(device)
    if x_normalizer == None: 
        x_normalizer = Normalizer_ts(method = 'ms', dim=0)
        _= x_normalizer.fit_normalize(torch.vstack([source_points, target_points]))
    # source_faces = torch.tensor(vtpface2graphface(source_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
    source_faces = torch.tensor(vtpface2graphface(source_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
    source_points_norm = x_normalizer.normalize(source_points)
    target_points_norm = x_normalizer.normalize(target_points)
    initial_verts =source_points_norm
    normals = CalPointNormal(initial_verts,source_faces.t())
    aera0 = ComputeArea(initial_verts, source_faces)
    # from torch_geometric.nn import fps, knn
    import torch_cluster.fps as fps
    torch.manual_seed(0)
    indices = fps(source_points_norm, ratio=fps_ratio) # 0.6
    # indices = torch.arange(len(source_points_norm)) # 0.6
    deform_verts = torch.nn.Parameter(initial_normal_factor*normals[indices])
    optimizer = torch.optim.Adam([deform_verts], lr=lr, weight_decay=1e-4)
    # Number of optimization steps
    Niter = epoch
    w_chamfer =w_chamfer#  1.#0.1
    w_edge = w_edge# 1. #1
    w_normal = w_normal# 0.01#0.01
    w_laplacian = w_laplacian# 0.1#0.1
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    strain_losses = []

    if landmark != None: 
        landmark_losses = []
        # w_landmark = w_landmark
        source_landmark_ids,target_landmark_ids = landmark
        source_landmarks = PCs([source_points[ele] for ele in source_landmark_ids])
        # landmark_points =x_normalizer.normalize(landmark_points.to(device))

    total_losses = []
    # criterion = torch.nn.MSELoss()
    
    for j in range(Niter+1):
        optimizer.zero_grad()
        my_deform = LDDMM(source_points_norm[indices], 
                        # normals[indices]*deform_verts.unsqueeze(1), 
                        deform_verts, 
                        initial_verts,
                        kernel_width = 0.8, #0.3, #0.17,#0.3,
                        T = T , #1/16.,##1./16,#1./256,
                        number_of_time_pts = number_of_time_pts)
        my_deform.shoot()
        my_deform.flow()
        new_vertses = my_deform.template_pts_t[-1] 
        new_source_mesh = Meshes(verts=[new_vertses], faces=[source_faces])
        if j ==0: 
            save_ply(osj(output_dir, 'fit_mesh_init.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            # template_mesh.point_data['normal'] =normals.cpu().numpy()
            # template_mesh.save('./results/Jan12_test1/fit_mesh_init_with_normal.vtp')

        # # We sample 5k points from the surface of each mesh 
        # sample_target = sample_points_from_meshes(label_mesh, 10000)
        # sample_source = sample_points_from_meshes(new_source_mesh, 10000)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        # loss_chamfer, _ = chamfer_distance(sample_target, sample_source)
        # print('debug',new_vertses.shape, target_points.shape)
        # loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0))
        loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0),
                                        single_directional = single_direction) # true
        chamfer_losses.append(loss_chamfer.item())

        loss_edge = mesh_edge_loss(new_source_mesh)#; print('edge:',loss_edge)
        edge_losses.append(loss_edge.item())

        loss_normal = mesh_normal_consistency(new_source_mesh)#; print('normal:',loss_normal)
        normal_losses.append(loss_normal.item())

        loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
        laplacian_losses.append(loss_laplacian.item())

        if not w_strain == 0.:
            # print('debug: ',ComputeArea(new_vertses, source_faces).shape,aera0.shape )
            loss_strain = torch.mean((ComputeArea(new_vertses, source_faces)-aera0)**2)
            strain_losses.append(loss_strain.item())

        if landmark == None: 
            loss = w_chamfer*loss_chamfer + w_edge*loss_edge + w_normal*loss_normal+w_laplacian*loss_laplacian
        else: 
            target_landmarks = PCs([new_vertses[ele] for ele in target_landmark_ids])
            loss_landmark = chamfer_distance(source_landmarks, target_landmarks,
                                            single_directional = single_direction)[0] # true
            # loss_landmark =torch.mean((new_vertses[landmark_ids] - landmark_points)**2*landmark_weights.unsqueeze(-1))
            # if j % save_interval ==50: # 100
            #     source_landmarks = pv.PolyData(new_vertses[landmark_ids].detach().cpu().numpy())
            #     target_landmarks = pv.PolyData(landmark_points.detach().cpu().numpy())
            #     source_landmarks.save(osj(output_dir,'source_landmarks_{:d}.vtp'.format(j)))
            #     target_landmarks.save(osj(output_dir,'target_landmarks_{:d}.vtp'.format(j)))
            landmark_losses.append(loss_landmark.item())
            loss = w_chamfer*loss_chamfer + w_edge*loss_edge + w_normal*loss_normal+w_laplacian*loss_laplacian+w_landmark*loss_landmark
            if not w_strain == 0.:
                loss += w_strain*loss_strain
        loss.backward()
        optimizer.step()
        total_losses.append(loss.item())
        # tqdm.write(str(loss.item()))

        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, Niter,loss.item()))
            save_ply(osj(output_dir,'fit_mesh_{:d}.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            
            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            er0 = torch.tensor(total_losses)
            er1 = torch.tensor(chamfer_losses)
            er2 = torch.tensor(edge_losses)
            er3 = torch.tensor(normal_losses)
            er4 = torch.tensor(laplacian_losses)

            ax.plot(er0,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
            ax.plot(er1,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='L_chamfer')
            ax.plot(er2,color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='L_edge')
            ax.plot(er3,color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='L_normal')
            ax.plot(er4,color = 'C3',linestyle='solid',linewidth=lw,alpha=1,label='L_lap')

            if landmark != None: 
                er5 = torch.tensor(landmark_losses)
                ax.plot(er5,color = 'C4',linestyle='solid',linewidth=lw,alpha=1,label='L_landmark')
            
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    deformed_mesh = pv.PolyData(x_normalizer.denormalize(new_source_mesh.verts_packed()).detach().cpu().numpy(),
                                faces = source_mesh.cells)
    animation_mesh = []
    print('saving animations')
    for i, template_pts in enumerate(my_deform.template_pts_t):
        temp_mesh = pv.PolyData(x_normalizer.denormalize(template_pts).detach().cpu().numpy(),
                            faces = source_mesh.cells)
        if landmark != None:
            temp_mesh.point_data['landmark'] = 0.
            # temp_mesh.point_data['landmark'][landmark_ids] = 1.
        temp_mesh.save(osj(output_dir,'animation','mesh{:d}.vtp'.format(i)))
        animation_mesh.append(temp_mesh)

        temp_mesh = pv.PolyData(x_normalizer.denormalize(my_deform.control_pts_t[i]).detach().cpu().numpy())
        temp_mesh.point_data['mu'] = my_deform.momenta_t[i].detach().cpu().numpy()
        temp_mesh.save(osj(output_dir,'animation','control_points_{:d}.vtp'.format(i)))
    torch.save(torch.stack(my_deform.template_pts_t),osj(output_dir,'lddmm_control_points.pt'))
    print('saving animations done')
    return deformed_mesh,animation_mesh


def DeformLDDMM_landmark_varifold(source_mesh, target_mesh, device=None, x_normalizer=None, output_dir ='./',
                                epoch=500, lr= 1e-3, save_interval=100, initial_normal_factor=0.,
                                number_of_time_pts = 20, T = 1/8, kernel_width = 0.8, 
                                w_varifold= 0., w_vnorm= 0.,w_chamfer= 0.,w_normal= 0., w_edge = 0., 
                                w_laplacian =0., w_strain =0., w_landmark=0.,
                                single_direction = False, landmark = None, mesh_type = 'vtk'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    source_points = torch.tensor(source_mesh.points,dtype = torch.float32, device = device)
    target_points = torch.tensor(target_mesh.points,dtype = torch.float32, device = device)
    if x_normalizer == None: 
        data = torch.vstack([source_points, target_points])
        data_mean = data.mean(0)
        data_std = torch.ones(data.shape[-1],dtype = torch.float32, device = device)*(torch.max(data-data_mean))
        x_normalizer = Normalizer_ts(method = 'ms', dim=0, params = [data_mean, data_std])
        # x_normalizer = Normalizer_ts(method = 'ms', dim=0)
        # _= x_normalizer.fit_normalize(torch.vstack([source_points, target_points]))
    if mesh_type == 'vtk':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
    elif mesh_type == 'vtp':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
    print('my debug1:',source_points.shape,source_faces.shape )
    print('my debug2:',target_points.shape,target_faces.shape )
    source_points_norm = x_normalizer.normalize(source_points)
    target_points_norm = x_normalizer.normalize(target_points)
    initial_cpts = source_points_norm.clone()
    temp = CalPointNormal(source_points_norm, source_faces.T)*initial_normal_factor
    initial_moms = torch.nn.Parameter(temp.to(device))
    # initial_moms = torch.nn.Parameter(torch.zeros(source_points_norm.shape, dtype = torch.float32, device = device))
    optimizer = torch.optim.Adam([initial_moms], lr=lr, weight_decay=1e-4)

    # defind loss functions: 
    # Number of optimization steps
    Niter = epoch
    loss_dict = {}
    weights = []
    if not w_varifold == 0.: # 1. 
        loss_dict['varifold'] = []
        weights.append(w_varifold)
        CalVarifoldloss = MakeVarifoldloss(source_faces, target_points_norm, target_faces, kernel_width)
    if not w_vnorm == 0.: # 0.1
        loss_dict['vnorm'] = []
        weights.append(w_vnorm)
        CalNormloss = MakeNormloss(kernel_width)
    if not w_chamfer == 0.: #1.
        loss_dict['chamfer'] = []
        weights.append(w_chamfer)
    if not w_normal == 0.: # 0.01
        loss_dict['normal'] = [] 
        weights.append(w_normal)
    if not w_edge == 0.: # 1.
        loss_dict['edge'] = []
        weights.append(w_edge)
    if not w_laplacian == 0.: # 0.1
        loss_dict['laplacian'] = []
        weights.append(w_laplacian)
    if not w_strain == 0.: # 0.1
        loss_dict['strain'] = []
        weights.append(w_strain)
        aera0 = ComputeArea(initial_verts, source_faces)
    loss_dict['total'] = []
    # criterion = torch.nn.MSELoss()
    if not w_landmark == 0.:
        landmark_losses = []
        weights.append(w_landmark)
        source_landmark_ids,target_landmark_ids = landmark
        target_landmarks = PCs([target_points_norm[ele] for ele in target_landmark_ids])

    for j in range(Niter+1):
        optimizer.zero_grad()
        my_deform = LDDMM(initial_cpts, 
                        # normals[indices]*initial_moms.unsqueeze(1), 
                        initial_moms, 
                        source_points_norm,
                        kernel_width = kernel_width, #0.3, #0.17,#0.3,0.8
                        T = T , #1/16.,##1./16,#1./256,
                        number_of_time_pts = number_of_time_pts)
        my_deform.shoot()
        my_deform.flow()
        new_vertses = my_deform.template_pts_t[-1]
        new_source_mesh = Meshes(verts=[new_vertses], faces=[source_faces])
        if j ==0: 
            save_ply(osj(output_dir, 'fit_mesh_init.ply'), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            save_ply(osj(output_dir, 'fit_mesh_label.ply'), target_points, target_faces)
        temp_loss_list = []
        if 'varifold' in loss_dict.keys(): 
            loss_varifold = CalVarifoldloss(new_vertses)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_varifold)
            loss_dict['varifold'].append(loss_varifold.item())
        if 'vnorm' in loss_dict.keys(): 
            loss_vnorm= CalNormloss(initial_cpts,initial_moms)
            temp_loss_list.append(loss_vnorm)
            loss_dict['vnorm'].append(loss_vnorm.item())
        if 'chamfer' in loss_dict.keys(): 
            loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0),
                                            single_directional = single_direction) # true
            temp_loss_list.append(loss_chamfer)
            loss_dict['chamfer'].append(loss_chamfer.item())
        if 'normal' in loss_dict.keys(): 
            loss_normal = mesh_normal_consistency(new_source_mesh)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_normal)
            loss_dict['normal'].append(loss_normal.item())
        if 'edge' in loss_dict.keys(): 
            loss_edge = mesh_edge_loss(new_source_mesh)#; print('edge:',loss_edge)
            temp_loss_list.append(loss_edge)
            loss_dict['edge'].append(loss_edge.item())
        if 'laplacian' in loss_dict.keys(): 
            loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
            temp_loss_list.append(loss_laplacian)
            loss_dict['laplacian'].append(loss_laplacian.item())
        if 'w_strain' in loss_dict.keys(): 
            loss_strain = torch.mean((ComputeArea(new_vertses, source_faces)-aera0)**2)
            temp_loss_list.append(loss_strain)
            loss_dict['strain'].append(loss_strain.item())
        if 'w_landmark' in loss_dict.keys(): 
            source_landmarks = PCs([new_vertses[ele] for ele in source_landmark_ids])
            loss_landmark = chamfer_distance(source_landmarks, target_landmarks,
                                            single_directional = single_direction)[0] # true
            temp_loss_list.append(loss_landmark)
            loss_dict['landmark'].append(loss_landmark.item())
        loss = torch.sum(torch.stack([weight*myloss for weight, myloss in zip(weights, temp_loss_list)]))
        loss_dict['total'].append(loss.item())
        loss.backward()
        optimizer.step()
        # print(weights);print(temp_loss_list);print(loss_dict)
        # tqdm.write(str(loss.item()))

        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, Niter,loss.item()))
            save_ply(osj(output_dir,'fit_mesh_{:d}.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            
            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            for i, (key,value) in enumerate(loss_dict.items()):
                if i == len(loss_dict)-1:
                    ax.plot(value,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
                else: 
                    ax.plot(value,color = 'C'+str(i),linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    deformed_mesh = pv.PolyData(x_normalizer.denormalize(new_source_mesh.verts_packed()).detach().cpu().numpy(),
                                faces = source_mesh.faces)
    animation_mesh = []
    print('saving animations')
    for i, template_pts in enumerate(my_deform.template_pts_t):
        temp_mesh = pv.PolyData(x_normalizer.denormalize(template_pts).detach().cpu().numpy(),
                            faces = source_mesh.faces)
        # if landmark != None:
        #     temp_mesh.point_data['landmark'] = 0.
            # temp_mesh.point_data['landmark'][landmark_ids] = 1.
        temp_mesh.save(osj(output_dir,'animation','mesh{:d}.vtp'.format(i)))
        animation_mesh.append(temp_mesh)

        temp_mesh = pv.PolyData(x_normalizer.denormalize(my_deform.control_pts_t[i]).detach().cpu().numpy())
        temp_mesh.point_data['mu'] = my_deform.momenta_t[i].detach().cpu().numpy()
        temp_mesh.save(osj(output_dir,'animation','control_points_{:d}.vtp'.format(i)))
    torch.save(torch.stack(my_deform.template_pts_t),osj(output_dir,'lddmm_control_points.pt'))
    print('saving animations done')
    return deformed_mesh,animation_mesh



def DeformNURBS(cps, initial_m, source_mesh, target_mesh, device=None, x_normalizer=None, output_dir ='./',
                                epoch=500, lr= 1e-3, save_interval=100, initial_normal_factor=0.,
                                number_of_time_pts = 20, T = 1/8, kernel_width = 0.8, 
                                w_varifold= 0., w_vnorm= 0.,w_chamfer= 0.,w_normal= 0., w_edge = 0., 
                                w_laplacian =0., w_strain =0., w_landmark=0.,
                                single_direction = False, landmark = None, mesh_type = 'vtk'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    cps = torch.tensor(cps, dtype = torch.float32, device = device)
    initial_m = torch.tensor(initial_m, dtype = torch.float32, device = device)
    source_points = torch.tensor(source_mesh.points,dtype = torch.float32, device = device)
    target_points = torch.tensor(target_mesh.points,dtype = torch.float32, device = device)
    if x_normalizer == None: 
        data = torch.vstack([source_points, target_points])
        data_mean = data.mean(0)
        data_std = torch.ones(data.shape[-1],dtype = torch.float32, device = device)*(torch.max(data-data_mean))
        x_normalizer = Normalizer_ts(method = 'ms', dim=0, params = [data_mean, data_std])
        # x_normalizer = Normalizer_ts(method = 'ms', dim=0)
        # _= x_normalizer.fit_normalize(torch.vstack([source_points, target_points]))
    if mesh_type == 'vtk':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
    elif mesh_type == 'vtp':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
    print('my debug1:',source_points.shape,source_faces.shape )
    print('my debug2:',target_points.shape,target_faces.shape )
    cps_norm_original = x_normalizer.normalize(cps)
    shape0 = cps_norm_original.size()
    shape = shape0[0]*shape0[1]
    cps_norm = cps_norm_original.view(shape,3)
    cps_norm = torch.tensor(cps_norm.detach().cpu().numpy(), dtype = torch.float32, device=device, requires_grad=True)
    source_points_norm = x_normalizer.normalize(source_points)
    target_points_norm = x_normalizer.normalize(target_points)
    initial_cpts = cps_norm.clone()
    temp = initial_m*initial_normal_factor
    temp = temp.view(shape,3)
    initial_moms = torch.tensor(temp.detach().cpu().numpy(), dtype = torch.float32, device=device, requires_grad=True)
    initial_moms = torch.nn.Parameter(initial_moms.to(device))
    # initial_moms = torch.nn.Parameter(torch.zeros(source_points_norm.shape, dtype = torch.float32, device = device))
    optimizer = torch.optim.Adam([initial_moms], lr=lr, weight_decay=1e-4)

    # defind loss functions: 
    # Number of optimization steps
    Niter = epoch
    loss_dict = {}
    weights = []
    if not w_varifold == 0.: # 1. 
        loss_dict['varifold'] = []
        weights.append(w_varifold)
        CalVarifoldloss = MakeVarifoldloss_NURBS(target_points_norm, target_faces, kernel_width)
    if not w_vnorm == 0.: # 0.1
        loss_dict['vnorm'] = []
        weights.append(w_vnorm)
        CalNormloss = MakeNormloss(kernel_width)
    if not w_chamfer == 0.: #1.
        loss_dict['chamfer'] = []
        weights.append(w_chamfer)
    if not w_normal == 0.: # 0.01
        loss_dict['normal'] = [] 
        weights.append(w_normal)
    if not w_edge == 0.: # 1.
        loss_dict['edge'] = []
        weights.append(w_edge)
    if not w_laplacian == 0.: # 0.1
        loss_dict['laplacian'] = []
        weights.append(w_laplacian)
    if not w_strain == 0.: # 0.1
        loss_dict['strain'] = []
        weights.append(w_strain)
        aera0 = ComputeArea(initial_verts, source_faces)
    loss_dict['total'] = []
    # criterion = torch.nn.MSELoss()
    if not w_landmark == 0.:
        landmark_losses = []
        weights.append(w_landmark)
        source_landmark_ids,target_landmark_ids = landmark
        target_landmarks = PCs([target_points_norm[ele] for ele in target_landmark_ids])

    for j in range(Niter+1):
        optimizer.zero_grad()
        my_deform = LDDMM(initial_cpts, 
                        # normals[indices]*initial_moms.unsqueeze(1), 
                        initial_moms, 
                        cps_norm,
                        kernel_width = kernel_width, #0.3, #0.17,#0.3,0.8
                        T = T , #1/16.,##1./16,#1./256,
                        number_of_time_pts = number_of_time_pts)
        my_deform.shoot()
        my_deform.flow()
        new_cpts = my_deform.template_pts_t[-1]
        new_cpts_original = new_cpts.view(shape0[0],shape0[1],3)
        new_vertses,new_faces = createAortaSurface(new_cpts_original)
        new_source_mesh = Meshes(verts=[new_vertses], faces=[new_faces])
        # new_source_mesh = Meshes(verts=[new_vertses], faces=[source_faces])
        if j ==0: 
            save_ply(osj(output_dir, 'fit_mesh_init.ply'), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            save_ply(osj(output_dir, 'fit_mesh_label.ply'), target_points, target_faces)
        temp_loss_list = []
        if 'varifold' in loss_dict.keys(): 
            loss_varifold = CalVarifoldloss(new_vertses,new_faces)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_varifold)
            loss_dict['varifold'].append(loss_varifold.item())
        if 'vnorm' in loss_dict.keys(): 
            loss_vnorm= CalNormloss(initial_cpts,initial_moms)
            temp_loss_list.append(loss_vnorm)
            loss_dict['vnorm'].append(loss_vnorm.item())
        if 'chamfer' in loss_dict.keys(): 
            loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0),
                                            single_directional = single_direction) # true
            temp_loss_list.append(loss_chamfer)
            loss_dict['chamfer'].append(loss_chamfer.item())
        if 'normal' in loss_dict.keys(): 
            loss_normal = mesh_normal_consistency(new_source_mesh)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_normal)
            loss_dict['normal'].append(loss_normal.item())
        if 'edge' in loss_dict.keys(): 
            loss_edge = mesh_edge_loss(new_source_mesh)#; print('edge:',loss_edge)
            temp_loss_list.append(loss_edge)
            loss_dict['edge'].append(loss_edge.item())
        if 'laplacian' in loss_dict.keys(): 
            loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
            temp_loss_list.append(loss_laplacian)
            loss_dict['laplacian'].append(loss_laplacian.item())
        if 'w_strain' in loss_dict.keys(): 
            loss_strain = torch.mean((ComputeArea(new_vertses, source_faces)-aera0)**2)
            temp_loss_list.append(loss_strain)
            loss_dict['strain'].append(loss_strain.item())
        if 'w_landmark' in loss_dict.keys(): 
            source_landmarks = PCs([new_vertses[ele] for ele in source_landmark_ids])
            loss_landmark = chamfer_distance(source_landmarks, target_landmarks,
                                            single_directional = single_direction)[0] # true
            temp_loss_list.append(loss_landmark)
            loss_dict['landmark'].append(loss_landmark.item())
        loss = torch.sum(torch.stack([weight*myloss for weight, myloss in zip(weights, temp_loss_list)]))
        loss_dict['total'].append(loss.item())
        loss.backward()
        optimizer.step()
        # print(weights);print(temp_loss_list);print(loss_dict)
        # tqdm.write(str(loss.item()))

        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, Niter,loss.item()))
            save_ply(osj(output_dir,'fit_mesh_{:d}.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            
            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            for i, (key,value) in enumerate(loss_dict.items()):
                if i == len(loss_dict)-1:
                    ax.plot(value,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
                else: 
                    ax.plot(value,color = 'C'+str(i),linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    deformed_mesh = pv.PolyData(x_normalizer.denormalize(new_source_mesh.verts_packed()).detach().cpu().numpy(),
                                faces = new_source_mesh.faces_packed().detach().cpu().numpy())
    animation_mesh = []
    print('saving animations')
    for i, template_pts in enumerate(my_deform.template_pts_t):
        # temp_mesh = pv.PolyData(x_normalizer.denormalize(template_pts).detach().cpu().numpy(),
        #                     faces = new_source_mesh.faces_packed().detach().cpu().numpy())
        # if landmark != None:
        #     temp_mesh.point_data['landmark'] = 0.
            # temp_mesh.point_data['landmark'][landmark_ids] = 1.
        temp_mesh = x_normalizer.denormalize(template_pts).view(shape0[0],shape0[1],3).cpu()
        temp_mesh = createAortaSurface2vtp(temp_mesh)
        temp_mesh.save(osj(output_dir,'animation','mesh{:d}.vtp'.format(i)))
        animation_mesh.append(temp_mesh)

        temp_mesh = pv.PolyData(x_normalizer.denormalize(my_deform.control_pts_t[i]).detach().cpu().numpy())
        temp_mesh.point_data['mu'] = my_deform.momenta_t[i].detach().cpu().numpy()
        temp_mesh.save(osj(output_dir,'animation','control_points_{:d}.vtp'.format(i)))
    torch.save(torch.stack(my_deform.template_pts_t),osj(output_dir,'lddmm_control_points.pt'))
    print('saving animations done')
    return deformed_mesh,animation_mesh


def DeformNURBS_length(lengths, tree_v, cl_points, source_mesh, target_mesh, device=None, x_normalizer=None, output_dir ='./',
                                epoch=500, lr= 1e-3, save_interval=100, initial_normal_factor=0.,
                                number_of_time_pts = 20, T = 1/8, kernel_width = 0.8, 
                                w_varifold= 0., w_vnorm= 0.,w_chamfer= 0.,w_normal= 0., w_edge = 0., 
                                w_laplacian =0., w_strain =0., w_landmark=0.,
                                single_direction = False, landmark = None, mesh_type = 'vtk'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    lengths = torch.tensor(lengths, dtype = torch.float32, device = device)
    tree_v = torch.tensor(tree_v, dtype = torch.float32, device = device)
    cl_points_old = torch.tensor(cl_points, dtype = torch.float32, device = device)
    cl_points = torch.zeros(tree_v.shape[0],tree_v.shape[1],tree_v.shape[2], dtype = torch.float32, device = device)
    for i in range(0,tree_v.shape[0]):
        for j in range(0,tree_v.shape[1]):
            cl_points[i,j] = cl_points_old[i].clone()
    source_points = torch.tensor(source_mesh.points,dtype = torch.float32, device = device)
    target_points = torch.tensor(target_mesh.points,dtype = torch.float32, device = device)
    if x_normalizer == None: 
        data = torch.vstack([source_points, target_points])
        data_mean = data.mean(0)
        data_std = torch.ones(data.shape[-1],dtype = torch.float32, device = device)*(torch.max(data-data_mean))
        x_normalizer = Normalizer_ts(method = 'ms', dim=None, params = [data_mean, data_std])
        # x_normalizer = Normalizer_ts(method = 'ms', dim=0)
        # _= x_normalizer.fit_normalize(torch.vstack([source_points, target_points]))
    if mesh_type == 'vtk':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
    elif mesh_type == 'vtp':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
    print('my debug1:',source_points.shape,source_faces.shape )
    print('my debug2:',target_points.shape,target_faces.shape )

    tree_v_norm = tree_v
    cl_points_norm = x_normalizer.normalize(cl_points)
    lengths_norm = lengths/data_std[0]
    cps_norm = cl_points_norm + tree_v_norm*lengths_norm
    shape0 = cps_norm.size()
    shape = shape0[0]*shape0[1]
    cps_norm = cps_norm.view(shape,3)
    cps_norm = torch.tensor(cps_norm.detach().cpu().numpy(), dtype = torch.float32, device=device, requires_grad=True)
    # cps_norm = torch.nn.Parameter(cps_norm.to(device))
    # cps_norm = x_normalizer.normalize(cps)
    source_points_norm = x_normalizer.normalize(source_points)
    target_points_norm = x_normalizer.normalize(target_points)
    initial_cpts = cps_norm.clone()
    # initial_m = torch.rand((shape,1), dtype = torch.float32, device=device)
    initial_m = tree_v_norm*lengths_norm
    # temp = initial_m*initial_normal_factor
    temp = initial_m
    temp = temp.view(shape,3)
    initial_moms = torch.tensor(temp.detach().cpu().numpy(), dtype = torch.float32, device=device, requires_grad=True)
    lengths_norm = torch.tensor(lengths_norm.detach().cpu().numpy(), dtype = torch.float32, device=device, requires_grad=True)
    lengths_norm = torch.nn.Parameter(lengths_norm.to(device))
    # initial_moms = torch.nn.Parameter(torch.zeros(source_points_norm.shape, dtype = torch.float32, device = device))
    optimizer = torch.optim.Adam([lengths_norm], lr=lr, weight_decay=1e-4)
    # print(tree_v_norm,cl_points_norm,cps_norm,initial_moms)

    # defind loss functions: 
    # Number of optimization steps
    Niter = epoch
    loss_dict = {}
    weights = []
    if not w_varifold == 0.: # 1. 
        loss_dict['varifold'] = []
        weights.append(w_varifold)
        CalVarifoldloss = MakeVarifoldloss_NURBS(target_points_norm, target_faces, kernel_width)
    if not w_vnorm == 0.: # 0.1
        loss_dict['vnorm'] = []
        weights.append(w_vnorm)
        CalNormloss = MakeNormloss(kernel_width)
    if not w_chamfer == 0.: #1.
        loss_dict['chamfer'] = []
        weights.append(w_chamfer)
    if not w_normal == 0.: # 0.01
        loss_dict['normal'] = [] 
        weights.append(w_normal)
    if not w_edge == 0.: # 1.
        loss_dict['edge'] = []
        weights.append(w_edge)
    if not w_laplacian == 0.: # 0.1
        loss_dict['laplacian'] = []
        weights.append(w_laplacian)
    if not w_strain == 0.: # 0.1
        loss_dict['strain'] = []
        weights.append(w_strain)
        aera0 = ComputeArea(initial_verts, source_faces)
    loss_dict['total'] = []
    # criterion = torch.nn.MSELoss()
    if not w_landmark == 0.:
        landmark_losses = []
        weights.append(w_landmark)
        source_landmark_ids,target_landmark_ids = landmark
        target_landmarks = PCs([target_points_norm[ele] for ele in target_landmark_ids])

    for j in range(Niter+1):
        optimizer.zero_grad()
        new_cpts = cl_points_norm + tree_v_norm*lengths_norm
        new_cpts = new_cpts.view(shape,3)
        initial_moms = tree_v_norm*lengths_norm
        initial_moms = initial_moms.view(shape,3)
        my_deform = LDDMM(new_cpts, 
                        # normals[indices]*initial_moms.unsqueeze(1), 
                        initial_moms, 
                        cps_norm,
                        kernel_width = kernel_width, #0.3, #0.17,#0.3,0.8
                        T = T , #1/16.,##1./16,#1./256,
                        number_of_time_pts = number_of_time_pts)
        my_deform.shoot()
        my_deform.flow()
        new_cpts = my_deform.template_pts_t[-1]
        new_cpts_original = new_cpts.view(shape0[0],shape0[1],3)
        new_skeleton = new_cpts_original
        # new_skeleton = torch.zeros((shape0[0],shape0[1],3), dtype = torch.float32, device = device)
        # for i in range(shape0[0]):
        #     for jj in range(shape0[1]):
        #         for k in range(0,3):
        #             new_skeleton[i,jj,k] = cl_points_norm[i,jj,k] + tree_v_norm[i,jj,k]*new_cpts_original[i,jj,0]
        new_vertses,new_faces = createAortaSurface(new_skeleton)
        new_source_mesh = Meshes(verts=[new_vertses], faces=[new_faces])
        # new_source_mesh = Meshes(verts=[new_vertses], faces=[source_faces])
        if j ==0: 
            save_ply(osj(output_dir, 'fit_mesh_init.ply'), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            save_ply(osj(output_dir, 'fit_mesh_label.ply'), target_points, target_faces)
        temp_loss_list = []
        if 'varifold' in loss_dict.keys(): 
            loss_varifold = CalVarifoldloss(new_vertses,new_faces)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_varifold)
            loss_dict['varifold'].append(loss_varifold.item())
        if 'vnorm' in loss_dict.keys(): 
            loss_vnorm= CalNormloss(initial_cpts,initial_moms)
            temp_loss_list.append(loss_vnorm)
            loss_dict['vnorm'].append(loss_vnorm.item())
        if 'chamfer' in loss_dict.keys(): 
            loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0),
                                            single_directional = single_direction) # true
            temp_loss_list.append(loss_chamfer)
            loss_dict['chamfer'].append(loss_chamfer.item())
        if 'normal' in loss_dict.keys(): 
            loss_normal = mesh_normal_consistency(new_source_mesh)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_normal)
            loss_dict['normal'].append(loss_normal.item())
        if 'edge' in loss_dict.keys(): 
            loss_edge = mesh_edge_loss(new_source_mesh)#; print('edge:',loss_edge)
            temp_loss_list.append(loss_edge)
            loss_dict['edge'].append(loss_edge.item())
        if 'laplacian' in loss_dict.keys(): 
            loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
            temp_loss_list.append(loss_laplacian)
            loss_dict['laplacian'].append(loss_laplacian.item())
        if 'w_strain' in loss_dict.keys(): 
            loss_strain = torch.mean((ComputeArea(new_vertses, source_faces)-aera0)**2)
            temp_loss_list.append(loss_strain)
            loss_dict['strain'].append(loss_strain.item())
        if 'w_landmark' in loss_dict.keys(): 
            source_landmarks = PCs([new_vertses[ele] for ele in source_landmark_ids])
            loss_landmark = chamfer_distance(source_landmarks, target_landmarks,
                                            single_directional = single_direction)[0] # true
            temp_loss_list.append(loss_landmark)
            loss_dict['landmark'].append(loss_landmark.item())
        loss = torch.sum(torch.stack([weight*myloss for weight, myloss in zip(weights, temp_loss_list)]))
        loss_dict['total'].append(loss.item())
        loss.backward()
        optimizer.step()
        # print(weights);print(temp_loss_list);print(loss_dict)
        # tqdm.write(str(loss.item()))

        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, Niter,loss.item()))
            save_ply(osj(output_dir,'fit_mesh_{:d}.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            
            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            for i, (key,value) in enumerate(loss_dict.items()):
                if i == len(loss_dict)-1:
                    ax.plot(value,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
                else: 
                    ax.plot(value,color = 'C'+str(i),linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    deformed_mesh = pv.PolyData(x_normalizer.denormalize(new_source_mesh.verts_packed()).detach().cpu().numpy(),
                                faces = new_source_mesh.faces_packed().detach().cpu().numpy())
    animation_mesh = []
    print('saving animations')
    for i, template_pts in enumerate(my_deform.template_pts_t):
        temp_mesh = (x_normalizer.denormalize(template_pts)).view(shape0[0],shape0[1],3)
        skeleton = temp_mesh.cpu()
        temp_mesh = createAortaSurface2vtp(skeleton)
        temp_mesh.save(osj(output_dir,'animation','mesh{:d}.vtp'.format(i)))
        animation_mesh.append(temp_mesh)
        skeleton_mesh = pv.PolyData(x_normalizer.denormalize(template_pts).detach().cpu().numpy())
        skeleton_mesh.save(osj(output_dir,'animation','control_points{:d}.vtp'.format(i)))
    np.save(osj(output_dir,'lengths'),(lengths_norm*data_std[0]).detach().cpu().numpy())
        # np.save(osj(output_dir,'animation','length{:d}'.format(i)),(template_pts*data_std[0]).view(shape0[0],shape0[1],1).detach().cpu().numpy())
    #     temp_mesh = pv.PolyData(x_normalizer.denormalize(my_deform.control_pts_t[i]).detach().cpu().numpy())
    #     temp_mesh.point_data['mu'] = my_deform.momenta_t[i].detach().cpu().numpy()
    #     temp_mesh.save(osj(output_dir,'animation','control_points_{:d}.vtp'.format(i)))
    # torch.save(torch.stack(my_deform.template_pts_t),osj(output_dir,'lddmm_control_points.pt'))
    print('saving animations done')
    return deformed_mesh,animation_mesh


def Deform(source_mesh, target_mesh, device=None, x_normalizer=None, output_dir ='./',
            epoch=500, lr= 1e-3, save_interval=100, initial_normal_factor=0.,
            w_varifold= 0., w_vnorm= 0.,w_chamfer= 0.,w_normal= 0., w_edge = 0., 
            w_laplacian =0., w_strain =0., w_landmark=0.,
            single_direction = False, landmark = None, mesh_type = 'vtp'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    source_points = torch.tensor(source_mesh.points,dtype = torch.float32, device = device)
    target_points = torch.tensor(target_mesh.points,dtype = torch.float32, device = device)
    if x_normalizer == None: 
        data = torch.vstack([source_points, target_points])
        data_mean = data.mean(0)
        data_std = torch.ones(data.shape[-1],dtype = torch.float32, device = device)*(torch.max(data-data_mean))
        x_normalizer = Normalizer_ts(method = 'ms', dim=0, params = [data_mean, data_std])
        # x_normalizer = Normalizer_ts(method = 'ms', dim=0)
        # _= x_normalizer.fit_normalize(torch.vstack([source_points, target_points]))
    if mesh_type == 'vtk':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.cells),dtype = torch.long, device = device) # <Nf, 3>
    elif mesh_type == 'vtp':
        source_faces = torch.tensor(vtpface2graphface(source_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
        target_faces = torch.tensor(vtpface2graphface(target_mesh.faces),dtype = torch.long, device = device) # <Nf, 3>
    # print('my debug1:',source_points.shape,source_faces.shape )
    # print('my debug2:',target_points.shape,target_faces.shape )
    source_points_norm = x_normalizer.normalize(source_points)
    target_points_norm = x_normalizer.normalize(target_points)
    initial_cpts = source_points_norm.clone()
    temp = CalPointNormal(source_points_norm, source_faces.T)*initial_normal_factor
    initial_dx = torch.nn.Parameter(temp.to(device))
    # initial_moms = torch.nn.Parameter(torch.zeros(source_points_norm.shape, dtype = torch.float32, device = device))
    optimizer = torch.optim.Adam([initial_dx], lr=lr, weight_decay=1e-4)

    # defind loss functions: 
    # Number of optimization steps
    Niter = epoch
    loss_dict = {}
    weights = []
    if not w_varifold == 0.: # 1. 
        loss_dict['varifold'] = []
        weights.append(w_varifold)
        CalVarifoldloss = MakeVarifoldloss(source_faces, target_points_norm, target_faces, kernel_width)
    if not w_vnorm == 0.: # 0.1
        loss_dict['vnorm'] = []
        weights.append(w_vnorm)
        CalNormloss = MakeNormloss(kernel_width)
    if not w_chamfer == 0.: #1.
        loss_dict['chamfer'] = []
        weights.append(w_chamfer)
    if not w_normal == 0.: # 0.01
        loss_dict['normal'] = [] 
        weights.append(w_normal)
    if not w_edge == 0.: # 1.
        loss_dict['edge'] = []
        weights.append(w_edge)
    if not w_laplacian == 0.: # 0.1
        loss_dict['laplacian'] = []
        weights.append(w_laplacian)
    if not w_strain == 0.: # 0.1
        loss_dict['strain'] = []
        weights.append(w_strain)
        aera0 = ComputeArea(initial_verts, source_faces)
    loss_dict['total'] = []
    # criterion = torch.nn.MSELoss()
    if not w_landmark == 0.:
        landmark_losses = []
        weights.append(w_landmark)
        source_landmark_ids,target_landmark_ids = landmark
        target_landmarks = PCs([target_points_norm[ele] for ele in target_landmark_ids])

    for j in range(Niter+1):
        optimizer.zero_grad()
        new_vertses = source_points_norm + initial_dx
        new_source_mesh = Meshes(verts=[new_vertses], faces=[source_faces])
        if j ==0: 
            save_ply(osj(output_dir, 'fit_mesh_init.ply'), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            save_ply(osj(output_dir, 'fit_mesh_label.ply'), target_points, target_faces)
        temp_loss_list = []
        if 'varifold' in loss_dict.keys(): 
            loss_varifold = CalVarifoldloss(new_vertses)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_varifold)
            loss_dict['varifold'].append(loss_varifold.item())
        if 'vnorm' in loss_dict.keys(): 
            loss_vnorm= CalNormloss(initial_cpts,initial_dx)
            temp_loss_list.append(loss_vnorm)
            loss_dict['vnorm'].append(loss_vnorm.item())
        if 'chamfer' in loss_dict.keys(): 
            loss_chamfer, _ = chamfer_distance(new_vertses.unsqueeze(0), target_points_norm.unsqueeze(0),
                                            single_directional = single_direction) # true
            temp_loss_list.append(loss_chamfer)
            loss_dict['chamfer'].append(loss_chamfer.item())
        if 'normal' in loss_dict.keys(): 
            loss_normal = mesh_normal_consistency(new_source_mesh)#; print('normal:',loss_normal)
            temp_loss_list.append(loss_normal)
            loss_dict['normal'].append(loss_normal.item())
        if 'edge' in loss_dict.keys(): 
            loss_edge = mesh_edge_loss(new_source_mesh)#; print('edge:',loss_edge)
            temp_loss_list.append(loss_edge)
            loss_dict['edge'].append(loss_edge.item())
        if 'laplacian' in loss_dict.keys(): 
            loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
            temp_loss_list.append(loss_laplacian)
            loss_dict['laplacian'].append(loss_laplacian.item())
        if 'w_strain' in loss_dict.keys(): 
            loss_strain = torch.mean((ComputeArea(new_vertses, source_faces)-aera0)**2)
            temp_loss_list.append(loss_strain)
            loss_dict['strain'].append(loss_strain.item())
        if 'w_landmark' in loss_dict.keys(): 
            source_landmarks = PCs([new_vertses[ele] for ele in source_landmark_ids])
            loss_landmark = chamfer_distance(source_landmarks, target_landmarks,
                                            single_directional = single_direction)[0] # true
            temp_loss_list.append(loss_landmark)
            loss_dict['landmark'].append(loss_landmark.item())
        loss = torch.sum(torch.stack([weight*myloss for weight, myloss in zip(weights, temp_loss_list)]))
        loss_dict['total'].append(loss.item())
        loss.backward()
        optimizer.step()
        # print(weights);print(temp_loss_list);print(loss_dict)
        # tqdm.write(str(loss.item()))

        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, Niter,loss.item()))
            save_ply(osj(output_dir,'fit_mesh_{:d}.ply'.format(j)), 
                        x_normalizer.denormalize(new_source_mesh.verts_packed()), new_source_mesh.faces_packed())
            
            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            for i, (key,value) in enumerate(loss_dict.items()):
                if i == len(loss_dict)-1:
                    ax.plot(value,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
                else: 
                    ax.plot(value,color = 'C'+str(i),linestyle='solid',linewidth=lw,alpha=1,label='L_'+key)
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    deformed_mesh = pv.PolyData(x_normalizer.denormalize(new_source_mesh.verts_packed()).detach().cpu().numpy(),
                                faces = source_mesh.faces)
    return deformed_mesh
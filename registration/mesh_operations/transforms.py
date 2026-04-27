import numpy as np
import pyvista as pv
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import join as osj
from os.path import exists as ose
import pyacvd
import open3d as o3d  
import shutil
import mcubes
import sys
import torch
from registration.mesh_operations.normalizer import *


def get_volume_pos(pts1,ct, mode = 'matrix'):
    """
    input:  pts1: np array of pixel values on the ct grid wrt the ct pixel indices 
            ct: vti file for the ct image 
    output: boundary_pts: a sequence of eucleandian coordinates of the shell points 
    """
    inds_1 = np.argwhere(pts1 == 1.0)# ; print(inds_1.shape) # get indeices of "1" pixels 

    if mode == 'forloop':
        # for loop mode 
        volume_pts = []
        for ele in inds_1:
            # print(ele)
            target = [0.0,0.0,0.0]
            ct.TransformIndexToPhysicalPoint(ele,target)
            # ct.TransformIndexToPhysicalPoint(ele[np.array([2,1,0])],target)
            volume_pts.append(np.array(target))
            # volume_pts.append(np.array(target)[np.array([0,1,2])])
        volume_pts = np.asarray(volume_pts)
    elif mode == 'matrix':
        # matrix mode 
        volume_pts = (inds_1*np.array(ct.spacing))+np.array(ct.origin)
    else: 
        sys.exit('error, no such mode')
    return volume_pts


def get_shell_pos(pts1,ct):
    """
    input:  pts1: np array of pixel values on the ct grid wrt the ct pixel indices 
            ct: vti file for the ct image 
    output: boundary_pts: a sequence of eucleandian coordinates of the shell points 
    """
    inds_1 = np.argwhere(pts1 == 1.0)# ; print(inds_1.shape) # get indeices of "1" pixels 
    pts_pad = np.zeros((pts1.shape[0]+2,pts1.shape[1]+2,pts1.shape[2]+2))
    pts_pad[1:-1, 1:-1, 1:-1] = pts1
    def get_neighbor_value(i,j,k, pts = pts_pad):
        i+=1;j+=1;k+=1
        # get neighbour vlaues of a given pixel, calculate product of them 
        return pts_pad[i-1,j,k]*pts_pad[i+1,j,k]*pts_pad[i,j-1,k]*pts_pad[i,j+1,k]*pts_pad[i,j,k-1]*pts_pad[i,j,k+1]


    shell_pts = []
    for ele in inds_1: 
        # print(ele)
        if get_neighbor_value(ele[0],ele[1],ele[2])==0.0:
            target = [0.0,0.0,0.0]
            ct.TransformIndexToPhysicalPoint(ele,target)
            shell_pts.append(target)
    shell_pts = np.asarray(shell_pts)
    return shell_pts


def laplacian(pts1, kernel_size= 3, threshold = 0.5):    
    pad_dim = int(kernel_size-1/2)
    pts_pad = np.zeros((pts1.shape[0]+pad_dim*2,pts1.shape[1]+pad_dim*2,pts1.shape[2]+pad_dim*2))
    pts_pad[pad_dim:-pad_dim, pad_dim:-pad_dim, pad_dim:-pad_dim] = pts1
    import torch
    pts_pad_ts = torch.tensor(pts_pad,dtype = torch.float32).unsqueeze(0).unsqueeze(0)
    Flap = torch.nn.Conv3d(1,1, kernel_size, stride=1, padding=0, 
                    dilation=1, bias=False, 
                    padding_mode='zeros'
    )
    kernel = torch.ones((kernel_size,kernel_size,kernel_size))*(1/kernel_size**3)
    with torch.no_grad():
        Flap.weight.copy_(kernel)
    # print(Flap.weight.data)
    pts_smooth = Flap(pts_pad_ts).detach().numpy()
    return np.squeeze(1*((pts_smooth>=threshold)))


def get_isosurf(npy, iso_value = 0.5):
    # X, Y, Z = np.mgrid[:npy.shape[0], :npy.shape[1], :npy.shape[2]]
    vertices, triangles = mcubes.marching_cubes(npy, iso_value)
    faces = (np.hstack(((np.ones(len(triangles))*3)[...,None],triangles))).astype(int)
    import pyvista as pv 
    mesh = pv.PolyData(vertices, faces = np.ravel(faces))
    return mesh



def ind2phy(id_set,ct):
    # this is slow, becuase of the for loop 
    pts = id_set.copy()
    for i,ele in enumerate(id_set):
        # print(ele)
        target = [0.0,0.0,0.0]
        ct.TransformContinuousIndexToPhysicalPoint(ele,target)
        # ct.TransformIndexToPhysicalPoint(ele[np.array([2,1,0])],target)
        pts[i] = target
    return pts


def ind2phy_fast(id_set,ct):
    # this is slow, becuase of the for loop 
    #TODO add directions , currently defualt direction 100 010 001
    return np.asarray(ct.origin)+ id_set*np.asarray(ct.spacing)

def phy2ind_fast(id_set,ct):
    # this is slow, becuase of the for loop 
    #TODO add directions , currently defualt direction 100 010 001
    return (id_set - np.asarray(ct.origin))* (1/np.asarray(ct.spacing))


def meshind2phy(mesh,ct):
    # this is slow, becuase of the for loop 
    pts = mesh.points
    pts_new = pts.copy()
    for i,ele in enumerate(pts):
        # print(ele)
        target = [0.0,0.0,0.0]
        ct.TransformContinuousIndexToPhysicalPoint(ele,target)
        # ct.TransformIndexToPhysicalPoint(ele[np.array([2,1,0])],target)
        pts_new[i] = target
    mesh.points = pts_new
        # volume_pts.append(np.array(target)[np.array([0,1,2])])
    return mesh




def ptc2surf_poisson_write(ptc, normal_knn = 10, write_mesh= False, ofile = None, temp_dir = './o3d_temp',remesh_params = [3,10000], **poisson_config):
    """
    input:  ptc: point cloud <N,3>
            **poisson_config: configuration of the o3d poisson reconstruction
    output: remesh: reconstructed surface 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptc)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    pcd.orient_normals_consistent_tangent_plane(normal_knn ) # seems like 10 is a sweet spot for normals
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    # poisson ######################################################################
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, **poisson_config)[0] 
    print('poisson reconstruction done!')
    # o3d.visualization.draw_geometries([pcd,poisson_mesh])

    # if ose(temp_dir):
    #     shutil.rmtree(temp_dir)
    # os.mkdir(temp_dir)
    o3d.io.write_triangle_mesh(osj(temp_dir,'temp_shell.ply'),poisson_mesh)
    poisson_mesh_pv = pv.read(osj(temp_dir,'temp_shell.ply'))
    poisson_mesh_pv.save(osj(temp_dir,'temp_shell.vtp'))

    clus = pyacvd.Clustering(poisson_mesh_pv)
    # # mesh is not dense enough for uniform remeshing
    clus.subdivide(remesh_params[0]);clus.cluster(remesh_params[1])
    print('remeshing done!')
    remesh = clus.create_mesh()# remesh
    # remesh.save(osj(temp_dir,'temp_shell_remesh.vtp'))
    if write_mesh == True: 
        remesh.save(ofile)
    return remesh 
    #############################################################################

def ptc2surf_poisson(ptc, temp_dir = './', normal_knn = 10, **poisson_config):
    """
    input:  ptc: point cloud <N,3>
            **poisson_config: configuration of the o3d poisson reconstruction
    output: remesh: reconstructed surface 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptc)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    pcd.orient_normals_consistent_tangent_plane(normal_knn) # seems like 10 is a sweet spot for normals
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    # poisson ######################################################################
    # print(pcd)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, **poisson_config)[0] 
    print('poisson reconstruction done!')
    # o3d.visualization.draw_geometries([pcd,poisson_mesh])
    # help(poisson_mesh)
    pts = np.asarray(poisson_mesh.vertices)
    tris = np.asarray(poisson_mesh.triangles)
    # o3d.io.write_triangle_mesh(osj(temp_dir,'temp_shell.ply'),poisson_mesh)
    # poisson_mesh_pv = pv.read(osj(temp_dir,'temp_shell.ply'))
    faces = np.hstack((np.ones(len(tris))[...,None]*3, tris))
    poisson_mesh_pv = pv.PolyData(pts, faces = np.ravel(faces.astype(int)))
    # os.remove(osj(temp_dir,'temp_shell.ply'))
    return poisson_mesh_pv


    # clus = pyacvd.Clustering(poisson_mesh_pv)
    # # # mesh is not dense enough for uniform remeshing
    # clus.subdivide(remesh_params[0]);clus.cluster(remesh_params[1])
    # print('remeshing done!')
    # remesh = clus.create_mesh()# remesh

    # return remesh 



def ACVD(input_mesh, keep_pts=None,  keep_ratio=1., subdivide = 3):
    N = len(input_mesh.points)

    if keep_pts ==None: 
        keep_pts = int(keep_ratio * N)

    clus = pyacvd.Clustering(input_mesh)
    clus.subdivide(subdivide);clus.cluster(keep_pts)
    remesh = clus.create_mesh()
    return remesh

def RotateAndMove(pts, dxyz, theta, pts_mean = None):
    if pts_mean == None: 
        pts_mean = torch.mean(pts, dim = 0) 
    pts = pts - pts_mean
    x_temp = (torch.cos(theta)*pts[:,0] - torch.sin(theta)*pts[:,1]).unsqueeze(-1)
    y_temp = (torch.sin(theta)*pts[:,0] + torch.cos(theta)*pts[:,1]).unsqueeze(-1)
    z_temp = (pts[:,-1]).unsqueeze(-1)
    pts_new = torch.cat((x_temp,y_temp,z_temp), dim = -1) +pts_mean+dxyz
    return pts_new

def ICP(source_pts,target_pts,x_indices,y_indices,device = None,
        epoch = 100,  lr= 1e-3,save_interval=100,output_dir ='./'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    dxyz = torch.nn.Parameter(torch.zeros(3, device = device))
    theta = torch.nn.Parameter(torch.zeros(1, device = device))
    optimizer = torch.optim.Adam([dxyz,theta], lr=lr, weight_decay=1e-4)
    # normalize the curve 
    x_normalizer = Normalizer_ts(method = 'ms', dim=0)
    source_pts_norm = x_normalizer.fit_normalize(source_pts)
    target_pts_norm = x_normalizer.normalize(target_pts)
    total_losses = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    for j in range(epoch):
        optimizer.zero_grad()
        prediction_pts_norm = RotateAndMove(source_pts_norm[x_indices].to(device), dxyz, theta)
        loss = torch.sum((prediction_pts_norm-target_pts_norm[y_indices].to(device))**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_losses.append(loss.item())
        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, epoch,loss.item()))
            # temp_mesh = pv.PolyData(rotated_points.detach().cpu().numpy())
            # temp_mesh = pv.PolyData(x_normalizer.denormalize(CPs_norm).detach().cpu().numpy())
            temp_pts = x_normalizer.denormalize(RotateAndMove(source_pts_norm.to(device), dxyz,theta,
                                                              torch.mean(source_pts_norm[x_indices].to(device), dim = 0) ))
            temp_mesh = pv.PolyData(temp_pts.detach().cpu().numpy())
            temp_mesh.point_data['register'] = 0.
            temp_mesh.point_data['register'][x_indices] = 1.
            temp_mesh.save(osj(output_dir, 'fit_mesh_{:d}.vtp'.format(j)))
            temp_mesh = pv.PolyData(temp_pts[x_indices].detach().cpu().numpy())
            temp_mesh.save(osj(output_dir, 'fit_mesh_register_{:d}.vtp'.format(j)))

            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            er0 = torch.tensor(total_losses)
            ax.plot(er0,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    return x_normalizer.denormalize(RotateAndMove(source_pts_norm.to(device), dxyz, theta)).detach(), dxyz.cpu(), theta.cpu(), torch.mean(source_pts_norm[x_indices], dim = 0), x_normalizer

def ICP_no_z(source_pts,target_pts,x_indices,y_indices,device = None,
        epoch = 100,  lr= 1e-3,save_interval=100,output_dir ='./'):
    if output_dir != None: 
        if ose(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        os.mkdir(output_dir+'/animation')
    temp_mesh = pv.PolyData(target_pts.detach().cpu().numpy())
    temp_mesh.point_data['register'] = 0.
    temp_mesh.point_data['register'][y_indices] = 1.
    temp_mesh.save(osj(output_dir, 'target_mesh.vtp'))
    temp_mesh = pv.PolyData(target_pts[y_indices].detach().cpu().numpy())
    temp_mesh.save(osj(output_dir, 'target_mesh_register.vtp'))

    dxy = torch.nn.Parameter(torch.zeros(2, device = device))
    theta = torch.nn.Parameter(torch.zeros(1, device = device))
    optimizer = torch.optim.Adam([dxy,theta], lr=lr, weight_decay=1e-4)
    # normalize the curve 
    x_normalizer = Normalizer_ts(method = 'ms', dim=0)
    source_pts_norm = x_normalizer.fit_normalize(source_pts)
    target_pts_norm = x_normalizer.normalize(target_pts)
    total_losses = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    for j in range(epoch):
        optimizer.zero_grad()
        dxyz = torch.cat((dxy, torch.tensor([0.], device = device)))
        # print('debug', dxyz)
        prediction_pts_norm = RotateAndMove(source_pts_norm[x_indices].to(device), dxyz, theta)
        loss = torch.sum((prediction_pts_norm-target_pts_norm[y_indices].to(device))**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_losses.append(loss.item())
        if j % save_interval ==0: # 100
            print('epoch {:d} / {:d} with loss: {:.8f}'.format(j, epoch,loss.item()))
            # temp_mesh = pv.PolyData(rotated_points.detach().cpu().numpy())
            # temp_mesh = pv.PolyData(x_normalizer.denormalize(CPs_norm).detach().cpu().numpy())
            temp_pts = x_normalizer.denormalize(RotateAndMove(source_pts_norm.to(device), dxyz,theta,
                                                              torch.mean(source_pts_norm[x_indices].to(device), dim = 0) ))
            temp_mesh = pv.PolyData(temp_pts.detach().cpu().numpy())
            temp_mesh.point_data['register'] = 0.
            temp_mesh.point_data['register'][x_indices] = 1.
            temp_mesh.save(osj(output_dir, 'fit_mesh_{:d}.vtp'.format(j)))
            temp_mesh = pv.PolyData(temp_pts[x_indices].detach().cpu().numpy())
            temp_mesh.save(osj(output_dir, 'fit_mesh_register_{:d}.vtp'.format(j)))

            # print('plotting')
            fig, ax = plt.subplots(figsize=(10,8))
            lw = 3
            er0 = torch.tensor(total_losses)
            ax.plot(er0,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
            ax.set_yscale('log')
            ax.legend()
            fig.savefig(osj(output_dir,'fit_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
            plt.close()
    
    registered_points = x_normalizer.denormalize(RotateAndMove(source_pts_norm.to(device), dxyz, theta)).detach()
    dxyz = dxyz.cpu()
    theta = theta.cpu()
    rxyz= torch.mean(source_pts_norm[x_indices], dim = 0)
    # save all the data
    torch.save(
        {'registered_points':registered_points,
        'dxyz': dxyz,
        'theta':theta,
        'rxyz':rxyz,
        'normalizer_param1':x_normalizer.get_params()[0],
        'normalizer_param2':x_normalizer.get_params()[1]
        },
        osj(output_dir,'algin_params.pt')
    )
    return registered_points,dxyz, theta, rxyz, x_normalizer

def TranformPC(source_pts,target_pts,x_indices,y_indices, dxyz, theta,device = None):
    x_normalizer = Normalizer_ts(method = 'ms', dim=0)
    source_pts_norm = x_normalizer.fit_normalize(source_pts)
    target_pts_norm = x_normalizer.normalize(target_pts)
    prediction_pts_norm = RotateAndMove(source_pts_norm[x_indices].to(device), dxyz, theta)
    loss = torch.sum((prediction_pts_norm-target_pts_norm[y_indices].to(device))**2)
    temp1 = pv.PolyData(prediction_pts_norm.detach().cpu().numpy())
    temp2 = pv.PolyData(target_pts_norm[y_indices].detach().numpy())
    temp1.save('./results/temp1_norm.vtp')
    temp2.save('./results/temp2_norm.vtp')
    prediction_pts_denorm = x_normalizer.denormalize(prediction_pts_norm)
    temp1 = pv.PolyData(prediction_pts_denorm.detach().cpu().numpy())
    temp2 = pv.PolyData(target_pts[y_indices].detach().numpy())
    temp1.save('./results/temp1.vtp')
    temp2.save('./results/temp2.vtp')
    print(loss.item())

def AlignMesh(source_pts, dxyz,theta,rxyz, x_normalizer):
    source_pts_norm = x_normalizer.normalize(source_pts)
    aligned_source_pts_norm =  RotateAndMove(source_pts_norm, dxyz, theta, rxyz)
    aligned_source_pts_denorm = x_normalizer.denormalize(aligned_source_pts_norm)
    return aligned_source_pts_denorm



import vtk

def read_vti(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def resample_image(image_data, scale_factor):
    resample = vtk.vtkImageResample()
    resample.SetInputData(image_data)
    resample.SetAxisMagnificationFactor(0, scale_factor)
    resample.SetAxisMagnificationFactor(1, scale_factor)
    resample.SetAxisMagnificationFactor(2, scale_factor)
    resample.Update()
    return resample.GetOutput()

def threshold_image(image_data):
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(image_data)
    threshold.ThresholdByUpper(0.5)  # Thresholding around 0.5 to get binary values
    threshold.SetInValue(1)          # Set values above threshold to 1
    threshold.SetOutValue(0)         # Set values below threshold to 0
    threshold.Update()
    return threshold.GetOutput()

def write_vti(image_data, filename):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
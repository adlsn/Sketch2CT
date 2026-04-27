
import numpy as np
import torch 
import pyvista as pv
from torch_geometric.nn import fps, knn


def GetTriIds(traj_pts, mesh, N_neighbor = 6):
    face_normals_ts = torch.tensor(mesh.cell_normals, dtype = torch.float32)
    ids = knn(torch.tensor(mesh.cell_centers().points, dtype = torch.float32),traj_pts, N_neighbor)
    possible_cell_ids = ids[1] #possible cell ->torch.Size([12000]) <N_traj_pts*N_neighbors>
    print('debug possible cell ids:',possible_cell_ids.shape )
    face_inds = mesh.faces.reshape(-1,4)[:,1:]
    relevant_points_ts = torch.tensor(mesh.points[face_inds[possible_cell_ids]], dtype = torch.float32) #<3*12000,3,3>
    print('debug relevant points ts :',relevant_points_ts.shape ) #<1200*10,3,3>

    ele1 = torch.cross(relevant_points_ts[:,0,:]-traj_pts[ids[0]], relevant_points_ts[:,1,:] - relevant_points_ts[:,0,:], dim =-1)
    ele2 = torch.cross(relevant_points_ts[:,1,:]-traj_pts[ids[0]] , relevant_points_ts[:,2,:] - relevant_points_ts[:,1,:], dim =-1)
    ele3 = torch.cross(relevant_points_ts[:,2,:]-traj_pts[ids[0]] , relevant_points_ts[:,0,:] - relevant_points_ts[:,2,:], dim =-1)
    whichside1 = (torch.sum(face_normals_ts[possible_cell_ids]*ele1,dim=-1))>0
    whichside2 = (torch.sum(face_normals_ts[possible_cell_ids]*ele2,dim=-1))>0
    whichside3 = (torch.sum(face_normals_ts[possible_cell_ids]*ele3,dim=-1))>0
    whichside = (whichside1*whichside2*whichside3).reshape(-1,N_neighbor)
    lost_ids = torch.nonzero(whichside.sum(dim = 1)==False)
    if not len(lost_ids) == 0: 
        print('some points missed')
        for myid in lost_ids: 
            print('left',whichside[myid-1])
            print('right',whichside[myid+1])
            whichside[myid] = whichside[myid-1]
    print('debug, see zero indices', whichside.numpy().shape, torch.nonzero(whichside).shape)
    print('checking consistency:', whichside.shape,torch.nonzero(whichside).shape,abs(torch.nonzero(whichside)[:,0]-torch.arange(len(whichside))).max())
    # if the first two are same shape and last one is zero. we are good!
    # print('debug, see zero indices neighbor', whichside.numpy()[])
    which_face = possible_cell_ids.reshape(-1,N_neighbor)[np.nonzero(whichside.numpy())]
    return which_face 

def BatchProjectPoint2Surface(pt1, mesh,which_face):
    '''
    input: 
        pt1, <b,3>
        mesh,
        normal, <b,3>
    '''
    face_normals_ts = torch.tensor(mesh.cell_normals, dtype = torch.float32)
    normals = face_normals_ts[which_face]
    face_inds = mesh.faces.reshape(-1,4)[:,1:]
    pts = torch.tensor(mesh.points[face_inds[which_face,0]], dtype=torch.float32)
    pt1s = pts-pt1
    return pt1+torch.sum(pt1s*normals,dim=-1,keepdim = True)*normals
# calculate barycentric coordinates 
def BatchXyz2Barycentric(pt1, tri_pts):
    '''
    input: 
        pt1, <b,3>
        tri_pts, <b,3,3>
    '''
    s1 = torch.norm(torch.cross(tri_pts[:,1,:]-pt1, tri_pts[:,2,:] - tri_pts[:,1,:], dim =-1), dim=-1)/2
    s2 = torch.norm(torch.cross(tri_pts[:,2,:]-pt1, tri_pts[:,0,:] - tri_pts[:,2,:], dim =-1), dim=-1)/2
    s3 = torch.norm(torch.cross(tri_pts[:,0,:]-pt1, tri_pts[:,1,:] - tri_pts[:,0,:], dim =-1), dim=-1)/2
    s123= s1+s2+s3
    return torch.cat(((s1/s123).unsqueeze(-1), (s2/s123).unsqueeze(-1), (s3/s123).unsqueeze(-1)),dim = -1)

def BatchBarycentric2Xyz(bary_coords, tri_pts):
    '''
    input: 
        bary_coords, <b,3>
        tri_pts, <b,3,3>
    '''
    return torch.sum(bary_coords.unsqueeze(-1)*tri_pts, dim=1)

def Traj2Bary(traj,mesh,tri_ids):
    face_inds = mesh.faces.reshape(-1,4)[:,1:]
    tri_pts = torch.tensor(mesh.points[face_inds[tri_ids]], dtype = torch.float32)
    bary_coords = BatchXyz2Barycentric(traj, tri_pts)
    return (tri_ids, bary_coords)

def Bary2Traj(bary_info, mesh):
    tri_ids, bary_coords = bary_info
    face_inds = mesh.faces.reshape(-1,4)[:,1:]
    tri_pts = torch.tensor(mesh.points[face_inds[tri_ids]], dtype = torch.float32)
    return BatchBarycentric2Xyz(bary_coords, tri_pts)

def ComputeArea(pos, tri_ids):
    tri_pts = pos[tri_ids]
    area_vectors = 0.5*torch.cross(tri_pts[:,2] - tri_pts[:,1], tri_pts[:,1] - tri_pts[:,0], dim = 1)
    return torch.norm(area_vectors, dim = -1)
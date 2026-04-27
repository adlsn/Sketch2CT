import vtk 
import numpy as np
import pyvista as pv
import matplotlib
import matplotlib.pyplot as plt
from os.path import join as osj
from os.path import exists as ose
import torch_geometric
import torch
from torch_geometric.data import Data

def cellArrayDivider(input_array):
    # divied vtk cell array into individual cells 
    N = len(input_array)
    cursor = 0
    head_id = []
    segs = []
    
    while(cursor<N):
        head_id.append(cursor)
        segs.append(input_array[cursor+1:cursor+input_array[cursor]+1])
        cursor = cursor+input_array[cursor]+1
        # print(cursor)
    return segs

def readvtp(file_name):
    # read in vtp based on filename 
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()

    return reader.GetOutput()


def vtk2GraphVertex(input_file, data = [], data_label = [], device = 'cpu'):
    """
    input: mesh file 
    output: graph 
    process: compute vertex normals, create edge_index, add pressure and wss data, add stmdist 
    """
    
    if type(input_file) == str:
        mesh = pv.read(input_file)
    else:
        mesh = input_file
    segs = np.array(cellArrayDivider(mesh.faces))
    points = np.array(mesh.points,dtype = np.float32)
    nodal_normals = np.array(mesh.point_normals,dtype = np.float32)
    nodal_features = np.array(nodal_normals, dtype = np.float32)
    transform = torch_geometric.transforms.FaceToEdge(remove_faces=False) ### undirected graph, meaning eij exists for every eji
    mesh_graph = Data(x = torch.tensor(nodal_features, device = device),#edge_index = torch.tensor(edge_connectivity.T),
                pos = torch.tensor(points, device = device), norm = torch.tensor(nodal_normals, device = device),
                face = torch.tensor(np.array(segs.T), device = device))
    if len(data_label) > 0:
        for i in range(len(data_label)):
            temp_data = np.array(mesh.point_data[data_label[i]],dtype = np.float32) 
            if len(temp_data.shape) == 1:
                temp_data = temp_data[...,None]
            mesh_graph[data_label[i]] = torch.tensor(temp_data, device = device)
    mesh_graph_transformed = transform(mesh_graph)
    return mesh_graph_transformed




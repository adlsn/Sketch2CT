import pyacvd
def remesh(mesh, N, subdivide = 3): 
    clus = pyacvd.Clustering(mesh)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(subdivide)
    clus.cluster(N)
    # remesh
    remesh = clus.create_mesh()
    return remesh 


import vtk 
from torch_geometric.nn import knn
import torch
import pyvista as pv
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

def extract_landmarks(mesh,landmark = [6,1,3,2,4,5]):
    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, feature_angle=30.0)
    stripper = vtk.vtkStripper()
    stripper.SetInputData(edges)
    stripper.Update()
    # Convert back to PyVista PolyData
    stripped_edges = pv.wrap(stripper.GetOutput())
    stripped_edges_list = cellArrayDivider(stripped_edges.lines)

    mesh.point_data['landmark']  = 0
    for i in range(len(stripped_edges_list)):
        xx = torch.tensor(stripped_edges.points[stripped_edges_list[i]])
        yy = torch.tensor(mesh.points)
        assign_index = knn(yy, xx, 1)
        mesh.point_data['landmark'][assign_index[1]] = '{:d}'.format(landmark[i])
    return mesh 

def make_tube(centerline_points, tube_radius, N = 100):
    centerline = pv.Spline(centerline_points, N)
    tube_mesh = centerline.tube(radius=tube_radius)
    return tube_mesh






import numpy as np
import pyvista as pyvista

# filter the points from the scan up to a circle region 
def filter_scan(point_cloud, center, radius):
    return point_cloud[np.linalg.norm((point_cloud - center)[:,:2], axis = -1)<radius]

def vtpface2graphface(vtpface):
    return vtpface.reshape(-1,4)[:,1:]

def graphface2vtpface(graphface):
    return np.hstack((np.repeat(3, len(graphface))[...,None],graphface))


def calculate_circle(pts):
    A,B,C = pts[0],pts[1],pts[2]
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    V = np.cross(B-A,B-C)
    V_norm = V/np.linalg.norm(V)
    return P,R, V_norm
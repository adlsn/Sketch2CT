import sys
from src.basics import * 
class BSplineCurve1D():
    def __init__(self,knots,CPs,
                 multiplicities = None,
                 degree         = 2,
                 res            = 100):
        """Knot vector class."""
        self.knots         = knots
        self.device        = knots.device
        self.degree        = degree
        self.knot_vector   = KnotVector(knots,multiplicities = multiplicities,degree=degree)
        self.CPs = CPs
        self.res = res
        self.dim = self.CPs.shape[-1]
        assert len(CPs) == self.knot_vector.n + 1, "number of points and knot vector length doesn't match"
        self.CurvePs = self.eval_curve()

    def eval_point(self, u):
        span = self.knot_vector.find_span_batch(u) # <1>
        Ns = self.knot_vector.eval_u_batch(u)[-1]
        return torch.sum([w*P for w,P in zip(Ns, self.CPs[span:span+self.degree+1])])

    def eval_curve(self, u=None):
        if u == None: 
            u = torch.linspace(self.knots[0],self.knots[-1],self.res)   
        b_size = len(u)
        Ns =torch.stack(self.knot_vector.eval_u_batch(u)[-1]).T #<b_size,degree+1>
        spans = self.knot_vector.find_span_batch(u) # <N_u>
        I = spans.unsqueeze(-1).repeat_interleave(self.degree+1, dim =-1)-self.degree+torch.arange(self.degree+1)
        print(self.CPs[I].shape,Ns.unsqueeze(-1).shape )
        CurvePs = (self.CPs[I] * Ns.unsqueeze(-1)).sum(1)
        return CurvePs
    def plot(self,):
        CurvePs_knots = self.eval_curve(self.knots)
        self.ax = plot_curve(self.CPs, self.CurvePs, self.knots, CurvePs_knots)



class PeriodicBSplineCurve1D():
    def __init__(self,Nu,CPs,
                 multiplicities = None,
                 degree         = 2,
                 res            = 100):
        """Knot vector class."""
        self.du = 1.
        device = CPs.device
        self.extended_knots = torch.arange(0.-degree*self.du,Nu+1+degree*self.du,self.du,device = device)
        # self.original_knots = torch.arange(0.,Nu+1,self.du,device = device)
        self.knots = self.extended_knots[degree:-degree]
        self.device        = device
        self.degree        = degree
        self.knot_vector   = KnotVector(self.extended_knots,
                                        multiplicities = multiplicities,
                                        degree=degree,
                                        open = True
                                        )
        self.CPs = CPs
        self.extended_CPs = torch.cat((self.CPs[-self.degree:],self.CPs), dim = 0)
        self.res = res
        self.dim = self.CPs.shape[-1]
        assert len(self.extended_CPs) == self.knot_vector.n + 1, "number of points and knot vector length doesn't match"
        # self.extended_CPs_pad = torch.cat((self.extended_CPs,self.extended_CPs[-1:]), dim = 0)
        self.CurvePs = self.eval_curve()

    def eval_point(self, u):
        span = self.knot_vector.find_span_batch(u) # <1>
        Ns = self.knot_vector.eval_u_batch(u)[-1]
        return torch.sum([w*P for w,P in zip(Ns, self.extended_CPs[span:span+self.degree+1])])

    def eval_curve(self, u=None):
        if u == None: 
            u = torch.linspace(self.knots[0],self.knots[-1],self.res)   
        b_size = len(u)
        Ns =torch.stack(self.knot_vector.eval_u_batch(u)[-1]).T #<b_size,degree+1>
        spans = self.knot_vector.find_span_batch(u) # <N_u>
        I = spans.unsqueeze(-1).repeat_interleave(self.degree+1, dim =-1)-self.degree+torch.arange(self.degree+1)
        # print(I)# print(self.extended_CPs.shape)
        print(self.extended_CPs[I].shape,Ns.unsqueeze(-1).shape )
        CurvePs = (self.extended_CPs[I] * Ns.unsqueeze(-1)).sum(1)
        # CurvePs = (self.extended_CPs_pad[I] * Ns.unsqueeze(-1)).sum(1)
        return CurvePs
    
    def plot(self,):
        CurvePs_knots = self.eval_curve(self.knots)
        self.ax = plot_curve(torch.cat((self.CPs, self.CPs[0].unsqueeze(0)),dim = 0), self.CurvePs, self.knots, CurvePs_knots)


class BSplineSurface2D():
    def __init__(self,u_knots, v_knots, CPs,
                 u_multiplicities = None,v_multiplicities = None,
                 u_degree         = 2,v_degree         = 2,
                 u_res            = 100,v_res          = 100):
        """Knot vector class."""
        self.u_knots         = u_knots
        self.v_knots         = v_knots
        self.device          = u_knots.device
        self.u_degree        = u_degree
        self.v_degree        = v_degree
        self.u_knot_vector   = KnotVector(u_knots,multiplicities = u_multiplicities,degree=u_degree)
        self.v_knot_vector   = KnotVector(v_knots,multiplicities = v_multiplicities,degree=v_degree)
        self.CPs = CPs
        self.u_res = u_res;self.v_res = v_res
        self.dim = self.CPs.shape[-1]
        assert CPs.shape[0] == self.u_knot_vector.n + 1, "number of points and knot vector length in u direction doesn't match"
        assert CPs.shape[1] == self.v_knot_vector.n + 1, "number of points and knot vector length in v direction doesn't match"
        self.SurfacePs = self.eval_surface()

    def eval_point(self, uv):
        u,v = uv
        u_span = self.u_knot_vector.find_span_batch(u) # <1>
        u_Ns = torch.cat(self.u_knot_vector.eval_u_batch(u)[-1])
        v_span = self.v_knot_vector.find_span_batch(v) # <1>
        v_Ns = torch.cat(self.v_knot_vector.eval_u_batch(v)[-1])

        Ns_matrix = (u_Ns.unsqueeze(1) @ v_Ns.unsqueeze(0))
        CPs_seg = self.CPs[u_span:u_span+self.u_degree+1,v_span:v_span+self.v_degree+1]
        return (CPs_seg* Ns_matrix.unsqueeze(-1)).sum((0,1))

    def eval_surface(self, u = None, v = None):
        if u == None: 
            u = torch.linspace(self.u_knots[0],self.u_knots[-1],self.u_res)   
        if v == None: 
            v = torch.linspace(self.v_knots[0],self.v_knots[-1],self.v_res)   
        u_size = len(u)
        u_Ns =torch.stack(self.u_knot_vector.eval_u_batch(u)[-1]).T #<u_size,ud>
        u_spans = self.u_knot_vector.find_span_batch(u) # <N_u>
        v_size = len(v)
        v_Ns =torch.stack(self.v_knot_vector.eval_u_batch(v)[-1]).T #<v_size,vd>
        v_spans = self.v_knot_vector.find_span_batch(v) # <N_u>

        u_I = u_spans.unsqueeze(-1).repeat_interleave(self.u_degree+1, dim =-1)-self.u_degree+torch.arange(self.u_degree+1) # <u_size, degree+1>
        u_I = u_I.unsqueeze(-1)#.repeat_interleave(self.v_degree+1, dim =-1)
        v_I = v_spans.unsqueeze(-1).repeat_interleave(self.v_degree+1, dim =-1)-self.v_degree+torch.arange(self.v_degree+1)# <v_size, degree+1>
        v_I = v_I.unsqueeze(-2)#.repeat_interleave(self.u_degree+1, dim =-2)
        # print(u_spans, v_spans)
        CPs_seg = self.CPs[u_I.unsqueeze(1), v_I.unsqueeze(0)] #<ub,vb,ud,vd,dim>

        Ns_matrix = (u_Ns.unsqueeze(1).unsqueeze(-1) @ v_Ns.unsqueeze(0).unsqueeze(-2)) #<ub,vb,ud,vd>
        SurfacePs = (CPs_seg* Ns_matrix.unsqueeze(-1)).sum((2,3))
        return SurfacePs


class AortaBSplineSurface2D():
    def __init__(self, Nu, Nv, CPs,
                 u_multiplicities = None,v_multiplicities = None,
                 u_degree         = 3,v_degree         = 3,
                 u_res            = 100,v_res          = 100):
        """Knot vector class."""
        self.du = 1. 
        self.dv = 1.
        device = CPs.device
        self.u_degree        = u_degree
        self.v_degree        = v_degree
        self.u_knots = torch.arange(0.,Nu+1,self.du,device = device)
        self.extended_v_knots = torch.arange(0.-v_degree*self.dv,Nv+1+v_degree*self.dv,self.dv,device = device)
        self.v_knots = self.extended_v_knots[v_degree:-v_degree]
        self.u_knot_vector   = KnotVector(self.u_knots,multiplicities = u_multiplicities,degree=u_degree)
        self.v_knot_vector   = KnotVector(self.extended_v_knots,
                                        multiplicities = v_multiplicities ,
                                        degree=v_degree,
                                        open = True
                                        )
        self.CPs = CPs
        self.extended_CPs = torch.cat((self.CPs[:,-self.v_degree:],self.CPs), dim = 1)
        self.u_res = u_res;self.v_res = v_res
        self.dim = self.CPs.shape[-1]
        assert self.extended_CPs.shape[0] == self.u_knot_vector.n + 1, "number of points and knot vector length in u direction doesn't match"
        assert self.extended_CPs.shape[1] == self.v_knot_vector.n + 1, "number of points and knot vector length in v direction doesn't match"
        self.SurfacePs = self.eval_surface(device=device)

    def eval_point(self, uv):
        u,v = uv
        u_span = self.u_knot_vector.find_span_batch(u) # <1>
        u_Ns = torch.cat(self.u_knot_vector.eval_u_batch(u)[-1])
        v_span = self.v_knot_vector.find_span_batch(v) # <1>
        v_Ns = torch.cat(self.v_knot_vector.eval_u_batch(v)[-1])

        Ns_matrix = (u_Ns.unsqueeze(1) @ v_Ns.unsqueeze(0))
        CPs_seg = self.CPs[u_span:u_span+self.u_degree+1,v_span:v_span+self.v_degree+1]
        return (CPs_seg* Ns_matrix.unsqueeze(-1)).sum((0,1))

    def eval_surface(self, u = None, v = None, device = None):
        if u == None: 
            u = torch.linspace(self.u_knots[0],self.u_knots[-1],self.u_res,device=device,requires_grad=True)   
        if v == None: 
            v = torch.linspace(self.v_knots[0],self.v_knots[-1],self.v_res,device=device,requires_grad=True)   
        u_size = len(u)
        u_Ns =torch.stack(self.u_knot_vector.eval_u_batch(u)[-1]).T #<u_size,ud>
        u_spans = self.u_knot_vector.find_span_batch(u) # <N_u>
        v_size = len(v)
        v_Ns =torch.stack(self.v_knot_vector.eval_u_batch(v)[-1]).T #<v_size,vd>
        v_spans = self.v_knot_vector.find_span_batch(v) # <N_u>

        u_I = u_spans.unsqueeze(-1).repeat_interleave(self.u_degree+1, dim =-1)-self.u_degree+torch.arange(self.u_degree+1,device=device) # <u_size, degree+1>
        u_I = u_I.unsqueeze(-1)#.repeat_interleave(self.v_degree+1, dim =-1)
        v_I = v_spans.unsqueeze(-1).repeat_interleave(self.v_degree+1, dim =-1)-self.v_degree+torch.arange(self.v_degree+1,device=device)# <v_size, degree+1>
        v_I = v_I.unsqueeze(-2)#.repeat_interleave(self.u_degree+1, dim =-2)
        # print(u_spans, v_spans)
        CPs_seg = self.extended_CPs[u_I.unsqueeze(1), v_I.unsqueeze(0)] #<ub,vb,ud,vd,dim>

        Ns_matrix = (u_Ns.unsqueeze(1).unsqueeze(-1) @ v_Ns.unsqueeze(0).unsqueeze(-2)) #<ub,vb,ud,vd>
        SurfacePs = (CPs_seg* Ns_matrix.unsqueeze(-1)).sum((2,3))
        return SurfacePs
    def plot(self,):
        plot_surface(self.CPs, self.SurfacePs)
    def export2vtk(self,output_path=None, triangulate= False):
        import pyvista as pv
        import numpy as np 
        # create the surface 
        # surface_mesh = pv.StructuredGrid(self.SurfacePs.numpy()[:,:,0], self.SurfacePs.numpy()[:,:,1], self.SurfacePs.numpy()[:,:,2])
        d1,d2,d3= self.SurfacePs.shape
        I = np.arange(d1*(d2-1)).reshape(d1,(d2-1))
        I = np.concatenate((I ,I[:,0:1]),axis = 1)
        f = np.stack((np.ones((d1-1)*(d2-1), dtype = np.int64)*4,I[:-1,:-1].ravel(),I[1:,:-1].ravel(),I[1:,1:].ravel(),I[:-1,1:].ravel())).T.ravel()
        surface_mesh = pv.PolyData(self.SurfacePs[:,:-1].reshape(-1,d3).detach().numpy(), f)
        if triangulate == True: 
            surface_mesh = surface_mesh.triangulate()
        if not output_path == None: 
            surface_mesh.save(output_path)
        return surface_mesh
    def export2mesh(self,):
        SurfacePs = self.SurfacePs
        device = SurfacePs.device
        d1, d2, d3 = SurfacePs.shape
        I = torch.arange(d1 * (d2 - 1), dtype=torch.long,device=device).reshape(d1, (d2 - 1))
        I = torch.cat((I, I[:, 0:1]), dim=1)  # Concatenate along columns
        f = torch.stack((
            torch.ones((d1 - 1) * (d2 - 1), dtype=torch.long,device=device) * 4,
            I[:-1, :-1].ravel(),
            I[1:, :-1].ravel(),
            I[1:, 1:].ravel(),
            I[:-1, 1:].ravel()
        ), dim=1).to(torch.int64).ravel()
        f = f.view(-1, 5)
        f = quadrilateral_to_triangles(f)
        vertices = SurfacePs[:, :-1].reshape(-1, d3).contiguous()
        new_verts = vertices
        new_faces = f
        return new_verts, new_faces
    
def quadrilateral_to_triangles(faces):
    # Each quadrilateral is split into two triangles
    # For a quadrilateral [v0, v1, v2, v3], create two triangles:
    # Triangle 1: [v0, v1, v2]
    # Triangle 2: [v0, v2, v3]
    device = faces.device
    num_faces = faces.size(0)
    triangles = torch.zeros((num_faces * 2, 3), dtype=torch.long,device=device)
    triangles[0:num_faces, 0] = faces[:, 1]
    triangles[0:num_faces, 1] = faces[:, 2]
    triangles[0:num_faces, 2] = faces[:, 3]        
    triangles[num_faces:, 0] = faces[:, 1]
    triangles[num_faces:, 1] = faces[:, 3]
    triangles[num_faces:, 2] = faces[:, 4]        
    return triangles    

def plot_curve(CPs, CurvePs, knots, CurvePs_knots):
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import pylab
    plt.style.use('ggplot')
    params = {'legend.fontsize': 18,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
    pylab.rcParams.update(params)
    ax = plt.figure().add_subplot()
    # Plot the curve
    lw = 3
    ax.plot(CurvePs.numpy()[:,0], CurvePs.numpy()[:,1], linestyle = 'dashed', color = 'C0', alpha = 0.4, lw=lw,label = 'Original')
    ax.plot(CPs.numpy()[:,0], CPs.numpy()[:,1], color = 'C1', alpha = 1, lw=lw) # ,label = 'curve1')
    ax.scatter(CurvePs_knots.numpy()[:,0], CurvePs_knots.numpy()[:,1],marker = '^') # ,label = 'inner knots')
    for i in range(len(CPs)):
        ax.text(CPs.numpy()[i,0]+0.005, CPs.numpy()[i,1]+0.012, 'P{:d}'.format(i))
    for i in range(len(CurvePs_knots)):
        ax.text(CurvePs_knots.numpy()[i,0]+0.005, CurvePs_knots.numpy()[i,1]-0.012, '{:.2f}'.format(knots[i]))
    # leg = ax.legend(loc='upper left',frameon = True)
    # leg.get_frame().set_edgecolor('black')
    ax.set(xlabel='X', ylabel='Y')
    return ax

def plot_surface(CPs, SurfacePs):
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import pylab
    plt.style.use('ggplot')
    params = {'legend.fontsize': 18,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
    pylab.rcParams.update(params)
    ax = plt.figure().add_subplot(projection='3d')
    # Plot the 3D surface
    ax.plot_surface(SurfacePs.numpy()[:,:,0], SurfacePs.numpy()[:,:,1], SurfacePs.numpy()[:,:,2], edgecolor='royalblue', lw=0.5,
                    alpha=0.3, label = 'surface')
    ax.plot_surface(CPs.numpy()[:,:,0], CPs.numpy()[:,:,1], CPs.numpy()[:,:,2], edgecolor='black', lw=1,linestyle = 'dashed',
                    alpha=0, label = 'control points')

    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    ax.view_init(elev=20., azim=45)

def createAortaSurface(skeleton):
    import sys
    sys.path.insert(1, './')
    import numpy as np
    from registration.modules.others import remesh
    p = 3
    Cps = skeleton[:,:-1]
    if skeleton.shape[0] == 16:
        a = 100
        b = 40
    else:
        a = 30
        b = 30
    C2 = AortaBSplineSurface2D(Cps.shape[0]-p, Cps.shape[1], Cps,u_degree = p,v_degree = p,u_res= a,v_res= b)
    new_verts, new_faces = C2.export2mesh()
    return new_verts, new_faces

def createAortaSurface2vtp(skeleton):
    import sys
    sys.path.insert(1, './')
    import numpy as np
    from registration.modules.others import remesh
    p = 3
    Cps = skeleton[:,:-1]
    if skeleton.shape[0] == 16:
        a = 100
        b = 40
    else:
        a = 30
        b = 30
    C2 = AortaBSplineSurface2D(Cps.shape[0]-p, Cps.shape[1], Cps,u_degree = p,v_degree = p,u_res= a,v_res= b)
    mesh = C2.export2vtk(triangulate = True)
    return mesh





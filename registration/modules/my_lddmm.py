import numpy as np
import torch 
import pyvista as pv

# kernel 
def gaussian(r2, s):
    return torch.exp(-r2 / (s * s))

def binet(prs):
    return prs * prs


# kernel class

class TorchKernel():
    def __init__(self, kernel_width=None):
        self.kernel_width = kernel_width
    def convolve(self, x, p, y=None, mode='gaussian'):        
        if y is None:
            y = x 
        sq = self._squared_distances(y,x)
        res = torch.mm(torch.exp(-sq / (self.kernel_width ** 2)), p)
        return res
    def convolve_long(self, x, y, u, v, p, mode='gaussian'):        
        sq = self._squared_distances(y,x)
        # print('debug torch kernel: ', sq.shape, (u.unsqueeze(1) * v.unsqueeze(0)).sum(-1).shape, p.shape)
        res = torch.mm(torch.exp(-sq / (self.kernel_width ** 2))*((u.unsqueeze(1) * v.unsqueeze(0)).sum(-1))** 2, p)
        return res
    
    def convolve_gradient(self, x,px, y=None, py=None):
        if y is None:
            y = x 
        if py is None:
            py = px
        # A=exp(-(x_i - y_j)^2/(ker^2)).
        sq = self._squared_distances(x, y)
        A = torch.exp(-sq / (self.kernel_width ** 2))
        # A = 1.0 / (1 + sq / self.kernel_width ** 2) ** 2

        # B=(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        res = (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.kernel_width ** 2)).t()
        return res
    
    @staticmethod
    def _squared_distances(x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist
    
    @staticmethod
    def _differences(x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin


# lddmm class
class LDDMM():
    def __init__(self, init_control_pts, init_momenta,
                 init_template_pts, 
                 kernel_type = 'gaussian',
                 kernel_width = 0.3,
                 T = 1./256,
                 number_of_time_pts = 10, 
                 evolver_type = 'euler',
                 device = None ):
        self.init_control_pts = init_control_pts
        self.init_momenta = init_momenta    
        self.init_template_pts = init_template_pts
        self.kernel = TorchKernel(kernel_width)
        self.T = T
        self.number_of_time_pts = number_of_time_pts
        self.dt = T/number_of_time_pts
        self.evolver_type = evolver_type

    def shoot(self):
        self.control_pts_t = [self.init_control_pts]
        self.momenta_t = [self.init_momenta]
        if self.evolver_type == 'euler':
            for i in range(self.number_of_time_pts - 1):
                new_cp, new_mom = self._euler_step(self.kernel, self.control_pts_t[i], self.momenta_t[i], self.dt)
                self.control_pts_t.append(new_cp)
                self.momenta_t.append(new_mom)
        else:
            for i in range(self.number_of_time_pts - 1):
                new_cp, new_mom = self._rk2_step(self.kernel, self.control_pts_t[i], self.momenta_t[i], self.dt,
                                                return_mom=True)
                self.control_pts_t.append(new_cp)
                self.momenta_t.append(new_mom)

    def flow(self):
        self.template_pts_t = [self.init_template_pts]
        for i in range(self.number_of_time_pts - 1):
            d_pos = self.kernel.convolve( self.control_pts_t[i], self.momenta_t[i],self.template_pts_t[i])
            self.template_pts_t.append(self.template_pts_t[i] + self.dt * d_pos)


    @staticmethod
    def _euler_step(kernel, cp, mom, h):
        """
        simple euler step of length h, with cp and mom. It always returns mom.
        """
        # print(cp,h,kernel.convolve(cp, cp, mom))
        return cp + h * kernel.convolve(cp, mom), \
               mom - h * kernel.convolve_gradient(cp, mom)

    @staticmethod
    def _rk2_step(kernel, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
        also used in parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        mid_cp = cp + h / 2. * kernel.convolve(cp, mom)
        mid_mom = mom - h / 2. * kernel.convolve_gradient(cp, mom)
        if return_mom:
            return cp + h * kernel.convolve( mid_cp, mid_mom), \
                   mom - h * kernel.convolve_gradient(mid_cp, mid_mom)
        else:
            return cp + h * kernel.convolve(mid_cp, mid_mom)


def MakeNormloss(kernel_width):
    K = TorchKernel(kernel_width)
    def loss(cp, mom):
        return 0.5 * (mom * K.convolve(cp, mom)).sum()
    return loss

def MakeVarifoldloss(FS, VT, FT, kernel_width):
    K = TorchKernel(kernel_width)
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K.convolve_long(CT, CT, NTn, NTn, LT)).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        # print('debug:',CS.shape, LS.shape, NSn.shape, CT.shape, LT.shape, NTn.shape)
        return (
            cst
            + (LS * K.convolve_long(CS, CS, NSn, NSn, LS)).sum()
            - 2 * (LS * K.convolve_long(CT, CS, NSn, NTn, LT)).sum()
        )
    return loss


def MakeVarifoldloss_NURBS(VT, FT, kernel_width):
    K = TorchKernel(kernel_width)
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K.convolve_long(CT, CT, NTn, NTn, LT)).sum()

    def loss(VS,FS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        # print('debug:',CS.shape, LS.shape, NSn.shape, CT.shape, LT.shape, NTn.shape)
        return (
            cst
            + (LS * K.convolve_long(CS, CS, NSn, NSn, LS)).sum()
            - 2 * (LS * K.convolve_long(CT, CS, NSn, NTn, LT)).sum()
        )
    return loss


def vtpface2graphface(vtpface):
    return vtpface.reshape(-1,4)[:,1:]

def graphface2vtpface(graphface):
    return np.hstack((np.repeat(3, len(graphface))[...,None],graphface))



if __name__ == '__main__':
    device = torch.device("cuda:0")
    test_kernel = TorchKernel(0.1)
    X = torch.randn((20,3), device = device)
    x = torch.randn((10,3), device = device)
    p = torch.randn((10,3), device = device)
    # x_dot = test_kernel.convolve(x,p)
    # p_dot = test_kernel.convolve_gradient(x,p)
    my_lddmm = LDDMM(x,p,X)
    my_lddmm.shoot()
    my_lddmm.flow()
    # my_lddmm.convert_to_mesh()

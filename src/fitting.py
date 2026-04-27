import sys
from src.basics import * 
from src.geometries import * 

# here we implement the fitting process of a curve accroding to the nurbs book. 

####
# nonmenclature

####

def GetUbars(Qs, method = "equal"):
    '''
        input: <(n+1), 3> points Qs that need to be interpoalted 
        output: (n+1) size vector
    '''
    n = len(Qs) -1 
    if  method == "equal" : 
        return torch.linspace(0., 1., n+1)
    elif method == "chord": 
        dists = torch.cumsum(torch.norm(Qs[1:] - Qs[:-1], dim = 1), dim=0)
        return torch.cat((torch.zeros(1),dists/dists[-1]))
    elif method == "centripetal": 
        dists = torch.cumsum(torch.norm(Qs[1:] - Qs[:-1], dim = 1)**0.5, dim=0)
        return torch.cat((torch.zeros(1),dists/dists[-1]))

def Average1DFilter(U, delta):
    steps = len(U) - delta +1 
    tempU = torch.zeros(steps)
    for i in range(delta):
        tempU += U[i: i+steps]
    return tempU/delta
    
def GetKnots( Ubar, p, method = "equal"):
    '''
        input: <(n+1)> u bar
               <0> scalar of order
        output: (m+1) size vector Knot vector
    '''
    n = len(Ubar)-1 
    m = n+1+p  
    Knot = torch.zeros(m+1) 
    # set the end vectors 
    Knot[-(p+1):] = 1.

    # set the interior knots 
    if method == "equal": 
        Knot[ (p+1) : -(p+1)] = torch.arange(1, n - p + 1 ) /  (n - p + 1 )
    elif method == "average":
        Knot[ (p+1) : -(p+1)] = Average1DFilter(Ubar[1:-1], p)
    elif method == "derivative":
        m = m+2
        Knot = torch.zeros(m+1) 
        # set the end vectors 
        Knot[-(p+1):] = 1.
        Knot[ (p+1) : -(p+1)] = Average1DFilter(Ubar, p)
    elif method == "derivative_cubic_special": 
        m = m+2
        Knot = torch.zeros(m+1) 
        # set the end vectors 
        Knot[-(p+1):] = 1.
        Knot[ (p+1) : -(p+1)] = Ubar[1:-1]
    return Knot

def SovleCPs(Qs, Ubars, Knots, p, fit = "onlyQs", D0 = None, DN = None): 
    '''
        output: control points 
    '''
    kv = KnotVector(Knots[p:-p] , degree = p)
    basis = kv.get_basis_function(Ubars)[-1].T
    if fit == "onlyQs": 
        # basis = kv.get_basis_function(Ubars)[-1].T
        CPs = torch.linalg.solve(basis, Qs)
    elif "QsD0DN" in fit: 
        assert (D0!=None and DN!=None), "must provide D0 and DN"
        # the end derivatives are specified 
        # basis = kv.get_basis_function(Ubars)[-1].T # this will not be a squre shape, but rather <k, k+2>
        row1 = torch.zeros(len(Qs)+2); row1[0] = -1.; row1[1] = 1.
        row2 = torch.zeros(len(Qs)+2); row2[-2] = -1.; row2[-1] = 1.
        B1 = D0* Knots[p+1]/p
        B2 = DN* (1.-Knots[-(p+1+1)]) / p
        print(Knots)
        A = torch.vstack((basis[:1],row1.unsqueeze(0),basis[1:-1],row2.unsqueeze(0),basis[-1:]))
        B = torch.vstack((Qs[:1],B1.unsqueeze(0),Qs[1:-1],B2.unsqueeze(0),Qs[-1:]))
        print(A.shape, B.shape)
        print(A)
        print(B)
        CPs = torch.linalg.solve(A, B)
    return CPs


def Fit1DCurve(Qs,p , fit = "onlyQs", derivative_scale = 30.,  parametric_type = "chord", Knot_type = "average",D0 = None, DN = None, return_type = "class", res = 100):
    '''
        input:
            fit: "onlyQs", "QsDODN", "QSD0DN_cubic_speicial"
            parametric_type: "equal", "chord", "centripetal"
            Knot_type: "equal", "average", "derivative", "derivative_cubic_special"
            D0: first deri
            DN: last deri
            return_type: "class", "curve"
    '''
    if "QsD0DN" in fit:
        D0 *= derivative_scale; DN *= derivative_scale
        assert "derivative" in Knot_type, "must set Knot_type to derivative"
        if fit=="QsD0DN_cubic_special":
            assert p==3,"must set p=3"
            assert Knot_type == "derivative_cubic_special", "must set Knot type to cubic special"
    Ubars = GetUbars(Qs, method = parametric_type)
    Knots = GetKnots(Ubars, p, method = Knot_type)

    CPs = SovleCPs(Qs, Ubars, Knots, p, fit = fit , D0 = D0, DN = DN)
    C1 = BSplineCurve1D(Knots[p:-p],CPs,degree = p, res = res)
    if return_type == "class":
        return C1
    elif return_type == "curve":
        return C1.CurvePs

# , parametric_type = "chord", Knot_type = "averaging"




# def fitting_curve1D(Qs, parametric_type, Knot_type, D0 = None, DN = None):
#     '''
#         input: < >
#     '''












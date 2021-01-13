import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D
from sympy import ordered, Matrix

def Gradient(f, X=None):
    v = list(ordered(f.free_symbols))
    grd = lambda fn, vn: Matrix([fn]).jacobian(vn)
    if X:
        return list(grd(f, v).subs(list(zip(v, X))))
    else:
        return grd(f, v)

def find_min(f, cf, x0, epslon, max_iter=10000):
    v = list(ordered(f.free_symbols))
    A = np.array(cf[0])[np.newaxis]
    B = cf[1]
    if len(x0)<3:
        P = (1/(np.dot(A, A.T))) * np.dot(A.T, A)
    else:
        P = A.T * np.linalg.inv(A * A.T) * A

    P = np.eye(len(x0)) - P
    g = Gradient(f)
    xk = x0
    alpha = 1e-4

    for i in range(max_iter):
        g_xk = np.array(list(g.subs(list(zip(v, xk)))), dtype=np.float)
        xk1 = xk - alpha * np.dot(P, g_xk)
        if np.linalg.norm(xk1 - xk) < epslon:
            xk = xk1
            break
        xk = xk1

    return xk



if __name__ == '__main__':
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    k, m, n = sp.symbols('k m n', integer=True)
    f, g, h = sp.symbols('f g h', cls=sp.Function)
    C = CoordSys3D('C', variable_names=('x1', 'x2', 'x3'))
    sp.init_printing(use_unicode=True)
    f = x1**2 + 2*(x2**2)
    cf = [[1, 1], 3]

    print(find_min(f, cf, [0, 3], .00001))







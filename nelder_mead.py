import copy
import math
import numpy as np


def nelder_mead(f, x0, step=[1, 2], epslon=1e-6, max=100,
                lambda1=1., chi=2., rho=1, sigma=0.5, gamma=0.5):

    prev_best = .0
    res = [[x0, f(x0)]]

    for i in range(2):
        x = copy.copy(x0)
        x[i] = x[i] + step[i]
        score = f(x)
        res.append([x, score])

    for i in range(max):
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        if abs(best-prev_best) < epslon:
            break

        # centroid
        x0 = [0.] * 2
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + rho*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + chi*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + gamma*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres
    return res[0]

if __name__ == "__main__":
    def f(x):
        return 2*(x[0]**2) + 5*(x[1]**2)
        #return math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[2]) + 1))

    #print(nelder_mead(f, np.array([0., 0., 0.])))

    print(nelder_mead(f, np.array([10, 5])))



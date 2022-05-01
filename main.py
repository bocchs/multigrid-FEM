import numpy as np
import matplotlib.pyplot as plt
import distmesh as dm
import numpy.linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import sys

# Gauss-seidel from Wikipedia
def gs(A, x0, b, tol=1e-8, maxiter=3):
    for k in range(1, maxiter+1):
        x_new = np.zeros_like(x0)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1 :], x0[i + 1 :])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x0, x_new, tol):
            # print("Gauss-Seidel converged after " + str(k) + " iterations")
            break
        x0 = x_new
    return x0


# Used as smoothing in multigrid in [Johnson, Sec 7.5]
def steepest_descent(A, x0, b, tol=1e-8, maxiter=3):
    r = b - A@x0
    r_ra = []
    for k in range(1, maxiter+1):
        r_prev = r.reshape(-1,1)
        w = A@r_prev
        a = r_prev.T@r_prev / (r_prev.T@w)
        x0 = x0 + a*r_prev
        r = (r_prev - a*w).reshape(-1,1)
        if np.linalg.norm(r) <= tol:
            # print("GD converged after " + str(k) + " iterations")
            break
    return x0


# Builds system for -u_xx = 1
def get_system(m):
    # m interior nodes
    x = np.linspace(0,1,m+2)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    for j in range(1, m+1):
        hj = x[j] - x[j-1]
        hj1 = x[j+1] - x[j]
        A[j-1,j-1] = 1/hj + 1/hj1
        if j < m:
            A[j-1,j] = -1/hj1
        if j > 1:
            A[j-1,j-2] = -1/hj
        b[j-1] = hj / 2 + hj1 / 2
    return A, b, x


def two_grid_step(A, b, x, uh):
    # x is domain (0,1)
    # uh is starting solution
    m = len(x) - 2
    h = x[1] - x[0]

    uh_smooth = gs(A, uh[1:-1], b) # pre-smoothing
    rh = np.zeros((m+2,1))
    rh[1:-1] = b - A@uh_smooth # compute residual

    r2h = rh[::2][1:-1] # restrict by just keeping the coarse node values, keep interior nodes

    m2 = len(r2h)
    A2, b2, x2 = get_system(m2)
    A2_sparse = csc_matrix(A2)
    correct_2h = np.zeros((m2+2,))
    # correct_2h[1:-1] = spsolve(A2_sparse, r2h) # correct on coarse grid
    correct_2h[1:-1] = np.linalg.solve(A2, r2h).reshape(-1)
    prolong_h = np.interp(x, x2, correct_2h).reshape(-1,1) # linear interpolation (prolongation)
    uh += prolong_h
    uh[1:-1] = gs(A, uh[1:-1], b) # post-smoothing
    return uh

# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.2421&rep=rep1&type=pdf
# pg 48
def multigrid(level, A, b, x, uh):
    # x is domain (0,1)
    # uh is starting solution
    if level == 0:
        uh[1:-1] = np.linalg.solve(A, b).reshape(-1,1)
        return uh
    else:
        m = len(x) - 2
        h = x[1] - x[0]
        uh_smooth = gs(A, uh[1:-1], b) # pre-smoothing
        rh = np.zeros((m+2,1))
        rh[1:-1] = b - A@uh_smooth # compute residual
        r2h = rh[::2][1:-1] # restrict by just keeping the coarse node values, keep interior nodes
        m2 = len(r2h)
        A2, b2, x2 = get_system(m2)
        uh2 = np.zeros((m2+2,1))
        correct_2h = np.zeros((m2+2,))
        correct_2h = multigrid(level-1, A2, r2h, x2, uh2).reshape(-1) # correct on coarse grid
        prolong_h = np.interp(x, x2, correct_2h).reshape(-1,1) # linear interpolation (prolongation)
        uh += prolong_h
        return uh


if __name__ == "__main__":
    val = 8
    m = 2**val-1 # m interior nodes
    num_el = m+1
    A, b, x = get_system(m)
    A_sparse = csc_matrix(A)
    uh_direct = np.zeros((m+2,))
    uh_direct[1:-1] = spsolve(A_sparse, b)
    plt.plot(x,uh_direct,'-o',label='Direct Solution')
    plt.xlabel('x')
    plt.ylabel('u_h')
    # plt.show()
    
    # multigrid
    # uh = np.zeros((m,1))
    # uh_smooth = steepest_descent(A, uh, b)
    # rh = np.zeros((m+2,1))
    # rh[1:-1] = b - A@uh_smooth

    # # T_restrict = np.zeros(((m+1)//2+1,m+2))
    # # r2h = T_restrict @ rh # restrict residual onto coarse grid
    # # print(r2h)

    # r2h = rh[::2][1:-1] # restrict by just keeping the coarse node values, keep interior nodes
    # m2 = len(r2h)
    # A2, b2, x2 = get_system(m2)
    # A2_sparse = csc_matrix(A2)
    # correct_2h = np.zeros((m2,))
    # correct_2h[1:-1] = spsolve(A2_sparse, r2h) # correct on coarse grid
    # prolong_h = np.interp(x, x2, correct_2h) # linear interpolation
    # uh += prolong_h

    num_iters = 1
    uh_two_grid = np.zeros((m+2,1))
    for k in range(num_iters):
        uh_two_grid = two_grid_step(A, b, x, uh_two_grid)
    uh_two_grid = uh_two_grid.reshape(-1)
    plt.plot(x,uh_two_grid,'-o',label='Two Grid Solution')

    uh_multigrid = np.zeros((m+2,1))
    levels = 1
    for k in range(num_iters):
        uh_multigrid = multigrid(levels, A, b, x, uh_multigrid)
    uh_multigrid = uh_multigrid.reshape(-1)
    plt.plot(x,uh_multigrid,'-o',label='Multigrid Solution')

    exact_soln = -1/2*x*(x-1)
    plt.plot(x,exact_soln,'-o',label='Exact Solution')
    plt.legend()
    plt.show()

    h = x[1] - x[0]
    two_grid_err = np.sqrt(h*np.sum((exact_soln-uh_two_grid)**2))
    print("two grid err = " + str(two_grid_err))

    multigrid_err = np.sqrt(h*np.sum((exact_soln-uh_multigrid)**2))
    print("multigrid err = " + str(multigrid_err))

    direct_err = np.sqrt(h*np.sum((exact_soln-uh_direct)**2))
    print("direct err = " + str(direct_err))



    sys.exit()

    # fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    fd = lambda p: dm.drectangle0(p,0,1,0,1)
    # fh = lambda p: 0.00+1*dm.dcircle(p,0,0,0.5)
    fh = dm.huniform
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # p: node positions Nx2
    # t: triangle indices nKx3
    p, t = dm.distmesh2d(fd, fh, .3, (0,0,1,1), pfix=nodes)
    p, t = dm.uniref(p, t)
    plt.figure()
    for inds in t:
        plt.scatter(p[inds,0], p[inds,1], c='b')
        plt.plot(p[inds,0], p[inds,1], '-b')
    plt.axis('equal')
    plt.title('2.a. Mesh')
    print(t[0])
    print(p[16])
    print(p[2])
    print(p[15])
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import distmesh as dm
import numpy.linalg
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


# Builds system for -u_xx = 1, u(0) = u(1) = 0
def get_system1D_old(m):
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


# Builds system for -u_xx = c*cos(ax)
def get_system1D_uneven_rhs_cos(x, a, c):
    # x[0] = x[m+1] = 0
    # m interior nodes
    m = len(x) - 2
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
        b[j-1] = rhs(x[j-1], x[j], x[j+1], a, c)
    return A, b


# Builds system for -u_xx = 1, u(0) = u(1) = 0
def get_system1D_uneven(x):
    # x[0] = x[m+1] = 0
    # m interior nodes
    m = len(x) - 2
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
    return A, b


# Builds system for -u_xx = 1, u(0) = u(1) = 0
def get_system1D(m):
    # m interior nodes
    h = 1 / (m+1)
    x = np.linspace(0,1,m+2)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    diagonals = [m*[2/h], m*[-1/h], m*[-1/h]]
    A = np.diag(m*[2/h], k=0) + np.diag((m-1)*[-1/h], k=1) + np.diag((m-1)*[-1/h], k=-1)
    b = h*np.ones((m,1))
    return A, b, x


def two_grid_step1D(A, b, x, uh):
    # x is domain (0,1)
    # uh is starting solution
    m = len(x) - 2
    h = 1 / (m+1)
    uh_smooth = gs(A, uh[1:-1], b) # pre-smoothing
    rh = np.zeros((m+2,1))
    rh[1:-1] = b - A@uh_smooth # compute residual
    r2h = rh[::2][1:-1] # restrict by just keeping the coarse node values, keep interior nodes
    m2 = len(r2h)
    A2, b2, x2 = get_system1D(m2)
    error_2h = np.zeros((m2+2,))
    error_2h[1:-1] = np.linalg.solve(A2, r2h).reshape(-1) # compute error on coarse grid
    error_h = np.interp(x, x2, error_2h).reshape(-1,1) # linear interpolation (prolongation)
    uh += error_h # add correction fine grid
    uh[1:-1] = gs(A, uh[1:-1], b) # post-smoothing
    return uh

# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.2421&rep=rep1&type=pdf
# pg 48
def multigrid1D(level, A_levels, b, x_levels, uh):
    A = A_levels[level]
    x = x_levels[level]
    if level == 0:
        uh[1:-1] = np.linalg.solve(A, b).reshape(-1,1) # solve error on coarse grid
        return uh
    else:
        m = len(x) - 2
        uh_smooth = gs(A, uh[1:-1], b) # pre-smoothing
        rh = np.zeros((m+2,1))
        rh[1:-1] = b - A@uh_smooth # compute residual
        r2h = rh[::2][1:-1] # restrict by just keeping the coarse nodes ([::2]), keep interior nodes ([1:-1])
        m2 = len(r2h)
        uh2 = np.zeros((m2+2,1))
        error_2h = multigrid1D(level-1, A_levels, r2h, x_levels, uh2).reshape(-1) # compute error on coarse grid
        x2 = x_levels[level-1]
        error_h = np.interp(x, x2, error_2h).reshape(-1,1) # linear interpolation (prolongation)
        uh += error_h # add correction on fine grid
        uh_smooth = np.zeros((m+2,1))
        uh_smooth[1:-1] = gs(A, uh[1:-1], b) # post-smoothing
        return uh_smooth

def run_multigrid_1D_experiments():
    # -u_xx = 1
    levels = 7
    m = 2**levels-1 # m interior nodes
    h = 1 / (m+1)
    A, b, x = get_system1D(m)
    exact_soln = -1/2*x*(x-1)
    uh_direct = np.zeros((m+2,))
    # uh_direct[1:-1] = spsolve(A, b)
    uh_direct[1:-1] = np.linalg.solve(A, b).reshape(-1)
    direct_err = np.sqrt(h*np.sum((exact_soln-uh_direct)**2))
    print("direct err = " + str(direct_err))
    plt.plot(x,uh_direct,'-o',label='Direct Solution')
    plt.xlabel('x')
    plt.ylabel('u_h')

    num_iters = 100

    uh_two_grid = np.zeros((m+2,1))
    for k in range(num_iters):
        uh_two_grid = two_grid_step1D(A, b, x, uh_two_grid)
    uh_two_grid = uh_two_grid.reshape(-1)
    two_grid_err = np.sqrt(h*np.sum((exact_soln-uh_two_grid)**2))
    print("two grid err = " + str(two_grid_err))
    plt.plot(x,uh_two_grid,'-o',label='Two Grid Solution')

    # intialize grids
    uh_multigrid = np.zeros((m+2,1))
    A_levels = []
    x_levels = []
    # b = []
    for level in range(1,levels+1):
        m = 2**level-1
        A, b, x = get_system1D(m)
        A_levels.append(A)
        x_levels.append(x)
    # run multigrid
    for k in range(num_iters):
        uh_multigrid = multigrid1D(levels-1, A_levels, b, x_levels, uh_multigrid)
    uh_multigrid = uh_multigrid.reshape(-1)
    multigrid_err = np.sqrt(h*np.sum((exact_soln-uh_multigrid)**2))
    print("multigrid err = " + str(multigrid_err))
    plt.plot(x,uh_multigrid,'-o',label='Multigrid Solution')

    
    plt.plot(x,exact_soln,'-o',label='Exact Solution')
    plt.legend()
    plt.show()

# f = c*cos(ax)
def rhs(xj_1, xj, xj1, a, c):
    hj = xj - xj_1
    hj1 = xj1 - xj
    term1 = 1/hj * (1/a*xj*np.sin(a*xj) + 1/a**2 * np.cos(a*xj) - (1/a*xj_1*np.sin(a*xj_1) + 1/a**2 * np.cos(a*xj_1)))
    term2 = -xj_1 / hj * (1/a * np.sin(a*xj) - (1/a * np.sin(a*xj_1)))
    term3 = 1 / hj1 * (1/a * xj1 * np.sin(a*xj1) + 1/a**2*np.cos(a*xj1) - (1/a * xj * np.sin(a*xj) + 1/a**2*np.cos(a*xj)))
    term4 = -xj / hj1 * (1/a * np.sin(a*xj1) - (1/a*np.sin(a*xj)))
    return c*(term1 + term2 + term3 + term4)


def adaptive_step(x, u):
    m = len(x) - 2
    delta = 2
    bisect_thresh = delta**2 / len(x)
    x_new = []
    x_new.append(x[0])
    x_new.append(x[1])
    for i in range(2, m):
        ux_back = (u[i] - u[i-1]) / (x[i] - x[i-1])
        ux_for = (u[i+1] - u[i]) / (x[i+1] - x[i])
        uxx = (ux_for - ux_back) / (x[i+1] - x[i-1])
        h2_norm = np.sqrt(np.abs(u[i]) + np.abs(ux_for) + np.abs(uxx))
        print(h2_norm)
        print(bisect_thresh)
        if h2_norm > bisect_thresh:
            x_new.append((x[i] + x[i-1]) / 2)
            x_new.append(x[i])
            x_new.append((x[i+1] + x[i]) / 2)
            # might have duplicate x_new's, handle by calling np.unique()
        else:
            x_new.append(x[i])
    x_new.append(x[-2])
    x_new.append(x[-1])
    x_new = np.array(x_new)
    x_new = np.unique(x_new)

    # A_new, b_new = get_system1D_uneven(x_new)
    A_new, b_new = get_system1D_uneven_rhs_cos(x_new, 1, 1)
    m_new = len(x_new) - 2
    u_new = np.zeros((m_new+2,))
    u_new[1:-1] = np.linalg.solve(A_new, b_new).reshape(-1)

    return x_new, u_new



def adaptive():
    num_iters = 0
    levels = 4
    m = 2**levels-1 # m interior nodes
    h = 1 / (m+1)
    A, b, x = get_system1D(m)
    exact_soln = -1/2*x*(x-1)


    x = np.linspace(0,1,m+2)    
    a = 10*np.pi
    c = a**2
    A, b = get_system1D_uneven_rhs_cos(x, a, c)
    exact_soln = -c / a**2 * np.cos(a*x)


    u = np.zeros((m+2,))
    u[1:-1] = np.linalg.solve(A, b).reshape(-1)
    plt.plot(x, u, '^g',label='Base Grid')

    for i in range(num_iters):
        print(i)
        x, u = adaptive_step(x, u)
    plt.plot(x, u, 'vr',label='Adapted Grid')
    plt.legend()
    plt.show()

    sys.exit()

    # uh_direct[1:-1] = spsolve(A, b)
    u[1:-1] = np.linalg.solve(A, b).reshape(-1)
    direct_err = np.sqrt(h*np.sum((exact_soln-u)**2))

    delta = 1
    bisect_thresh = delta**2 / len(x)

    x_new = []
    x_new.append(x[0])
    x_new.append(x[1])
    for i in range(2, m):
        ux_back = (u[i] - u[i-1]) / (x[i] - x[i-1])
        ux_for = (u[i+1] - u[i]) / (x[i+1] - x[i])
        uxx = (ux_for - ux_back) / (x[i+1] - x[i-1])
        h2_norm = np.sqrt(np.abs(u[i]) + np.abs(ux_for) + np.abs(uxx))
        # print(h2_norm)
        # print(bisect_thresh)
        if h2_norm > bisect_thresh:
            x_new.append((x[i] + x[i-1]) / 2)
            x_new.append(x[i])
            x_new.append((x[i+1] + x[i]) / 2)
            # might have duplicate x_new's, handle by calling np.unique()
        else:
            x_new.append(x[i])
    x_new.append(x[-2])
    x_new.append(x[-1])
    x_new = np.array(x_new)
    x_new = np.unique(x_new)

    A_new, b_new = get_system1D_uneven(x_new)

    m_new = len(x_new) - 2
    u_new = np.zeros((m_new+2,))
    u_new[1:-1] = np.linalg.solve(A_new, b_new).reshape(-1)

    plt.plot(x_new, u_new, 'vr',label='Adapted Grid')
    plt.plot(x, u, '^g',label='Base Grid')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # run_multigrid_1D_experiments()
    adaptive()

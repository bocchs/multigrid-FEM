import numpy as np
import matplotlib.pyplot as plt
import distmesh as dm
import numpy.linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import sys

def plot_mesh(p,t):
    for inds in t:
        plt.plot(p[inds[[0, 1]],0], p[inds[[0,1]],1], '-b')
        plt.plot(p[inds[[0, 2]],0], p[inds[[0,2]],1], '-b')
        plt.plot(p[inds[[1, 2]],0], p[inds[[1,2]],1], '-b')
    plt.scatter(p[t,0], p[t,1], c='b')
    plt.xlim([-.1, 1.1])
    plt.ylim([-.1, 1.1])
    plt.show()

def plot_mesh_par(p,t,parents):
    e = dm.boundedges(p,t)
    boundary_inds = np.unique(e) # indices of boundary nodes
    for inds in t:
        for child in inds:
            if child not in boundary_inds:
                pars = parents[child]
                # plt.scatter(p[inds,0], p[inds,1], c='b')
                # plt.plot(p[inds[[0, 1]],0], p[inds[[0,1]],1], '-b')
                # plt.plot(p[inds[[0, 2]],0], p[inds[[0,2]],1], '-b')
                # plt.plot(p[inds[[1, 2]],0], p[inds[[1,2]],1], '-b')
                plt.scatter(p[child,0], p[child,1], c='g')
                plt.scatter(p[pars,0], p[pars,1], c='r')
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.show()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()


def plot_solution(title, filename, p, t, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(p[:,0], p[:,1], u, triangles=t, cmap=plt.cm.viridis)
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    # ax.set_zlim(0,.08)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()
    # plt.savefig(filename)

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


# returns node numbers (indices) that make up triangle k's neighboring triangles
def get_neighboring_triangles(t, level_triangles, k):
    assert(k in level_triangles)
    neighboring_triangles = np.zeros((0,), int)
    tri_k_nodes = t[k]
    triangle_counts = {} # key: triangle ID, value: number of nodes in triangle shared with triangle k
    for triangle in level_triangles:
        triangle_counts[triangle] = 0
    # iterate over each node in triangle k
    for node in tri_k_nodes:
        # iterate over each triangle, check if node in triangle
        for triangle in level_triangles:
            # check if node is shared between triangle k and current triangle
            if node in t[triangle] and not (triangle == k):
                triangle_counts[triangle] += 1
    # threshold number of shared nodes for classifying as neighboring triangle
    for triangle in level_triangles:
        if triangle_counts[triangle] >= 2:
            neighboring_triangles = np.concatenate((neighboring_triangles, np.array([triangle],int)))
    return neighboring_triangles


def create_meshes(levels):
    # m interior nodes
    # p: node positions Nx2
    # t: triangle indices nKx3
    p = np.array([[0,0], [1,0], [1,1], [0,1]])
    t = np.array([[0, 1, 2], [0, 2, 3]])
    parents = {} # keeps track of all parents of a node resulting from bisection
    parents[0] = np.array([[-1,-1]])
    parents[1] = np.array([[-1,-1]])
    parents[2] = np.array([[-1,-1]])
    parents[3] = np.array([[-1,-1]])
    # dictionary of triangle indices in t corresponding to a level
    # level_triangles[0] corresponds to coarsest grid
    level_triangles = {}
    level_triangles[0] = np.array([0,1], int)
    level_nodes = {} # dictionary of nodes that were added for a level
    level_nodes[0] = np.array([0,1,2,3], int)
    num_nodes_up_to_level = {} # total number of nodes in a level
    num_nodes_up_to_level[0] = 4
    t_count = 2
    for lev in range(1, levels+1):
        level_triangles[lev] = np.zeros((0,), int)
        level_nodes[lev] = np.zeros((0,), int)
        # iterate over each triangle in previous level
        for k in level_triangles[lev-1]:
            tri_inds = t[k] # get triangle node inds
            node0 = tri_inds[0]
            node1 = tri_inds[1]
            node2 = tri_inds[2]
            p0 = p[node0] # curr triangle nodes
            p1 = p[node1]
            p2 = p[node2]
            # bisect triangle
            new_p0 = ((p1 + p0) / 2).reshape(1,2)
            new_p1 = ((p2 + p1) / 2).reshape(1,2)
            new_p2 = ((p2 + p0) / 2).reshape(1,2)

            # check if node already exists (from bisecting other triangle)
            dists0 = np.linalg.norm(new_p0 - p, axis=1)
            closeness0 = np.isclose(dists0,0)
            assert(closeness0.sum() <= 1)
            if np.any(closeness0):
                # node already exists, use it
                new_node0 = np.argmin(dists0)
            else:
                # node does not already exist, so add it
                p = np.concatenate((p, new_p0), axis=0)
                new_node0 = p.shape[0]-1
                level_nodes[lev] = np.concatenate((level_nodes[lev], np.array([new_node0])),axis=0)
                par0 = np.array([node0, node1])
                parents[new_node0] = par0

            dists1 = np.linalg.norm(new_p1 - p, axis=1)
            closeness1 = np.isclose(dists1,0)
            assert(closeness1.sum() <= 1)
            if np.any(closeness1):
                new_node1 = np.argmin(dists1)
            else:
                p = np.concatenate((p, new_p1), axis=0)
                new_node1 = p.shape[0]-1
                level_nodes[lev] = np.concatenate((level_nodes[lev], np.array([new_node1])),axis=0)
                par1 = np.array([node1, node2])
                parents[new_node1] = par1

            dists2 = np.linalg.norm(new_p2 - p, axis=1)
            closeness2 = np.isclose(dists2,0)
            assert(closeness2.sum() <= 1)
            if np.any(closeness2):
                new_node2 = np.argmin(dists2)
            else:
                p = np.concatenate((p, new_p2), axis=0)
                new_node2 = p.shape[0]-1
                level_nodes[lev] = np.concatenate((level_nodes[lev], np.array([new_node2])),axis=0)
                par2 = np.array([node0, node2])
                parents[new_node2] = par2

            new_triangle0 = np.array([[node0, new_node0, new_node2]])
            new_triangle1 = np.array([[new_node0, node1, new_node1]])
            new_triangle2 = np.array([[new_node0, new_node1, new_node2]])
            new_triangle3 = np.array([[new_node2, new_node1, node2]])
            t = np.concatenate((t,new_triangle0,new_triangle1,new_triangle2,new_triangle3), axis=0)
            
            level_triangles[lev] = np.concatenate((level_triangles[lev], 
                            np.array([t_count,t_count+1,t_count+2,t_count+3], int)), axis=0)
            t_count += 4
        num_nodes_up_to_level[lev] = num_nodes_up_to_level[lev-1] + level_nodes[lev].shape[0]

    return p, t, level_triangles, level_nodes, num_nodes_up_to_level, parents


# Builds -(u_xx + u_yy) = 1, u(0) = u(1) = 0
def build_system(p, t, level, num_nodes_up_to_level):
    # p: node positions Nx2
    # t: triangle indices nKx3

    nK = t.shape[0] # nK triangles
    N = num_nodes_up_to_level[level] # N nodes

    e = dm.boundedges(p,t)
    boundary_inds = np.unique(e) # indices of boundary nodes
    for ind in boundary_inds:
        if 0 not in p[ind] and 1 not in p[ind]:
            boundary_inds = boundary_inds[boundary_inds != ind]
    nv = N - len(boundary_inds)

    A = np.zeros((N, N))
    b = np.zeros((N,))
    for k in range(nK):
        tri_inds = t[k] # get kth triangle vertex indices
        a1 = p[tri_inds[0]] # get kth triangle vertices
        a2 = p[tri_inds[1]]
        a3 = p[tri_inds[2]]
        area = .5 * ((a2[0]-a1[0])*(a3[1]-a1[1]) - (a3[0]-a1[0])*(a2[1]-a1[1]))
        u = a2 - a3
        v = a3 - a1
        w = a1 - a2
        aK = np.array([ [np.dot(u,u), np.dot(u,v), np.dot(u,w)], 
                        [np.dot(v,u), np.dot(v,v), np.dot(v,w)], 
                        [np.dot(w,u), np.dot(w,v), np.dot(w,w)] 
                    ]) / (4 * area)
        for i1 in range(3):
            i = t[k,i1]
            b[i] += area / 3
            for j1 in range(3):
                j = t[k,j1]
                A[i,j] += aK[i1,j1]

    # enforce boundary conditions
    for bound_ind in boundary_inds:
        A[bound_ind] = np.zeros((N,))
        A[bound_ind, bound_ind] = 1
        b[bound_ind] = 0

    return A, b


def multigrid2D(level, A_levels, b, uh, parents, level_nodes, num_nodes_up_to_level):
    A = A_levels[level]
    num_nodes_in_level = num_nodes_up_to_level[level] #len(level_nodes[level])
    coarser_level_end_node = num_nodes_up_to_level[level]
    if level == 0:
        uh = np.linalg.solve(A, b).reshape(-1,1) # solve error on coarse grid
        return uh
    else:
        uh_smooth = gs(A, uh, b) # pre-smoothing
        rh = b.reshape(-1,1) - A@uh_smooth # compute residual
        r2h = rh[:coarser_level_end_node] # restrict only to previous level's nodes
        uh2 = np.zeros((len(r2h),1))
        # compute error on coarse grid
        error_2h = multigrid2D(level-1, A_levels, r2h, uh2, parents, level_nodes, num_nodes_up_to_level).reshape(-1)
        error_h = np.zeros((len(rh),))
        error_h[:coarser_level_end_node] = error_2h
        for i, node in enumerate(level_nodes[level]):
            # interpolate using parent nodes
            error_h[coarser_level_end_node+i] = (error_2h[parents[node][0]] + error_2h[parents[node][1]]) / 2
        uh += error_h.reshape(-1,1) # add correction on fine grid
        uh_smooth = gs(A, uh, b) # post-smoothing
        return uh_smooth


def run_multigrid_2D_experiments():
    num_iters = 10
    levels = 6
    p, t, level_triangles, level_nodes, num_nodes_up_to_level, parents = create_meshes(levels)
    t_level = t[level_triangles[levels]]

    uh_multigrid = np.zeros((p.shape[0],1))
    A_levels = []
    x_levels = []
    for level in range(1,levels+1):
        t_level = t[level_triangles[level]]
        A, b = build_system(p, t_level, level, num_nodes_up_to_level)
        A_levels.append(A)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # run multigrid
    for k in range(num_iters):
        uh_multigrid = multigrid2D(levels-1, A_levels, b, uh_multigrid, parents, level_nodes, num_nodes_up_to_level)
        title = 'temp'
        filename = 'temp.png'
    plot_solution(title, filename, p, t_level, uh_multigrid.reshape(-1))
    plt.show()


    direct_u = np.linalg.solve(A, b)
    title = 'Direct Solution'
    filename = 'exact.png'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_solution(title, filename, p, t_level, direct_u)
    plt.show()


def adaptive_step(levels, p, t, level_triangles, level_nodes, num_nodes_up_to_level):
    # print(level_triangles[levels])
    # print(t.shape)
    # print("t before = " + str(t.shape[0]))
    t_level = t[level_triangles[levels]]

    A, b = build_system(p, t_level, levels, num_nodes_up_to_level)
    A_sparse = csc_matrix(A)
    u = spsolve(A_sparse, b)

    # intialize starting grid level
    t_count = 0
    for i in range(levels+1):
        t_count += len(level_triangles[i])
    t_count = np.max(level_triangles[levels])
    delta = 1.5
    num_triangles_level = len(level_triangles[levels])
    bisect_thresh = delta**2 / num_triangles_level

    triangles_to_bisect = np.zeros((0,), int)
    level_triangles[levels+1] = level_triangles[levels]
    # t = np.concatenate((t, t[level_triangles[levels]]), axis=0)

    level_nodes[levels+1] = np.zeros((0,), int)

    for triangle in level_triangles[levels]:
        x0, y0 = p[t[triangle][0]]
        x1, y1 = p[t[triangle][1]]
        x2, y2 = p[t[triangle][2]]

        # use average of three nodes as value in center of triangle
        x_center = (x0 + x1 + x2) / 3
        y_center = (y0 + y1 + y2) / 3
        u_center = np.mean(u[t[triangle]])
        
        nei_tri = get_neighboring_triangles(t, level_triangles[levels], triangle)
        max_ux = 0
        max_uy = 0
        max_uxx = 0
        max_uyy = 0

        for tri1 in nei_tri:
            x0, y0 = p[t[tri1][0]]
            x1, y1 = p[t[tri1][1]]
            x2, y2 = p[t[tri1][2]]
            x_center_nei1 = (x0 + x1 + x2) / 3
            y_center_nei1 = (y0 + y1 + y2) / 3
            u_center_nei1 = np.mean(u[t[tri1]])
            hx = x_center_nei1 - x_center
            if hx != 0:
                ux = (u_center_nei1 - u_center) / hx
                if np.abs(ux) > max_ux:
                    max_ux = np.abs(ux)
            hy = y_center_nei1 - y_center
            if hy != 0:
                uy = (u_center_nei1 - u_center) / hy
                if np.abs(uy) > max_uy:
                    max_uy = np.abs(uy)

            for tri2 in nei_tri:
                if tri2 == tri1:
                    continue
                x0, y0 = p[t[tri2][0]]
                x1, y1 = p[t[tri2][1]]
                x2, y2 = p[t[tri2][2]]
                x_center_nei2 = (x0 + x1 + x2) / 3
                y_center_nei2 = (y0 + y1 + y2) / 3
                u_center_nei2 = np.mean(u[t[tri1]])
                # print("neighbor2 (x,y) = (" + str(x_center_nei2) + ", " + str(y_center_nei2) + ")")
                hx2 = x_center_nei2 - x_center_nei1
                if hx2 != 0:
                    uxx = (u_center_nei1 - 2*u_center + u_center_nei2) / hx2
                    if np.abs(uxx) > max_uxx:
                        max_uxx = np.abs(uxx)
                hy2 = y_center_nei2 - y_center_nei1
                if hy2 != 0:
                    uyy = (u_center_nei1 - 2*u_center + u_center_nei2) / hy2
                    if np.abs(uyy) > max_uyy:
                        max_uyy = np.abs(uyy)
        H2_norm = np.sqrt(np.abs(u_center) + max_ux + max_uy + max_uxx + max_uyy)
        side1_len = np.linalg.norm([x1-x0, y1-y0])
        side2_len = np.linalg.norm([x2-x0, y2-y0])
        side3_len = np.linalg.norm([x2-x1, y2-y1])
        hk = np.max([side1_len, side2_len, side3_len])
        val = (hk * H2_norm)**2
        # print("val = " + str(val))
        # print("thresh = " + str(bisect_thresh))
        if val > bisect_thresh:
            triangles_to_bisect = np.concatenate((triangles_to_bisect, np.array([triangle], int)))


    for k in triangles_to_bisect:
        tri_inds = t[k] # get triangle node inds
        node0 = tri_inds[0]
        node1 = tri_inds[1]
        node2 = tri_inds[2]
        p0 = p[node0] # curr triangle nodes
        p1 = p[node1]
        p2 = p[node2]
        # bisect triangle
        new_p0 = ((p1 + p0) / 2).reshape(1,2)
        new_p1 = ((p2 + p1) / 2).reshape(1,2)
        new_p2 = ((p2 + p0) / 2).reshape(1,2)

        # check if node already exists (from bisecting other triangle)
        dists0 = np.linalg.norm(new_p0 - p, axis=1)
        closeness0 = np.isclose(dists0,0)
        assert(closeness0.sum() <= 1)
        if np.any(closeness0):
            # node already exists, use it
            new_node0 = np.argmin(dists0)
        else:
            # node does not already exist, so add it
            p = np.concatenate((p, new_p0), axis=0)
            new_node0 = p.shape[0]-1
            level_nodes[levels+1] = np.concatenate((level_nodes[levels+1], np.array([new_node0])),axis=0)

        dists1 = np.linalg.norm(new_p1 - p, axis=1)
        closeness1 = np.isclose(dists1,0)
        assert(closeness1.sum() <= 1)
        if np.any(closeness1):
            new_node1 = np.argmin(dists1)
        else:
            p = np.concatenate((p, new_p1), axis=0)
            new_node1 = p.shape[0]-1
            level_nodes[levels+1] = np.concatenate((level_nodes[levels+1], np.array([new_node1])),axis=0)

        dists2 = np.linalg.norm(new_p2 - p, axis=1)
        closeness2 = np.isclose(dists2,0)
        assert(closeness2.sum() <= 1)
        if np.any(closeness2):
            new_node2 = np.argmin(dists2)
        else:
            p = np.concatenate((p, new_p2), axis=0)
            new_node2 = p.shape[0]-1
            level_nodes[levels+1] = np.concatenate((level_nodes[levels+1], np.array([new_node2])),axis=0)

        new_triangle0 = np.array([[node0, new_node0, new_node2]])
        new_triangle1 = np.array([[new_node0, node1, new_node1]])
        new_triangle2 = np.array([[new_node0, new_node1, new_node2]])
        new_triangle3 = np.array([[new_node2, new_node1, node2]])
        t = np.concatenate((t,new_triangle0,new_triangle1,new_triangle2,new_triangle3), axis=0)

        level_triangles[levels+1] = np.concatenate((level_triangles[levels+1], 
                        np.array([t_count,t_count+1,t_count+2,t_count+3], int)), axis=0)
        t_count += 4

    # remove bisected triangles
    for k in triangles_to_bisect:
        level_triangles[levels+1] = level_triangles[levels+1][level_triangles[levels+1] != k]
    # print("num bisected = " + str(len(triangles_to_bisect)))

    num_nodes_up_to_level[levels+1] = num_nodes_up_to_level[levels] + level_nodes[levels+1].shape[0] 

    # print("t after = " + str(t.shape[0]))
    return p, t, level_triangles, level_nodes, num_nodes_up_to_level


def run_adaptive_2D_experiments():
    start_level = 2
    p, t, level_triangles, level_nodes, num_nodes_up_to_level, parents = create_meshes(start_level)
    t_level = t[level_triangles[start_level]]

    num_iters = 2
    end_level = start_level + num_iters
    for lev in range(start_level, end_level):
        p, t, level_triangles, level_nodes, num_nodes_up_to_level = \
                    adaptive_step(lev, p, t, level_triangles, level_nodes, num_nodes_up_to_level)


    plot_mesh(p,t[level_triangles[end_level]])
    A, b = build_system(p, t[level_triangles[end_level]], end_level, num_nodes_up_to_level)
    A_sparse = csc_matrix(A)
    u = spsolve(A_sparse, b)
    plot_solution('', 'a', p, t[level_triangles[end_level]], u)
    plt.show()


if __name__ == "__main__":
    run_multigrid_2D_experiments()
    # run_adaptive_2D_experiments()



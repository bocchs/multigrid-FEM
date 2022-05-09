import numpy as np
import matplotlib.pyplot as plt
import distmesh as dm
import numpy.linalg
import sys

def plot_mesh(p,t):
    for inds in t:
        plt.scatter(p[inds,0], p[inds,1], c='b')
        plt.plot(p[inds[[0, 1]],0], p[inds[[0,1]],1], '-b')
        plt.plot(p[inds[[0, 2]],0], p[inds[[0,2]],1], '-b')
        plt.plot(p[inds[[1, 2]],0], p[inds[[1,2]],1], '-b')
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


def plot_solution(title, filename, p, t, u, ax):
    ax.clear()
    ax.plot_trisurf(p[:,0], p[:,1], u, triangles=t, cmap=plt.cm.viridis)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax.set_zlim(0,.08)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.pause(.0001)
    # plt.show()
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
            if node in t[triangle] and not triangle == k:
                triangle_counts[triangle] += 1
    # threshold number of shared nodes for classifying as neighboring triangle
    for triangle in level_triangles:
        if triangle_counts[triangle] >= 1:
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
    num_iters = 1000
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
        plot_solution(title, filename, p, t_level, uh_multigrid.reshape(-1), ax)
    plt.show()


    direct_u = np.linalg.solve(A, b)
    title = 'Direct Solution'
    filename = 'exact.png'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_solution(title, filename, p, t_level, direct_u, ax)
    plt.show()


def run_adaptive_2D_experiments():
    levels = 4
    p, t, level_triangles, level_nodes, num_nodes_up_to_level, parents = create_meshes(levels)
    t_level = t[level_triangles[levels]]

    print(level_triangles[levels])
    for triangle in level_triangles[levels]:
        x0, y0 = p[t[triangle][0]]
        x1, y1 = p[t[triangle][1]]
        x2, y2 = p[t[triangle][2]]

        x_center = (x0 + x1 + x2) / 3
        y_center = (y0 + y1 + y2) / 3
        u_center = np.mean(u[t[triangle]])
        print(u_center)
        
        
        nei_tri = get_neighboring_triangles(t, level_triangles[levels], triangle)
        for tri in nei_tri:
            x0, y0 = p[t[tri][0]]
            x1, y1 = p[t[tri][1]]
            x2, y2 = p[t[tri][2]]
            x_center_nei = (x0 + x1 + x2) / 3
            y_center_nei = (y0 + y1 + y2) / 3
            u_center_nei = np.mean(u[t[tri]])

            

            
    # plot_mesh(p,t[nei_tri])
    # plot_mesh(p,t[[triangle]])

    


if __name__ == "__main__":
    # run_multigrid_2D_experiments()
    run_adaptive_2D_experiments()



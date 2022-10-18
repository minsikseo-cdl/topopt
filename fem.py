from itertools import combinations
from multiprocessing.dummy import Array
from typing import Tuple, Union, Callable, List
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.config import config
from mesh import Mesh


ArrayLike = Union[List[float], np.ndarray, jnp.DeviceArray]


config.update('jax_enable_x64', True)


GQ3 = {
    1: ([1], [(1/3, 1/3)]),
    2: ([1/3, 1/3, 1/3], [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)])
}


GQ4 = {
    1: ([1], [(1/4, 1/4, 1/4)]),
    2: ([0.0416666667, 0.0416666667, 0.0416666667, 0.0416666667], 
        [(0.5854101966, 0.1381966011, 0.1381966011), (0.1381966011, 0.5854101966, 0.1381966011), 
         (0.1381966011, 0.1381966011, 0.5854101966), (0.1381966011, 0.1381966011, 0.1381966011)])
}


def shapefn(*s: Tuple[float]) -> jnp.DeviceArray:
    """ Isoparametric element shape functions
    s = xi for 1-D
    s = [xi, eta] for 2-D
    s = [xi, eta, zeta] for 3-D

    N = [1 - xi, xi] for 1-D
    N = [1 - xi - eta, xi, eta] for 2-D
    N = [1 - xi - eta - zeta, xi, eta, zeta] for 3-D
    """
    ndim = len(s)
    if ndim == 1:
        return jnp.asarray([1 - s[0], s[0]])
    elif ndim == 2:
        return jnp.asarray([1 - s[0] - s[1], s[0], s[1]])
    elif ndim == 3:
        return jnp.asarray([1 - s[0] - s[1] - s[2], s[0], s[1], s[2]])


def gradfn(*s: Tuple[float]) -> jnp.DeviceArray:
    """ Gradients of shape function w.r.t. isoparametric coordinates
    s = xi for 1-D
    s = [xi, eta] for 2-D
    s = [xi, eta, zeta] for 3-D

    dNds = dNdxi for 1-D
    dNds = [dNdxi, dNdeta] for 2-D
    dNds = [dNdxi, dNdeta, dNdzeta] for 3-D
    """
    ndim = len(s)
    if ndim == 1:
        return jnp.asarray([-1., 1.])
    elif ndim == 2:
        return jnp.asarray([[-1., 1., 0.], [-1., 0., 1.]])
    elif ndim == 3:
        return jnp.asarray([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])


def chainrule(*s: Tuple[float], X: ArrayLike) -> Tuple[jnp.DeviceArray]:
    """ Compute gradients of shape function w.r.t. spatial coordinates r
    r = [x, y] for 2-D
    r = [x, y, z] for 3-D

    dNdr = [dNdx, dNdy] for 2-D
    dNdr = [dNdx, dNdy, dNdz] for 3-D

    Returns
    =======
    detJ, dNdr
    """
    _, ndim = X.shape
    assert len(s) == ndim
    dNds = gradfn(*s)
    J = dNds@X # Jacobian drds
    return jnp.linalg.det(J), jnp.linalg.solve(J, dNds)


def strainmat(dNdr: ArrayLike) -> jnp.DeviceArray:
    n = len(dNdr)
    I = jnp.eye(n)
    B = [jnp.kron(dN, i) for dN, i in zip(dNdr, I)]
    ij = list(combinations(range(n), 2))
    ij.reverse()
    for i, j in ij:
        B += [jnp.kron(dNdr[i], I[j]) + jnp.kron(dNdr[j], I[i])]
    return jnp.stack(B)


@jit
def compute_stiffness_matrix(D: jnp.DeviceArray, X: jnp.DeviceArray) -> jnp.DeviceArray:
    nn, ndim = X.shape
    order = 1 if D.ndim == 2 else 2
    ws, ss = GQ3[order] if nn == 3 else GQ4[order]
    Ke = jnp.zeros((nn*ndim, nn*ndim))
    for w, s in zip(ws, ss):
        if D.ndim == 3: D = jnp.einsum('n, nij -> ij', shapefn(*s), D)
        detJ, dNdr = chainrule(*s, X=X)
        B = strainmat(dNdr)
        Ke += w*detJ*(B.T@D@B)/2
    return Ke.ravel()


compute_stiffness_matrices = vmap(compute_stiffness_matrix, in_axes=(0, 0))


@jit
def compute_helmholtz_matrix(X: ArrayLike) -> Tuple[jnp.DeviceArray]:
    nn, _ = X.shape
    w, s = GQ3[1] if nn == 3 else GQ4[1]
    N = shapefn(s[0])
    detJ, B = chainrule(s[0], X)
    BB = w[0]*detJ*(B.T@B)/2
    NN = w[0]*detJ*jnp.einsum('i, j -> ij', N, N)
    return BB.ravel(), NN.ravel()


@jit
def compute_traction_vector(X: ArrayLike, t: Union[ArrayLike, Callable]) -> jnp.DeviceArray:
    nn, ndim = X.shape
    assert nn == ndim
    I = jnp.eye(ndim)
    tt = np.zeros(nn*ndim)
    if ndim == 2:
        w = 1/2
        for s in [0.0, 1.0]:
            N = shapefn(s)
            r = N@X
            tr = t(r) if isinstance(t, Callable) else t
            NN = jnp.stack([jnp.kron(N, i) for i in jnp.eye(ndim)])
            ds = jnp.linalg.norm(gradfn(s)@X)
            tt += w*NN.T@tr*ds
    elif ndim == 3:
        for w, s in zip(*GQ3[1]):
            N = shapefn(*s)
            r = N@X
            tr = t(r) if isinstance(t, Callable) else t
            NN = np.stack([jnp.kron(N, i) for i in jnp.eye(ndim)])
            dA = np.linalg.norm(np.cross(*gradfn(*s)@X))
            tt += w*NN.T@tr*dA
    return tt


compute_traction_vectors = vmap(compute_traction_vector, in_axes=(0, None))


from scipy.sparse import coo_matrix, bmat, linalg
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt

class FEM:
    def __init__(self, mesh: Mesh, t: ArrayLike):

        self.ndim = mesh.ndim
        self.mesh = mesh
        self.T = Triangulation(*mesh.coords.T, triangles=mesh.elems)

        # Build load vector
        fdofs = mesh.ndim*np.repeat(mesh.traction, mesh.ndim, axis=1) + np.tile(np.arange(mesh.ndim), mesh.ndim)
        sF = compute_traction_vectors(mesh.coords[mesh.traction], t).ravel()
        self.f = coo_matrix((sF, (fdofs.ravel(), np.zeros(len(sF), dtype=int))), shape=(mesh.num_dofs, 1))

        # Build constraint matrix
        self.assemble_bc_matrix()

    def assemble_bc_matrix(self):
        num_fixed_dofs = len(self.mesh.fixed_dofs)
        self.C = coo_matrix((np.ones(num_fixed_dofs), (np.arange(num_fixed_dofs), self.mesh.fixed_dofs)), shape=(num_fixed_dofs, self.mesh.num_dofs))
        self.b = np.zeros((num_fixed_dofs, 1))

    def assemble_system_matrices(self, Ds):
        Xs = self.mesh.coords[self.mesh.elems]
        sK = compute_stiffness_matrices(Ds, Xs)
        K = coo_matrix((sK.ravel(), (self.mesh.iK, self.mesh.jK)), shape=(self.mesh.num_dofs, self.mesh.num_dofs))
        return sK, K

    def solve(self, K):
        KC = bmat([[K, self.C.T], [self.C, None]]).tocsr()
        fb = bmat([[self.f], [self.b]]).tocsr()
        u, lam = np.split(linalg.spsolve(KC, fb), [self.mesh.num_dofs])
        return u, lam

    def eval_element_strain(self, u):
        ue = u[self.mesh.edofs]
        xe = self.mesh.coords[self.mesh.elems]
        ee = vmap(lambda x, u: strainmat(chainrule(1/3, 1/3, X=x)[1])@u, in_axes=(0, 0))(xe, ue)
        return ee

    def eval_element_stress(self, Ds, u):
        """ Modification required for nodal TO """
        ue = u[self.mesh.edofs]
        xe = self.mesh.coords[self.mesh.elems]
        se = vmap(lambda D, x, u: jnp.asarray(D)@(strainmat(chainrule(1/3, 1/3, X=x)[1])@u), in_axes=(0, 0, 0))(Ds, xe, ue)
        return se

    def plot_nodes(self, val, ax = None, axis = 'on'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.tricontourf(self.T, val)
        ax.set_aspect('equal')
        ax.axis(axis)
        plt.tight_layout()
        plt.show()

    def plot_elems(self, val, ax = None, axis = 'on'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.tripcolor(self.T, val)
        ax.set_aspect('equal')
        ax.axis(axis)
        plt.tight_layout()
        plt.show()

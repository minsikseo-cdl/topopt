from optparse import Option
from typing import Optional
import numpy as np
import gmsh


def get_mesh_data(model: gmsh.model, N: Optional[int]):
    ndim = model.getDimension()

    # Get node coordinates
    idx, coords, _ = model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :ndim]

    # Get elements
    elements = dict()
    for dim, tag in model.getPhysicalGroups():
        key = model.getPhysicalName(dim, tag)
        entities = model.getEntitiesForPhysicalGroup(dim, tag)
        elementNodeTags = np.concatenate(
            [model.mesh.getElements(dim, t)[-1][0].reshape(-1, dim+1).astype(int) - 1 for t in entities])
        elements[key] = (dim, elementNodeTags)
    if 'nondesign' not in elements:
        elements['nondesign'] = (ndim, np.zeros((0, ndim+1), dtype=int))
    domain = np.concatenate([elements['design'][1], elements['nondesign'][1]])
    max_node_idx = domain.ravel().max()
    coords = coords[idx-1 <= max_node_idx]

    # Partition mesh
    if N:
        numElements = len(gmsh.model.mesh.getElements(2)[1][0])
        numPartitions = numElements // N
        gmsh.model.mesh.partition(numPartitions)

        partitions = []
        for _, tag in model.getEntities(2):
            _, elementTags, elementNodeTags = model.mesh.getElements(2, tag)
            if len(elementTags):
                nodeTags = []
                for _, tag2 in model.getBoundary([(2, tag)], oriented=False):
                    _, _, nodeTags_ = model.mesh.getElements(1, tag2)
                    nodeTags.append(nodeTags_[0].reshape(-1, 2).astype(int) - 1)
                partitions.append(dict(
                    domainNodes=np.unique(elementNodeTags[0].ravel().astype(int)) - 1,
                    domainElements=elementNodeTags[0].reshape(-1, 3).astype(int) - 1,
                    boundaryElements=np.concatenate(nodeTags, axis=0)
                ))
    else:
        partitions = None

    return coords, elements, partitions


def clever2d(L: float = 2, H: float = 1, alpha: float = 0.1, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('clever2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 3)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 4)
    gmsh.model.geo.addPoint(L, (H - alpha)/2, 0, hmax, 5) # Points for traction boundary
    gmsh.model.geo.addPoint(L, (H + alpha)/2, 0, hmax, 6)

    # Add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 5, 2)
    gmsh.model.geo.addLine(5, 6, 3)
    gmsh.model.geo.addLine(6, 3, 4)
    gmsh.model.geo.addLine(3, 4, 5)
    gmsh.model.geo.addLine(4, 1, 6)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [6], 2, name="spc01")
    gmsh.model.addPhysicalGroup(1, [3], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [1, 2, 4, 5], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Cantilever 2d')


def simplebeam2d(L: float = 3, H: float = 1, alpha: float = 0.1, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('simplebeam2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(L - alpha, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 3)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 4)
    gmsh.model.geo.addPoint(alpha, H, 0, hmax, 5)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 6)

    # Add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 1, 6)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([6, 1, 2, 3, 4, 5], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [6], 2, name="spc0")
    gmsh.model.addPhysicalGroup(1, [2], 3, name="spc1")
    gmsh.model.addPhysicalGroup(1, [5], 4, name="traction")
    gmsh.model.addPhysicalGroup(1, [1, 3, 4], 5, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Simply Supported Beam 2d')


def bridge2d(L: float = 2, H: float = 1, alpha: float = 0.1, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('bridge2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(alpha, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint((L - alpha)/2, 0, 0, hmax, 3)
    gmsh.model.geo.addPoint((L + alpha)/2, 0, 0, hmax, 4)
    gmsh.model.geo.addPoint(L - alpha, 0, 0, hmax, 5)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 6)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 7)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 8)

    # Add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 1, 8)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [1, 5], 2, name="spc01")
    gmsh.model.addPhysicalGroup(1, [3], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [2, 4, 6, 7, 8], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Bridge 2d')


def michell2d(L: float = 2, H: float = 1.5, R: float = 0.5, alpha: float = 0.1, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('michell2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(0, R, 0, hmax, 2)
    gmsh.model.geo.addPoint(0, -R, 0, hmax, 3)
    gmsh.model.geo.addPoint(0, -H/2, 0, hmax, 4)
    gmsh.model.geo.addPoint(L, -H/2, 0, hmax, 5)
    gmsh.model.geo.addPoint(L, -alpha/2, 0, hmax, 6)
    gmsh.model.geo.addPoint(L, alpha/2, 0, hmax, 7)
    gmsh.model.geo.addPoint(L, H/2, 0, hmax, 8)
    gmsh.model.geo.addPoint(0, H/2, 0, hmax, 9)

    # Add lines
    gmsh.model.geo.addCircleArc(3, 1, 2, 1)
    gmsh.model.geo.addLine(2, 9, 2)
    gmsh.model.geo.addLine(9, 8, 3)
    gmsh.model.geo.addLine(8, 7, 4)
    gmsh.model.geo.addLine(7, 6, 5)
    gmsh.model.geo.addLine(6, 5, 6)
    gmsh.model.geo.addLine(5, 4, 7)
    gmsh.model.geo.addLine(4, 3, 8)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [1], 2, name="spc01")
    gmsh.model.addPhysicalGroup(1, [5], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [2, 3, 4, 6, 7, 8], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Michell 2d')


def halfcircle2d(R: float = 1, alpha: float = 0.1, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('halfcircle2d')

    # Add points
    gmsh.model.geo.addPoint(-R, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(-R + alpha, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(-alpha/2, 0, 0, hmax, 3)
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 4)
    gmsh.model.geo.addPoint(alpha/2, 0, 0, hmax, 5)
    gmsh.model.geo.addPoint(R - alpha, 0, 0, hmax, 6)
    gmsh.model.geo.addPoint(R, 0, 0, hmax, 7)

    # Add lines
    gmsh.model.geo.addCircleArc(7, 4, 1, 1)
    gmsh.model.geo.addLine(1, 2, 2)
    gmsh.model.geo.addLine(2, 3, 3)
    gmsh.model.geo.addLine(3, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [2, 6], 2, name="spc01")
    gmsh.model.addPhysicalGroup(1, [4], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [1, 3, 5], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Half Circle 2d')


def flower2d(R: float = 1, r: float = 0.25, alpha: float = 0.1, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('flower2d')

    # Add points
    dq = alpha/R
    gmsh.model.geo.addPoint(0, 0, 0, hmax)
    gmsh.model.geo.addPoint(0, r, 0, hmax)
    gmsh.model.geo.addPoint(0, -r, 0, hmax)
    for q in np.linspace(0.0, 2*np.pi, 6)[:-1]:
        gmsh.model.geo.addPoint(R*np.cos(q-dq), R*np.sin(q-dq), 0, hmax)
        gmsh.model.geo.addPoint(R*np.cos(q+dq), R*np.sin(q+dq), 0, hmax)
    
    # Add lines
    gmsh.model.geo.addCircleArc(2, 1, 3)
    gmsh.model.geo.addCircleArc(3, 1, 2)
    idx = np.r_[np.arange(4, 14), 4]
    for i in range(len(idx)-1):
        ij = idx[i:i+2]
        gmsh.model.geo.addCircleArc(ij[0], 1, ij[1])

    # Add surfaces
    gmsh.model.geo.addCurveLoop([4, 5, 6, 7, 8, 9, 10, 11, 12, 3], 1)
    gmsh.model.geo.addCurveLoop([1, 2], 2)
    gmsh.model.geo.addPlaneSurface([1, 2], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [1, 2], 2, name="spc01")
    gmsh.model.addPhysicalGroup(1, [3, 5, 7, 9, 11], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [4, 6, 8, 10, 12], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Flower 2d')


def wrench2d(L: float = 2, R1: float = 0.5, R2: float = 0.3, r1: float = 0.3, r2: float = 0.175, hmax: float = 0.1, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('wrench2d')

    # Add points
    x1 = R1*(R1 - R2)/L
    x2 = (L**2 - R2**2 + R1*R2)/L
    y1 = R1*np.sqrt((L + R1 - R2)*(L - R1 + R2))/L
    y2 = R2*np.sqrt((L + R1 - R2)*(L - R1 + R2))/L
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(r1, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(-r1, 0, 0, hmax, 3)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 4)
    gmsh.model.geo.addPoint(L + r2, 0, 0, hmax, 5)
    gmsh.model.geo.addPoint(L - r2, 0, 0, hmax, 6)
    gmsh.model.geo.addPoint(x1, -y1, 0, hmax, 7)
    gmsh.model.geo.addPoint(x2, -y2, 0, hmax, 8)
    gmsh.model.geo.addPoint(x2, y2, 0, hmax, 9)
    gmsh.model.geo.addPoint(x1, y1, 0, hmax, 10)
    gmsh.model.geo.addPoint(-R1, 0, 0, hmax, 11)
    
    
    # Add lines
    gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    gmsh.model.geo.addCircleArc(3, 1, 2, 2)
    gmsh.model.geo.addCircleArc(5, 4, 6, 3)
    gmsh.model.geo.addCircleArc(6, 4, 5, 4)
    gmsh.model.geo.addLine(7, 8, 5)
    gmsh.model.geo.addCircleArc(8, 4, 9, 6)
    gmsh.model.geo.addLine(9, 10, 7)
    gmsh.model.geo.addCircleArc(10, 1, 11, 8) 
    gmsh.model.geo.addCircleArc(11, 1, 7, 9) 

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2], 1)
    gmsh.model.geo.addCurveLoop([3, 4], 2)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8, 9], 3)
    gmsh.model.geo.addPlaneSurface([3, 1, 2], 1)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(2, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(1, [1, 2], 2, name="spc01")
    gmsh.model.addPhysicalGroup(1, [4], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [3, 5, 6, 7, 8, 9], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='Wrench 2d')


def clever3d(L: float, H: float, W: float, alpha: float, hmax: float, N: Optional[int] = None):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('clever3d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 2)
    gmsh.model.geo.addPoint(0, H, W, hmax, 3)
    gmsh.model.geo.addPoint(0, 0, W, hmax, 4)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 5)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 6)
    gmsh.model.geo.addPoint(L, H, W, hmax, 7)
    gmsh.model.geo.addPoint(L, 0, W, hmax, 8)
    gmsh.model.geo.addPoint(L, (0.5 - alpha/2)*H, (0.5 - alpha/2)*W, hmax, 9)
    gmsh.model.geo.addPoint(L, (0.5 + alpha/2)*H, (0.5 - alpha/2)*W, hmax, 10)
    gmsh.model.geo.addPoint(L, (0.5 + alpha/2)*H, (0.5 + alpha/2)*W, hmax, 11)
    gmsh.model.geo.addPoint(L, (0.5 - alpha/2)*H, (0.5 + alpha/2)*W, hmax, 12)

    # Add lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)
    gmsh.model.geo.addLine(9, 10, 9)
    gmsh.model.geo.addLine(10, 11, 10)
    gmsh.model.geo.addLine(11, 12, 11)
    gmsh.model.geo.addLine(12, 9, 12)
    gmsh.model.geo.addLine(5, 1, 13)
    gmsh.model.geo.addLine(6, 2, 14)
    gmsh.model.geo.addLine(7, 3, 15)
    gmsh.model.geo.addLine(8, 4, 16)

    # Add surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addCurveLoop([7, 8, 5, 6], 2)
    gmsh.model.geo.addCurveLoop([12, 9, 10, 11], 3)
    gmsh.model.geo.addPlaneSurface([2, 3], 2)
    gmsh.model.geo.addPlaneSurface([3], 3)
    gmsh.model.geo.addCurveLoop([13, 1, -14, -5], 4)
    gmsh.model.geo.addPlaneSurface([4], 4)
    gmsh.model.geo.addCurveLoop([6, 15, -2, -14], 5)
    gmsh.model.geo.addPlaneSurface([5], 5)
    gmsh.model.geo.addCurveLoop([7, 16, -3, -15], 6)
    gmsh.model.geo.addPlaneSurface([6], 6)
    gmsh.model.geo.addCurveLoop([8, 13, -4, -16], 7)
    gmsh.model.geo.addPlaneSurface([7], 7)

    # Add volume
    gmsh.model.geo.addSurfaceLoop([1, 4, 7, 2, 6, 5, 3], 1)
    gmsh.model.geo.addVolume([1], 1)

    # Synchronize
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()

    # Define physical groups
    gmsh.model.addPhysicalGroup(3, [1], 1, name='design')
    gmsh.model.addPhysicalGroup(2, [1], 2, name="spc01")
    gmsh.model.addPhysicalGroup(2, [3], 3, name="traction")
    gmsh.model.addPhysicalGroup(2, [2, 4, 5, 6, 7], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(3)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)
    gmsh.write('/workspace/tmp.msh')

    gmsh.finalize()


    return Mesh(*mesh_data, name='clever3d')


class Mesh:
    ELEM_TYPES = ['line', 'tria', 'tetra']
    PROB_TYPES = ['structure', 'thermal']

    def __init__(self,
        coords: np.ndarray, elements: dict, partitions: Optional[dict] = None,
        name: str = 'noname', prob_type: str = 'structure'):
        assert prob_type in self.PROB_TYPES

        self.name = name
        self.ndim = coords.shape[1]
        self.prob_type = prob_type

        # Nodes
        self.coords = coords
        self.num_nodes = coords.shape[0]
        self.num_dofs = self.ndim*self.num_nodes if prob_type == 'structure' else self.num_nodes

        # Elements
        self.elems = np.concatenate([elements['design'][1], elements['nondesign'][1]])
        self.num_elems = self.elems.shape[0]
        self.num_design_elems = elements['design'][1].shape[0]
        self.num_nondesign_elems = elements['nondesign'][1].shape[0]
        self.passive = {
            'elements': np.arange(self.num_design_elems, self.num_elems),
            'nodes': np.unique(elements['nondesign'][1])
        }

        # System matrix indices
        self.get_matrix_indices()

        # Boundary conditions
        self.get_bc_indices(elements)

        self.parts = partitions if partitions is not None else []
        self.num_parts = len(self.parts)

    def get_matrix_indices(self):
        if self.prob_type == 'structure':
            self.edofs = self.ndim*np.repeat(self.elems, self.ndim, axis=1) + np.tile(np.arange(self.ndim), self.elems.shape[1])
            self.iK = np.repeat(self.edofs, self.ndim*self.elems.shape[1], axis=1).ravel()
            self.jK = np.tile(self.edofs, (1, self.ndim*self.elems.shape[1])).ravel()
        else:
            self.edofs = self.elems
            self.iK = np.repeat(self.edofs, self.elems.shape[1], axis=1)
            self.jK = np.tile(self.edofs, (1, self.elems.shape[1]))

    def get_bc_indices(self, elements):
        if self.prob_type == 'structure':
            self.traction = elements['traction'][1]
            spc = []
            for key, (_, elem) in elements.items():
                if 'spc' in key:
                    dofs = key[3:]
                    nodes = np.unique(elem)
                    for dof in dofs:
                        spc += [self.ndim*nodes + int(dof)]
            self.fixed_dofs = np.concatenate(spc)
        else:
            pass

    def __repr__(self) -> str:
        desc  = f"A {self.ndim}-dimensional {self.name} mesh with\n"
        desc +=  "===============================================\n"
        desc += f"{self.num_nodes}\tnodes\n"
        desc += f"{self.num_elems}\tdomain elements\n"
        desc += f"{self.num_parts}\tpartitions\n"
        return desc


if __name__ == '__main__':
    mesh = wrench2d()
    print(mesh)
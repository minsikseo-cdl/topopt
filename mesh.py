from typing import Optional
import numpy as np
import gmsh


def get_mesh_data(model: gmsh.model, N: Optional[int]):
    ndim = gmsh.model.getDimension()

    # Get node coordinates
    coords = model.mesh.getNodes()[1].reshape(-1, 3)[:, :ndim]

    # Get elements
    elements = dict()
    for dim, tag in model.getPhysicalGroups():
        key = model.getPhysicalName(dim, tag)
        _, _, elementNodeTags = model.mesh.getElements(dim, tag)
        elements[key] = (dim, elementNodeTags[0].reshape(-1, dim+1).astype(int) - 1)

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


def clever2d(L: float, H: float, alpha: float, hmax: float, N: Optional[int] = None):
    """
    L: longitudinal length 
    H: Height
    alpha: fractional length of traction boundary w.r.t. height
    hmax: Maximum mesh size
    N: Expected number of domain elements per partition
    """
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add('clever2d')

    # Add points
    gmsh.model.geo.addPoint(0, 0, 0, hmax, 1)
    gmsh.model.geo.addPoint(L, 0, 0, hmax, 2)
    gmsh.model.geo.addPoint(L, H, 0, hmax, 3)
    gmsh.model.geo.addPoint(0, H, 0, hmax, 4)
    gmsh.model.geo.addPoint(L, (0.5 - alpha/2)*H, 0, hmax, 5) # Points for traction boundary
    gmsh.model.geo.addPoint(L, (0.5 + alpha/2)*H, 0, hmax, 6)

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
    gmsh.model.addPhysicalGroup(2, [1], 1, name='domain')
    gmsh.model.addPhysicalGroup(1, [6], 2, name="fixed")
    gmsh.model.addPhysicalGroup(1, [3], 3, name="traction")
    gmsh.model.addPhysicalGroup(1, [1, 2, 4, 5], 4, name="free")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Get mesh data
    mesh_data = get_mesh_data(gmsh.model, N)

    gmsh.finalize()

    return Mesh(*mesh_data, name='clever2d')


class Mesh:
    def __init__(
        self, coords: np.ndarray, elements: dict, partitions: Optional[dict] = None, name: str = 'noname'):
        self.name = name
        self.ndim = coords.shape[1]
        self.coords = coords
        self.elems = elements
        self.parts = partitions if partitions is not None else []

    def get_dofs(self, ncomps: int, group: str) -> np.ndarray:
        ndim, elems = self.elems[group]
        return ncomps*np.repeat(elems, ncomps, axis=1) + np.tile(np.arange(ncomps), ndim+1)

    def num_nodes(self) -> int:
        return len(self.coords)

    def num_elems(self) -> int:
        return len(self.elems['domain'][1])

    def num_parts(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        desc  = f"A {self.ndim}-dimensional {self.name} mesh with\n"
        desc +=  "===============================================\n"
        desc += f"{self.num_nodes()}\tnodes\n"
        desc += f"{self.num_elems()}\telements\n"
        desc += f"{self.num_parts()}\tpartitions\n"
        return desc


if __name__ == '__main__':
    mesh = clever2d(2, 1, 0.1, 0.05, 100)
    print(mesh)
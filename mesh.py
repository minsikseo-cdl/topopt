from typing import Union, Optional, List, Tuple, Dict
import numpy as np

Index = Union[List[int], np.ndarray[int]]
Group = Tuple[int, Index]
Groups = Dict[str, Group]

class Mesh:
    ELEM_TYPES: List[str] = ['line', 'tria', 'tetra']

    def __init__(
        self, nodes: np.ndarray, elems: List[np.ndarray],
        groups: Groups, parts: Optional[List[np.ndarray]] = None, name: str = ''):
        self.name = name
        self.ndim = len(elems)
        self.nodes = nodes[:, :self.ndim]
        self.elems = elems
        self.groups = groups
        self.parts = parts
        if 'design' not in groups:
            self.groups['design'] = (self.ndim, np.arange(len(elems[-1])))

    def num_nodes(self):
        return len(self.nodes)

    def num_elems(self):
        return [len(e) for e in self.elems]

    def get_group(self, group_name: str, return_index: bool = True):
        assert group_name in self.groups, f'There is no group named "{group_name}"'
        ndim, idx = self.groups[group_name]
        if return_index:
            return ndim, idx
        else:
            return self.get_elems(ndim, idx)

    def get_elems(self, ndim: int, idx: Union[int, np.ndarray]):
        return self.elems[ndim-1][idx]

    def get_partitions(self, parts: Optional[Union[int, List[int]]] = None, return_index: bool = True):
        assert self.parts is not None, "Mesh does not have any partition."
        if parts is None:
            parts = np.arange(len(self.parts))
        if isinstance(parts, int):
            parts = [parts]
        return_list = []
        for part in parts:
            out = (self.ndim, part) if return_index else self.elems[self.ndim-1][self.parts[part]]
            return_list += [out]
        return return_list

    def __repr__(self):
        title = f"A {self.ndim}-dimensional {self.name} mesh with\n"
        stats = f"{self.num_nodes()}\tnodes\n"
        for i, e in enumerate(self.num_elems()):
            stats += f"{e}\t{self.ELEM_TYPES[i]} elements\n"
        groups = f"{len(self.groups)}\tgroups ("
        for i, key in enumerate(self.groups):
            groups += f"{key}" if i == 0 else f", {key}"
        groups += ")\n"
        parts = ""
        num_parts = 0 if self.parts is None else len(self.parts)
        parts += f"{num_parts}\tpartitions\n"
        desc = title + stats + groups + parts
        return desc
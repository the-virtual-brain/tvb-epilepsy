import numpy as np

from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict


class Surface(object):
    vertices = np.array([])
    triangles = np.array([])
    vertex_normals = np.array([])
    triangle_normals = np.array([])

    def __init__(self, vertices, triangles, vertex_normals=np.array([]), triangle_normals=np.array([])):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = vertex_normals
        self.triangle_normals = triangle_normals

    def __repr__(self):
        d = {"a. vertices": self.vertices,
             "b. triangles": self.triangles,
             "c. vertex_normals": self.vertex_normals,
             "d. triangle_normals": self.triangle_normals}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()
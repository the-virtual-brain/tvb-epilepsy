# coding=utf-8

import numpy as np
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict


class SurfaceH5Field(object):
    VERTICES = "vertices"
    TRIANGLES = "triangles"
    VERTEX_NORMALS = "vertex_normals"


class Surface(object):
    vertices = np.array([])
    triangles = np.array([])
    vertex_normals = np.array([])
    triangle_normals = np.array([])

    def __init__(self, vertices, triangles, surface_subtype="CORTICAL", vertex_normals=np.array([]),
                 triangle_normals=np.array([])):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = vertex_normals
        self.triangle_normals = triangle_normals
        self.surface_subtype = surface_subtype
        self.vox2ras = np.array([])

    def __repr__(self):
        d = {"00. surface subtype": self.surface_subtype,
             "01. vertices": self.vertices,
             "02. triangles": self.triangles,
             "03. vertex_normals": self.vertex_normals,
             "04. triangle_normals": self.triangle_normals,
             "05. voxel to ras matrix": self.vox2ras}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

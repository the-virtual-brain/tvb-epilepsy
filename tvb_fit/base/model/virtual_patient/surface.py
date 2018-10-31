# coding=utf-8

import numpy as np
from tvb_fit.base.utils.data_structures_utils import formal_repr, sort_dict


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
        self.get_vertex_normals()
        self.get_triangle_normals()
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

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    @property
    def n_triangles(self):
        return self.triangles.shape[0]

    def get_vertex_triangles(self):
        vertex_triangles = [[] for _ in range(self.n_vertices)]
        for k in range(self.n_triangles):
            vertex_triangles[self.triangles[k, 0]].append(k)
            vertex_triangles[self.triangles[k, 1]].append(k)
            vertex_triangles[self.triangles[k, 2]].append(k)
        return vertex_triangles

    def compute_vertex_normals(self):
        # TODO test by generating points on unit sphere: vtx pos should equal
        # normal

        vf = self.vertices[self.triangles]
        fn = np.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0])
        vf = [set() for _ in range(self.n_vertices)]
        for i, fi in enumerate(self.triangles):
            for j in fi:
                vf[j].add(i)
        vn = np.zeros_like(self.vertices)
        for i, fi in enumerate(vf):
            fni = fn[list(fi)]
            norm = fni.sum(axis=0)
            norm2 = norm / np.sqrt((norm ** 2).sum())
            vn[i] = norm2
        return vn

    def get_vertex_normals(self):
        # If there is at least 3 vertices and 1 triangle...
        if self.n_vertices > 2 and self.n_triangles > 0:
            if self.vertex_normals.shape[0] != self.n_vertices:
                self.vertex_normals = self.compute_vertex_normals()
        return self.vertex_normals

    def compute_triangle_normals(self):
        """Calculates triangle normals."""
        tri_u = self.vertices[self.triangles[:, 1], :] - self.vertices[self.triangles[:, 0], :]
        tri_v = self.vertices[self.triangles[:, 2], :] - self.vertices[self.triangles[:, 0], :]
        tri_norm = np.cross(tri_u, tri_v)

        try:
            triangle_normals = tri_norm / np.sqrt(np.sum(tri_norm ** 2, axis=1))[:, np.newaxis]
        except FloatingPointError:
            # TODO: NaN generation would stop execution, however for normals this case could maybe be
            #  handled in a better way.
            triangle_normals = tri_norm
        return triangle_normals

    def get_triangle_normals(self):
        # If there is at least 3 vertices and 1 triangle...
        if self.n_vertices > 2 and self.n_triangles > 0:
            if self.triangle_normals.shape[0] != self.n_triangles:
                self.triangle_normals = self.compute_triangle_normals()
        return self.triangle_normals

    def get_triangle_areas(self):
        """Calculates the area of triangles making up a surface."""
        tri_u = self.vertices[self.triangles[:, 1], :] - self.vertices[self.triangles[:, 0], :]
        tri_v = self.vertices[self.triangles[:, 2], :] - self.vertices[self.triangles[:, 0], :]
        tri_norm = np.cross(tri_u, tri_v)
        triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
        triangle_areas = triangle_areas[:, np.newaxis]
        return triangle_areas

    def get_vertex_areas(self):
        triangle_areas = self.get_triangle_areas()
        vertex_areas = np.zeros((self.n_vertices,))
        for triang, vertices in enumerate(self.triangles):
            for i in range(3):
                vertex_areas[vertices[i]] += 1. / 3. * triangle_areas[triang]
        return vertex_areas

    def add_vertices_and_triangles(self, new_vertices, new_triangles,
                                   new_vertex_normals=np.array([]),  new_triangle_normals=np.array([])):
        self.triangles = np.array(self.triangles.tolist() + (new_triangles + self.n_vertices).tolist())
        self.vertices = np.array(self.vertices.tolist() + new_vertices.tolist())
        self.vertex_normals = np.array(self.vertex_normals.tolist() + new_vertex_normals.tolist())
        self.triangle_normals = np.array(self.triangle_normals.tolist() + new_triangle_normals.tolist())
        self.get_vertex_normals()
        self.get_triangle_normals()

    def compute_surface_area(self):
        """
            This function computes the surface area
            :param: surface: input surface object
            :return: (sub)surface area, float
            """
        return np.sum(self.get_triangle_areas())


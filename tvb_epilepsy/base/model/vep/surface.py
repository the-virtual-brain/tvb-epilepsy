
import numpy as np

from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, construct_import_path
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class Surface(object):
    vertices = np.array([])
    triangles = np.array([])
    vertex_normals = np.array([])
    triangle_normals = np.array([])

    def __init__(self, vertices, triangles, surface_subtype="CORTICAL", vertex_normals=np.array([]), triangle_normals=np.array([])):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_normals = vertex_normals
        self.triangle_normals = triangle_normals
        self.surface_subtype = surface_subtype
        self.vox2ras = np.array([])
        self.context_str = "from " + construct_import_path(__file__) + " import Surface"
        self.create_str = "Surface(np.array([]), np.array([]))"

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

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "Surface")
        h5_model.add_or_update_metadata_attribute("Surface_subtype", self.surface_subtype)
        h5_model.add_or_update_metadata_attribute("Number_of_triangles", self.triangles.shape[0])
        h5_model.add_or_update_metadata_attribute("Number_of_vertices", self.vertices.shape[0])
        h5_model.add_or_update_metadata_attribute("Voxel_to_ras_matrix",
                                                  str(self.vox2ras.flatten().tolist())[1:-1].replace(",", ""))
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

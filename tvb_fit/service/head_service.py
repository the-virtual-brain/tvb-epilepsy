import numpy as np

from tvb.datatypes.connectivity import Connectivity as TVB_Connectivity
from tvb_fit.base.constants import SEEG_DIPOLE_SIGMA
from tvb_fit.base.model.virtual_patient.connectivity import Connectivity
from tvb_fit.base.model.virtual_patient.surface import Surface
from tvb_fit.base.model.virtual_patient.sensors import Sensors
from tvb_fit.base.utils.data_structures_utils import ensure_list
from tvb_fit.base.utils.log_error_utils import warning, raise_value_error, initialize_logger
from tvb_fit.base.computations.math_utils import select_greater_values_array_inds, \
    select_by_hierarchical_group_metric_clustering


SIGMA = SEEG_DIPOLE_SIGMA


class HeadService(object):
    logger = initialize_logger(__name__)

    def compute_nearest_regions_to_sensors(self, head, sensors=None, target_contacts=None, s_type=Sensors.TYPE_SEEG,
                                           sensors_id=0, n_regions=None, gain_matrix_th=None,
                                           gain_matrix_percentile=None):
        if not (isinstance(sensors, Sensors)):
            sensors = head.get_sensors_id(s_type=s_type, sensor_ids=sensors_id)
        n_contacts = sensors.labels.shape[0]
        if isinstance(target_contacts, (list, tuple, np.ndarray)):
            target_contacts = ensure_list(target_contacts)
            for itc, tc in enumerate(target_contacts):
                if isinstance(tc, int):
                    continue
                elif isinstance(tc, basestring):
                    target_contacts[itc] = sensors.contact_label_to_index([tc])
                else:
                    raise_value_error("target_contacts[" + str(itc) + "] = " + str(tc) +
                                      "is neither an integer nor a string!")
        else:
            target_contacts = range(n_contacts)
        if n_regions is "all":
            n_regions = head.connectivity.number_of_regions
        nearest_regions = []
        for tc in target_contacts:
            projs = np.abs(sensors.gain_matrix[tc])
            inds = np.argsort(projs)[::-1]
            n_regions = select_greater_values_array_inds(projs[inds], threshold=gain_matrix_th,
                                                         percentile=gain_matrix_percentile, n_regions=n_regions)
            inds = inds[:n_regions]
            nearest_regions.append((inds, head.connectivity.region_labels[inds], projs[inds]))
        return nearest_regions

    def select_sensors_power(self, sensors, power, selection=[], power_th=None, power_percentile=None, n_sensors=None):
        if len(selection) == 0:
            selection = range(sensors.number_of_sensors)
        return (np.array(selection)[select_greater_values_array_inds(power, power_th, power_percentile,
                                                                     n_sensors)]).tolist()

    def select_sensors_rois(self, sensors, rois=None, initial_selection=[], gain_matrix_th=0.5,
                            gain_matrix_percentile=None, n_rois=None):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        selection = []
        if sensors.gain_matrix is None:
            raise_value_error("Projection matrix is not set!")
        else:
            for proj in np.abs(sensors.gain_matrix[initial_selection].T[rois]):
                selection += (np.array(initial_selection)
                [select_greater_values_array_inds(proj, gain_matrix_th, gain_matrix_percentile, n_rois)]).tolist()
        return np.unique(selection).tolist()

    def sensors_in_electrodes_disconnectivity(self, sensors, sensors_labels=[]):
        if len(sensors_labels) < 2:
            sensors_labels = sensors.labels
        n_sensors = len(sensors_labels)
        elec_labels, elec_inds = sensors.group_sensors_to_electrodes(sensors_labels)
        if len(elec_labels) >= 2:
            disconnectivity = np.ones((n_sensors, n_sensors))
            for ch in elec_inds:
                disconnectivity[np.meshgrid(ch, ch)] = 0.0
        return disconnectivity

    def select_sensors_corr(self, sensors, distance, initial_selection=[], n_electrodes=10, sensors_per_electrode=1,
                            power=None, group_electrodes=False):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        n_sensors = len(initial_selection)
        if n_sensors > 2:
            initial_selection = np.array(initial_selection)
            distance = 1.0 - distance
            if group_electrodes:
                disconnectivity = self.sensors_in_electrodes_disconnectivity(sensors, sensors.labels[initial_selection])
            selection = \
                select_by_hierarchical_group_metric_clustering(distance, disconnectivity, metric=power,
                                                               n_groups=n_electrodes,
                                                               members_per_group=sensors_per_electrode)
            return np.unique(np.hstack(initial_selection[selection])).tolist()
        else:
            self.logger.warning("Number of sensors' left < 2!\n" + "Skipping clustering and returning all of them!")
            return initial_selection

    def vp2tvb_connectivity(self, vp_conn, model_connectivity=None, time_delay_flag=1):
        if model_connectivity is None:
            model_connectivity = vp_conn.normalized_weights
        return TVB_Connectivity(use_storage=False, weights=model_connectivity,
                                tract_lengths=time_delay_flag * vp_conn.tract_lengths,
                                region_labels=vp_conn.region_labels, centres=vp_conn.centres,
                                hemispheres=vp_conn.hemispheres, orientations=vp_conn.orientations, areas=vp_conn.areas)

    def tvb2vp_connectivity(self, tvb_conn, model_connectivity=None, time_delay_flag=1):
        if model_connectivity is None:
            model_connectivity = tvb_conn.weights
        return Connectivity("", model_connectivity, time_delay_flag * tvb_conn.tract_lengths,
                            labels=tvb_conn.region_labels, centres=tvb_conn.centres, hemispheres=tvb_conn.hemispheres,
                            orientations=tvb_conn.orientations, areas=tvb_conn.areas, normalized_weights=np.array([]))

    # The following use tvb_make/util/gain_matrix_seeg.py and the respective TVB code

    def merge_surfaces(self, surfaces,  surface_subtype="", vox2ras=None):
        """
        Merge several surfaces, and their region mappings.
        :return: the merge result surface and region mapping.
        """
        n_surfaces = len(surfaces)
        out_surface = Surface(np.array([]), np.array([]))
        out_surface.surface_subtype = surface_subtype
        if vox2ras is None:
            out_surface.vox2ras = surfaces[0].vox2ras
        else:
            out_surface.vox2ras = vox2ras
        for i_srf in range(n_surfaces):
            out_surface.add_vertices_and_triangles(surfaces[i_srf].vertices, surfaces[i_srf].triangles,
                                                   surfaces[i_srf].vertex_normals, surfaces[i_srf].triangle_normals)
            if np.any(surfaces[i_srf].vox2ras != out_surface.vox2ras):
                raise_value_error("Surface %s has a different vox2ras: %s \n than the one of the merged surface!: %s"
                                  % (str(i_srf), str(surfaces[i_srf].vox2ras), str(out_surface.vox2ras)))
        return out_surface

    def _gain_matrix_dipole(self, vertices, orientations, verts_areas, sensors):
        """
        Parameters
        ----------
        vertices             np.ndarray of floats of size n x 3, where n is the number of dipoles
        orientations         np.ndarray of floats of size n x 3
        verts_areas          np.ndarray of floats of size n x 1
        sensors              np.ndarray of floats of size m x 3, where m is the number of sensors
        Returns
        -------
        np.ndarray of size m x n
        """

        nverts = vertices.shape[0]
        nsens = sensors.shape[0]

        # For EEG from TVB:
        # center = np.mean(vertices, axis=0)[np.newaxis,]
        # radius = 1.05125 * max(np.sqrt(np.sum((vertices - center) ** 2, axis=1)))
        # sen_dis = np.sqrt(np.sum((sensors) ** 2, axis=1))
        # sensors = sensors / sen_dis[:, np.newaxis] * radius + center

        dipole_gain = np.zeros((nsens, nverts)).astype("f")
        for sens_ind in range(nsens):
            a = sensors[sens_ind, :] - vertices
            na = np.sqrt(np.sum(a ** 2, axis=1))
            dipole_gain[sens_ind, :] = (np.sum(orientations * a, axis=1) / na ** 3) / (4.0 * np.pi * SIGMA)

        return verts_areas * dipole_gain

    def _gain_matrix_distance(self, vertices, verts_areas, sensors):
        """
        Parameters
        ----------
        vertices            np.ndarray of floats of size n x 3, where n is the number of dipoles
        orientations         np.ndarray of floats of size n x 3
        verts_areas          np.ndarray of floats of size n x 1
        sensors              np.ndarray of floats of size m x 3, where m is the number of sensors
        Returns
        -------
        np.ndarray of size m x n
        """
        nverts = vertices.shape[0]
        nsens = sensors.shape[0]

        gain_mtx_vert = np.zeros((nsens, nverts))
        for sens_ind in range(nsens):
            a = sensors[sens_ind, :] - vertices
            na = np.sqrt(np.sum(a ** 2, axis=1))
            gain_mtx_vert[sens_ind, :] = verts_areas / na ** 2

        return gain_mtx_vert

    def _get_verts_regions_matrix(self, nvertices, nregions, region_mapping):
        reg_map_mtx = np.zeros((nvertices, nregions), dtype=int)
        for i, region in enumerate(region_mapping):
            if region >= 0:
                reg_map_mtx[i, region] = 1

        return reg_map_mtx

    def _normalize_gain_matrix(self, gain_matrix, normalize=100.0):
        if normalize:
            gain_matrix /= np.percentile(np.abs(gain_matrix), normalize)
        return gain_matrix

    def _compute_vertex_areas_and_verts2regions_mat(self, cort_surf, subcort_surf, cort_rm, subcort_rm):
        cort_rm = ensure_list(cort_rm)
        subcort_rm = ensure_list(subcort_rm)

        region_list = np.unique(cort_rm + subcort_rm)

        nr_regions = len(region_list)
        nr_vertices = cort_surf.vertices.shape[0] + subcort_surf.vertices.shape[0]

        verts_regions_mat = self._get_verts_regions_matrix(nr_vertices, nr_regions, cort_rm + subcort_rm)

        # Weight each vertex with the 1/3 of the areas of all triangles it is involved with
        # This is to account for inhomogeneous spacing of vertices on the surface
        cort_verts_areas = cort_surf.get_vertex_areas()
        subcort_verts_areas = subcort_surf.get_vertex_areas()

        return cort_verts_areas, subcort_verts_areas, verts_regions_mat

    def _compute_areas_for_regions(self, regions, surface, region_mapping):
        """Compute the areas of given regions"""

        region_surface_area = np.zeros(len(regions))
        avt = np.array(surface.get_vertex_triangles())
        # NOTE: Slightly overestimates as it counts overlapping border triangles,
        #       but, not really a problem provided triangle-size << region-size.
        for i, k in enumerate(regions):
            regs = list(map(set, avt[np.array(region_mapping) == k]))
            if len(regs) == 0:
                continue
            region_triangles = set.union(*regs)
            if region_triangles:
                region_surface_area[i] = surface.get_triangle_areas()[list(region_triangles)].sum()

        return region_surface_area

    def _compute_seeg_dipole_gain_matrix(self, sensor_locations, cort_surf, subcort_surf, cort_rm, subcort_rm,
                                         normalize=100.0):

        cort_normals = cort_surf.get_vertex_normals()

        cort_verts_areas, subcort_verts_areas, verts_regions_mat \
            = self._compute_vertex_areas_and_verts2regions_mat(cort_surf, subcort_surf, cort_rm, subcort_rm)

        gain_matrix_cort = self._gain_matrix_dipole(cort_surf.vertices, cort_normals, cort_verts_areas, sensor_locations)

        gain_matrix_subcort = self._gain_matrix_distance(subcort_surf.vertices, subcort_verts_areas, sensor_locations)

        gain_total = np.concatenate((gain_matrix_cort, gain_matrix_subcort), axis=1)

        gain_out = np.matmul(gain_total, verts_regions_mat)
        gain_out = self._normalize_gain_matrix(gain_out, normalize)

        return gain_out

    def _compute_seeg_distance_gain_matrix(self, sensor_locations, cort_surf, subcort_surf, cort_rm, subcort_rm,
                                           normalize=100.0):

        cort_verts_areas, subcort_verts_areas, verts_regions_mat \
            = self._compute_vertex_areas_and_verts2regions_mat(cort_surf, subcort_surf, cort_rm, subcort_rm)

        gain_matrix_cort = self._gain_matrix_distance(cort_surf.vertices, cort_verts_areas, sensor_locations)

        gain_matrix_subcort = self._gain_matrix_distance(subcort_surf.vertices, subcort_verts_areas, sensor_locations)

        gain_total = np.concatenate((gain_matrix_cort, gain_matrix_subcort), axis=1)

        gain_out = np.matmul(gain_total, verts_regions_mat)
        gain_out = self._normalize_gain_matrix(gain_out, normalize)

        return gain_out

    def _compute_seeg_regions_distance_gain_matrix(self, sensor_locations, centers, areas=None,
                                                   cort_surf=None, subcort_surf=None, cort_rm=None, subcort_rm=None,
                                                   normalize=100.0):
        if areas is None:
            if cort_surf is None or subcort_surf is None or cort_rm is None or subcort_rm is None:
                warning("Regions' areas are not in the input and cannot be computed "
                        "because cortical and/or surfaces area also missing!"
                        "\nComputing regions' distances' seeg gain matrix assuming "
                        "equal areas weighting to 1 for all regions!")
                areas = np.ones(centers.shape)
            else:
                #cort_verts_areas, subcort_verts_areas, verts_regions_mat \
                #     = self._compute_vertex_areas_and_verts2regions_mat(cort_surf, subcort_surf, cort_rm, subcort_rm)
                # verts_areas = np.concatenate((cort_verts_areas, subcort_verts_areas), axis=0)
                #  areas = np.matmul(verts_areas[:, np.newaxis].T, verts_regions_mat).squeeze()
                surface = self.merge_surfaces([cort_surf, subcort_surf],  surface_subtype="BRAIN")
                region_mapping = ensure_list(cort_rm) + ensure_list(subcort_rm)
                areas = self._compute_areas_for_regions(list(range(centers.shape[0])), surface, region_mapping)
        gain_matrix = self._gain_matrix_distance(centers, areas, sensor_locations)
        gain_matrix = self._normalize_gain_matrix(gain_matrix, normalize)
        return gain_matrix

    def compute_gain_matrix(self, head, sensors, method="dipole", normalize=100.0, ceil=False, **kwargs):
        if isinstance(sensors, Sensors):
            sensor_locations = sensors.locations
        else:
            sensor_locations = sensors
        if method == "dipole" or method == "distance":
            if head.cortical_surface is not None and head.subcortical_surface is not None and \
                head.cortical_region_mapping is not None and head.subcortical_region_mapping is not None:
                if method == "dipole":
                    gain_mat = \
                        self._compute_seeg_dipole_gain_matrix(sensor_locations, head.cortical_surface,
                                                              head.subcortical_surface, head.cortical_region_mapping,
                                                              head.subcortical_region_mapping, normalize)
                else:
                    gain_mat = \
                        self._compute_seeg_distance_gain_matrix(sensor_locations, head.cortical_surface,
                                                                head.subcortical_surface, head.cortical_region_mapping,
                                                                head.subcortical_region_mapping, normalize)
            else:
                raise_value_error("No %s gain computation is possible since some of the cortical/subcortical surfaces "
                                  "and/or region mappings are missing!" % method)
        else:
            gain_mat = self._compute_seeg_regions_distance_gain_matrix(sensor_locations, head.connectivity.centres,
                                                                       kwargs.get("areas", None),
                                                                       getattr(head, "cortical_surface", None),
                                                                       getattr(head, "subcortical_surface", None),
                                                                       getattr(head, "cortical_region_mapping", None),
                                                                       getattr(head, "subcortical_region_mapping", None),
                                                                       normalize)
        if ceil:
            gain_mat[np.abs(gain_mat) > ceil] = ceil * np.sign(gain_mat[np.abs(gain_mat) > ceil])

        return gain_mat
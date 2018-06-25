import h5py

from tvb_infer.base.utils.log_error_utils import initialize_logger

class LSAH5Reader(object):
    logger = initialize_logger(__name__)

    def read_lsa_service(self, path):
        """
        :param path: Path towards a LSAService H5 file
        :return: LSAService object
        """
        self.logger.info("Starting to read LSAService from: %s" % path)
        h5_file = h5py.File(path, 'r', libver='latest')
        from tvb_infer.tvb_lsa.lsa_service import LSAService
        lsa_service = LSAService()

        for dataset in h5_file.keys():
            lsa_service.set_attribute(dataset, h5_file["/" + dataset][()])

        for attr in h5_file.attrs.keys():
            lsa_service.set_attribute(attr, h5_file.attrs[attr])

        h5_file.close()
        return lsa_service
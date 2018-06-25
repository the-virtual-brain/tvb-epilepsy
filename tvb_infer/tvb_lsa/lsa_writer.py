from tvb_infer.io.h5_writer_base import H5WriterBase


class LSAH5Writer(H5WriterBase):

    def write_lsa_service(self, lsa_service, path, nr_regions=None):
        """
        :param lsa_service: LSAService object to write in H5
        :param path: H5 path to be written
        """
        self.write_object_to_file(path, lsa_service, "HypothesisModel", nr_regions)
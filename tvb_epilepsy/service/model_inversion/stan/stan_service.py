import os
import pickle
from abc import ABCMeta, abstractmethod
from scipy.io import savemat, loadmat
from scipy.stats import describe
import numpy as np
from tvb_epilepsy.base.constants.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list, sort_dict
from tvb_epilepsy.io.rdump import rdump, rload
from tvb_epilepsy.io.csv import parse_csv
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer

LOG = initialize_logger(__name__)


class StanService(object):
    __metaclass__ = ABCMeta

    def __init__(self, model_name="", model=None, model_dir=FOLDER_RES, model_code=None, model_code_path="",
                 model_data_path="", fitmethod="sampling", logger=LOG):
        self.logger = logger
        self.fitmethod = fitmethod
        self.model_name = model_name
        self.model = model
        if not (os.path.isdir(model_dir)):
            os.mkdir(model_dir)
        self.model_path = os.path.join(model_dir, self.model_name)
        self.model_code = model_code
        if os.path.isfile(model_code_path):
            self.model_code_path = model_code_path
        else:
            self.model_code_path = self.model_path + ".stan"
        if model_data_path == "":
            self.model_data_path = os.path.join(model_dir, "ModelData.h5")
        self.compilation_time = 0.0

    @abstractmethod
    def compile_stan_model(self, store_model=True, **kwargs):
        pass

    @abstractmethod
    def set_model_from_file(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, model_data, **kwargs):
        pass

    def write_model_data_to_file(self, model_data, reset_path=False, **kwargs):
        model_data_path = kwargs.get("model_data_path", self.model_data_path)
        if reset_path:
            self.model_data_path = model_data_path
        extension = model_data_path.split(".", -1)[-1]
        if isequal_string(extension, "npy"):
            np.save(model_data_path, model_data)
        elif isequal_string(extension, "mat"):
            savemat(model_data_path, model_data)
        elif isequal_string(extension, "pkl"):
            with open(model_data_path, 'wb') as f:
                pickle.dump(model_data, f)
        elif isequal_string(extension, "R"):
            rdump(model_data_path, model_data)
        else:
            H5Writer().write_dictionary(model_data, os.path.join(os.path.dirname(model_data_path),
                                                                 os.path.basename(model_data_path)))

    def load_model_data_from_file(self, reset_path=False, **kwargs):
        model_data_path = kwargs.get("model_data_path", self.model_data_path)
        if reset_path:
            self.model_data_path = model_data_path
        extension = self.model_data_path.split(".", -1)[-1]
        if isequal_string(extension, "R"):
            return rload(self.model_data_path)
        elif isequal_string(extension, "npy"):
            return np.load(self.model_data_path).item()
        elif isequal_string(extension, "mat"):
            return loadmat(self.model_data_path)
        elif isequal_string(extension, "pkl"):
            with open(self.model_data_path, 'wb') as f:
                return pickle.load(f)
        elif isequal_string(extension, "h5"):
            return H5Reader().read_dictionary(self.model_data_path)
        else:
            raise_not_implemented_error("model_data file (" + model_data_path +
                                        ") that are not one of (.R, .npy, .mat, .pkl) cannot be read!")

    def set_or_compile_model(self, **kwargs):
        try:
            self.set_model_from_file(**kwargs)
        except:
            self.logger.info("Trying to compile model from file: " + str(self.model_code_path) + str("!"))
            self.compile_stan_model(store_model=kwargs.get("store_model", True), **kwargs)

    def read_output_csv(self, output_filepath, **kwargs):
        csvs = parse_csv(output_filepath.replace(".csv", "*"), merge=kwargs.pop("merge_outputs", False))
        ests = []
        for csv in ensure_list(csvs):
            est = {}
            for pkey, pval in csv.iteritems():
                try:
                    est[pkey + "_s"] = csv[pkey]
                    est[pkey + "_low"], est[pkey], est[pkey + "_std"] = describe(csv[pkey])[1:4]
                    est[pkey + "_high"] = est[pkey + "_low"][1]
                    est[pkey + "_low"] = est[pkey + "_low"][0]
                    est[pkey + "_std"] = np.sqrt(est[pkey + "_std"])
                    for skey in [pkey, pkey + "_low", pkey + "_high", pkey + "_std"]:
                        est[skey] = np.squeeze(est[skey])
                except:
                    est[pkey] = csv[pkey]
            ests.append(sort_dict(est))
        if len(ests) == 1:
            return ests[0], csv[0]
        else:
            return ests, csv

    def trace_nuts(self, csv, extras='', skip=0):
        from pylab import subplot, plot, gca, title, grid, xticks
        if isinstance(extras, str):
            extras = extras.split()
            for csvi in csv:
                i = 1
                for key in csvi.keys():
                    if key[-2:] == '__' or key in extras:
                        subplot(4, 4, i)
                        plot(csvi[key][skip:], alpha=0.5)
                        if key in ('stepsize__',):
                            gca().set_yscale('log')
                        title(key)
                        grid(1)
                        if ((i - 1) / 4) < 4:
                            xticks(xticks()[0], [])
                        i += 1

    def pair_plots(self, csv, keys, skip=0):
        import pylab as pl
        n = len(keys)
        if isinstance(csv, dict):
            csv = [csv]  # following assumes list of chains' results
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                pl.subplot(n, n, i * n + j + 1)
                for csvi in csv:
                    if i == j:
                        pl.hist(csvi[key_i][skip:], 20, log=True)
                    else:
                        pl.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
                if i == 0:
                    pl.title(key_j)
                if j == 0:
                    pl.ylabel(key_i)
        pl.tight_layout()

    # def plot_HMC(self, csv, extras, output_file_path, figure_name):
    #     outout_folder = os.path.dirname(output_file_path)
    #     self.trace_nuts(csv)
    #     # tight_layout()
    #     pyplot.savefig(os.path.join(outout_folder, figure_name + "_stats.png"))
    #     pyplot.ion()
    #     pyplot.show()
    #     pyplot.figure(figsize=(10, 10))
    #     self.pair_plots(csv, extras, skip=0)
    #     pyplot.savefig(os.path.join(outout_folder, figure_name + "_pairplots.png"))
    #     pyplot.ion()
    #     pyplot.show()
    #     for i, csvi in enumerate(csv):
    #         pyplot.figure()
    #         self.phase_space(csvi)
    #         pyplot.suptitle("Chain {" + str(i) + "}")
    #         # tight_layout()
    #         pyplot.savefig(os.path.join(outout_folder, figure_name + "_state_space_" + str(i) + ".png"))
    #     self.ppc_seeg(csv[1], skip=200)
    #     pyplot.savefig(os.path.join(outout_folder, figure_name + "_seeg.png"))
    #     vep_stan.lib.violin_x0(csv)
    #     pyplot.savefig(os.path.join(outout_folder, figure_name + "_x0.png"))
    #     pyplot.imshow(csv[0]['FC'].mean(axis=0))
    #     pyplot.colorbar()

import os
from abc import ABCMeta, abstractmethod
import pickle

import pylab as pl
import numpy as np

from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path, isequal_string
from tvb_epilepsy.base.h5_model import convert_to_h5_model


LOG = initialize_logger(__name__)


class StanService(object):
    __metaclass__ = ABCMeta

    def __init__(self, model_name="", model=None, model_dir=os.path.join(FOLDER_VEP_HOME, "stan_models"),
                 model_code=None, model_code_path="", fitmethod="sampling", logger=LOG, **options):
        self.logger = logger
        self.fitmethod = fitmethod
        self.model_name = model_name
        self.model = model
        if not(os.path.isdir(model_dir)):
            os.mkdir(model_dir)
        self.model_path = os.path.join(model_dir, self.model_name)
        self.model_code = model_code
        if os.path.exist(model_code_path):
            self.model_code_path = model_code_path
        else:
            self.model_code_path = self.model_path + ".stan"
        self.compilation_time = 0.0
        self.options = options
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "ProbabilityDistributionModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.type + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    @abstractmethod
    def compile_stan_model(self, store_model=True, **kwargs):
        pass

    @abstractmethod
    def set_model_from_file(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, model_data, **kwargs):
        pass

    def set_or_compile_model(self, **kwargs):
        try:
            self.set_model_from_file(**kwargs)
        except:
            self.logger.info("Failed to load the model from file: " + str(self.model_path) + " !" +
                             "\nTrying to compile model from file: " + str(self.model_code_path) + str("!"))
            self.compile_stan_model(store_model=kwargs.get("store_model", True), **kwargs)

    def write_model_data_to_file(self, model_data, model_data_path, filemode="rdumb", **kwargs):
        if isequal_string(filemode, "numpy"):
            np.save(model_data_path, model_data)
        elif isequal_string(filemode, "pickle"):
            with open(model_data_path, 'wb') as f: pickle.dump(model_data, f)
        else:
            rdump(model_data_path, model_data)

    def load_model_data_from_file(self, model_data_path, **kwargs):
        if isequal_string(model_data_path[-3:], "npy"):
            model_data = np.load(model_data_path).item()
        elif isequal_string(model_data_path[-3:], "pkl"):
            with open(model_data_path, 'wb') as f:
                pickle.load(model_data_path)
        else:
            raise_not_implemented_error("model_data files that are neither .npy nor .pkl cannot be read!")

def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size,) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{0}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(
            key=key, c=c, dim=dim)
        return struct


def rdump(filepath, data):
    """Dump a dict of data to a R dump format file.
    """
    with open(filepath, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            else:
                try:
                    val = val.flat[0]
                except:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')


def merge_csv_data(*csvs):
    data_ = {}
    for csv in csvs:
        for key, val in csv.items():
            if key in data_:
                data_[key] = np.concatenate(
                    (data_[key], val),
                    axis=0
                )
            else:
                data_[key] = val
    return data_


def parse_csv(fname, merge=True):
    if '*' in fname:
        import glob
        return parse_csv(glob.glob(fname), merge=merge)
    if isinstance(fname, (list, tuple)):
        csv = [parse_csv(_) for _ in fname]
        if merge:
            csv = merge_csv_data(*csv)
        return csv

    lines = []
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            if not line.startswith('#'):
                lines.append(line.strip().split(','))
    names = [field.split('.') for field in lines[0]]
    data = np.array([[float(f) for f in line] for line in lines[1:]])

    namemap = {}
    maxdims = {}
    for i, name in enumerate(names):
        if name[0] not in namemap:
            namemap[name[0]] = []
        namemap[name[0]].append(i)
        if len(name) > 1:
            maxdims[name[0]] = name[1:]

    for name in maxdims.keys():
        dims = []
        for dim in maxdims[name]:
            dims.append(int(dim))
        maxdims[name] = tuple(reversed(dims))

    # data in linear order per Stan, e.g. mat is col maj
    # TODO array is row maj, how to distinguish matrix v array[,]?
    data_ = {}
    for name, idx in namemap.items():
        new_shape = (-1,) + maxdims.get(name, ())
        data_[name] = data[:, idx].reshape(new_shape)

    return data_


def csv2mode(csv_fname, mode=None):
    csv = parse_csv(csv_fname)
    data = {}
    for key, val in csv.items():
        if key.endswith('__'):
            continue
        if mode is None:
            val_ = val[0]
        elif mode == 'mean':
            val_ = val.mean(axis=0)
        elif mode[0] == 'p':
            val_ = np.percentile(val, int(mode[1:]), axis=0)
        data[key] = val_
    return data


def csv2r(csv_fname, r_fname=None, mode=None):
    data = csv2mode(csv_fname, mode=mode)
    r_fname = r_fname or csv_fname + '.R'
    rdump(r_fname, data)


def viz_phase_space(data):
    opt = len(data['x']) == 1
    npz = np.load('data.R.npz')
    tr = lambda A: np.transpose(A, (0, 2, 1))
    x, z = tr(data['x']), tr(data['z'])
    tau0 = npz['tau0']
    X, Z = np.mgrid[-5.0:5.0:50j, -5.0:5.0:50j]
    dX = (npz['I1'] + 1.0) - X ** 3.0 - 2.0 * X ** 2.0 - Z
    x0mean = data['x0'].mean(axis=0)
    Kmean = data['K'].mean(axis=0)

    def nullclines(i):
        pl.contour(X, Z, dX, 0, colors='r')
        dZ = (1.0 / tau0) * (4.0 * (X - x0mean[i])) - Z - Kmean * (-npz['Ic'][i] * (1.8 + X))
        pl.contour(X, Z, dZ, 0, colors='b')

    for i in range(x.shape[-1]):
        pl.subplot(2, 3, i + 1)
        if opt:
            pl.plot(x[0, :, i], z[0, :, i], 'k', alpha=0.5)
        else:
            for j in range(1 if opt else 10):
                pl.plot(x[-j, :, i], z[-j, :, i], 'k', alpha=0.2, linewidth=0.5)
        nullclines(i)
        pl.grid(True)
        pl.xlabel('x(t)')
        pl.ylabel('z(t)')
        # pl.title(f'node {i}')
    pl.tight_layout()


def viz_pair_plots(csv, keys, skip=0):
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


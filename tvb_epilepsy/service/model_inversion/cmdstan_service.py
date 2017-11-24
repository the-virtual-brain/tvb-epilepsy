import os
# import pickle
import time
import subprocess

# import numpy as np
import pystan as ps

from tvb_epilepsy.base.constants.configurations import FOLDER_VEP_HOME, CMDSTAN_PATH
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error, raise_not_implemented_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import construct_import_path
from tvb_epilepsy.service.model_inversion.stan_service import StanService


LOG = initialize_logger(__name__)


class CmdStanService(StanService):

    def __init__(self, model_name=None, model=None, model_dir=os.path.join(FOLDER_VEP_HOME, "stan_models"),
                 model_code=None, model_code_path="", fitmode="sampling", cmdstanpath=CMDSTAN_PATH, logger=LOG):
        super(CmdStanService, self).__init__(model_name, model, model_dir, model_code, model_code_path, fitmode, logger)
        if not os.path.exists(os.path.join(cmdstanpath, 'runCmdStanTests.py')):
            raise_value_error('Please provide CmdStan path, e.g. lib.cmdstan_path("/path/to/")!')
        self.path = cmdstanpath
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.create_str = self.__class__.__name__ + "()"

    def load_or_compile_model(self, **kwargs):
        self.model_path = kwargs.get("model_code_path", self.model_path)
        if not(os.path.isfile(self.model_path)):
            self.compile_stan_model(**kwargs)

    def compile_stan_model(self, **kwargs):
        self.model_code_path = kwargs.get("model_code_path", self.model_code_path)
        tic = time.time()
        self.logger.info("Compiling model...")
        proc = subprocess.Popen(['make', self.model_code_path],
                                cwd=self.path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = proc.stdout.read().decode('ascii').strip()
        if stdout:
            print(stdout)
        stderr = proc.stderr.read().decode('ascii').strip()
        if stderr:
            print(stderr)
        self.compilation_time = time.time() - tic
        self.logger.info(str(self.compilation_time) + ' sec required to compile')

    def _rdump_array(key, val):
        c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
        if (val.size,) == val.shape:
            return '{key} <- {c}'.format(key=key, c=c)
        else:
            dim = '.Dim = c{0}'.format(val.shape)
            struct = '{key} <- structure({c}, {dim})'.format(
                key=key, c=c, dim=dim)
            return struct

    def rdump(fname, data):
        """Dump a dict of data to a R dump format file.
        """
        with open(fname, 'w') as fd:
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





def compile_model(stan_fname):
    path = os.path.abspath(os.path.dirname(stan_fname))
    name = stan_fname[:-5]
    target = os.path.join(path, name)
    proc = subprocess.Popen(['make', target], cwd=cmdstan_path(),  stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
    stdout = proc.stdout.read().decode('ascii').strip()
    if stdout:
        print(stdout)
    stderr = proc.stderr.read().decode('ascii').strip()
    if stderr:
        print(stderr)

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
    pl.title(f'node {i}')
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

    # def fit_stan_model(self, model_data=None, **kwargs):
    #     self.logger.info("Model fitting with " + self.fitmode + "...")
    #     tic = time.time()
    #     fit = getattr(self.model, self.fitmode)(data=model_data, **kwargs)
    #     self.fitting_time = time.time() - tic
    #     self.logger.info(str(self.fitting_time) + ' sec required to fit')
    #     if self.fitmode is "optimizing":
    #         return fit,
    #     else:
    #         self.logger.info("Extracting estimates...")
    #         if self.fitmode is "sampling":
    #             est = fit.extract(permuted=True)
    #         elif self.fitmode is "vb":
    #             est = self.read_vb_results(fit)
    #         return est, fit
    #
    # def read_vb_results(self, fit):
    #     est = {}
    #     for ip, p in enumerate(fit['sampler_param_names']):
    #         p_split = p.split('.')
    #         p_name = p_split.pop(0)
    #         p_name_samples = p_name + "_s"
    #         if est.get(p_name) is None:
    #             est.update({p_name_samples: []})
    #             est.update({p_name: []})
    #         if len(p_split) == 0:
    #             # scalar parameters
    #             est[p_name_samples] = fit["sampler_params"][ip]
    #             est[p_name] = fit["mean_pars"][ip]
    #         else:
    #             if len(p_split) == 1:
    #                 # vector parameters
    #                 est[p_name_samples].append(fit["sampler_params"][ip])
    #                 est[p_name].append(fit["mean_pars"][ip])
    #             else:
    #                 ii = int(p_split.pop(0)) - 1
    #                 if len(p_split) == 0:
    #                     # 2D matrix parameters
    #                     if len(est[p_name]) < ii + 1:
    #                         est[p_name_samples].append([fit["sampler_params"][ip]])
    #                         est[p_name].append([fit["mean_pars"][ip]])
    #                     else:
    #                         est[p_name_samples][ii].append(fit["sampler_params"][ip])
    #                         est[p_name][ii].append(fit["mean_pars"][ip])
    #                 else:
    #                     if len(est[p_name]) < ii + 1:
    #                         est[p_name_samples].append([])
    #                         est[p_name].append([])
    #                     jj = int(p_split.pop(0)) - 1
    #                     if len(p_split) == 0:
    #                         # 3D matrix parameters
    #                         if len(est[p_name][ii]) < jj + 1:
    #                             est[p_name_samples][ii].append([fit["sampler_params"][ip]])
    #                             est[p_name][ii].append([fit["mean_pars"][ip]])
    #                         else:
    #                             if len(est[p_name][ii]) < jj + 1:
    #                                 est[p_name_samples][ii].append([])
    #                                 est[p_name][ii].append([])
    #                             est[p_name_samples][ii][jj].append(fit["sampler_params"][ip])
    #                             est[p_name][ii][jj].append(fit["mean_pars"][ip])
    #                     else:
    #                         raise_not_implemented_error("Extracting of parameters of more than 3 dimensions is not " +
    #                                                     "implemented yet for vb!", self.logger)
    #     for key in est.keys():
    #         if isinstance(est[key], list):
    #             est[key] = np.squeeze(np.array(est[key]))
    #     return est

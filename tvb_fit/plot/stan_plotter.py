from tvb_fit.base.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)

from tvb_fit.samplers.stan.stan_interface import merge_samples

from tvb_utils.data_structures_utils import ensure_list, extract_dict_stringkeys
from tvb_plot.base_plotter import BasePlotter


class STANplotter(BasePlotter):

    def plot_HMC(self, samples, skip_samples=0, title='HMC NUTS trace', figure_name=None,
                 figsize=FiguresConfig.LARGE_SIZE):
        nuts = []
        for sample in ensure_list(samples):
            nuts.append(extract_dict_stringkeys(sample, "__", modefun="find"))
        if len(nuts) > 1:
            nuts = merge_samples(nuts)
        else:
            nuts = nuts[0]
        n_chains_or_runs = nuts.values()[0].shape[0]
        legend = {nuts.keys()[0]: ["chain/run " + str(ii + 1) for ii in range(n_chains_or_runs)]}
        return self.plots(nuts, shape=(4, 2), transpose=True, skip=skip_samples, xlabels={}, xscales={},
                          yscales={"stepsize__": "log"}, lgnd=legend, title=title,
                          figure_name=figure_name, figsize=figsize)
"""
Various plotting tools will be placed here.
"""
# TODO: make a plot function for sensitivity analysis results
import os
import numpy as np
from matplotlib import pyplot
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG

try:
    # https://github.com/joferkington/mpldatacursor
    # pip install mpldatacursor
    # Not working with the MacosX graphic's backend
    from mpldatacursor import HighlightingDataCursor  # datacursor

    MOUSEHOOVER = True
except:
    warning("\nNo mpldatacursor module found! MOUSEHOOVER will not be available.")
    MOUSEHOOVER = False


# TODO: deprecated
def check_show(show_flag=SHOW_FLAG):
    if show_flag:
        # mp.use('TkAgg')
        pyplot.ion()
        pyplot.show()
    else:
        # mp.use('Agg')
        pyplot.ioff()
        pyplot.close()


# TODO: deprecated
def figure_filename(fig=None, figure_name=None):
    if fig is None:
        fig = pyplot.gcf()
    if figure_name is None:
        figure_name = fig.get_label()
    figure_name = figure_name.replace(": ", "_").replace(" ", "_").replace("\t", "_")
    return figure_name


# TODO: deprecated
def save_figure(save_flag=SAVE_FLAG, fig=None, figure_name=None, figure_dir=FOLDER_FIGURES,
                figure_format=FIG_FORMAT):
    if save_flag:
        figure_name = figure_filename(fig, figure_name)
        figure_name = figure_name[:np.min([100, len(figure_name)])] + '.' + figure_format
        if not (os.path.isdir(figure_dir)):
            os.mkdir(figure_dir)
        pyplot.savefig(os.path.join(figure_dir, figure_name))

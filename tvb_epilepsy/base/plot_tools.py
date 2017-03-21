"""
Various plotting tools will be placed here.
"""

import numpy 
from scipy.stats.mstats import zscore
import matplotlib as mp
from matplotlib import pyplot, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tvb_epilepsy.base.constants import *
from tvb_epilepsy.base.utils import calculate_in_degree
from tvb_epilepsy.base.calculations import calc_fx1, calc_fx1, calc_fz, calc_fz, calc_fx1_2d_taylor
from tvb_epilepsy.base.equilibrium_computation import calc_eq_y1
from tvb_epilepsy.tvb_api.epileptor_models import *

try:
    #https://github.com/joferkington/mpldatacursor
    #pip install mpldatacursor
    #Not working with the MacosX graphic's backend
    from mpldatacursor import HighlightingDataCursor #datacursor
    MOUSEHOOVER = True
except ImportError:
    pass


def plot_head(head, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,figsize=LARGE_SIZE):
    plot_connectivity(head.connectivity, show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                      figure_format=figure_format, figsize=figsize)

    plot_head_stats(head.connectivity, show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                    figure_format=figure_format)

    count = _show_projections_dict(head, head.sensorsEEG, 1, show_flag=show_flag,
                                   save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)
    count = _show_projections_dict(head, head.sensorsSEEG, count, show_flag=show_flag,
                                   save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)
    _show_projections_dict(head, head.sensorsMEG, count, show_flag=show_flag,
                           save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)


def plot_connectivity(conn, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                      figure_format=FIG_FORMAT, figure_name='Connectivity ', figsize=LARGE_SIZE):

    mp.pyplot.figure(figure_name + str(conn.number_of_regions), figsize)
    #_plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
    _plot_regions2regions(conn.normalized_weights, conn.region_labels, 121, "normalised weights")
    _plot_regions2regions(conn.tract_lengths, conn.region_labels, 122, "tract lengths")

    if save_flag:
        _save_figure(figure_dir=figure_dir, figure_format=figure_format,
                     figure_name=figure_name.replace(" ", "_").replace("\t", "_"))
    _check_show(show_flag=show_flag)


def plot_head_stats(conn, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                    figure_name='HeadStats '):
    mp.pyplot.figure("Head stats " + str(conn.number_of_regions), figsize=LARGE_SIZE)
    ax = _plot_vector(calculate_in_degree(conn.normalized_weights), conn.region_labels, 121, "w in-degree")
    ax.invert_yaxis()
    if conn.areas is not None:
        ax = _plot_vector(conn.areas, conn.region_labels, 122, "region areas")
        ax.invert_yaxis()
    if save_flag:
        _save_figure(figure_dir=figure_dir, figure_format=figure_format,
                     figure_name=figure_name.replace(" ", "").replace("\t", ""))
    _check_show(show_flag=show_flag)


def _check_show(show_flag=SHOW_FLAG):
    if show_flag:
        # mp.use('TkAgg')
        mp.pyplot.ion()
        mp.pyplot.show()
    else:
        # mp.use('Agg')
        mp.pyplot.ioff()
        mp.pyplot.close()


def _save_figure(figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name='figure'):
    if not (os.path.isdir(figure_dir)):
        os.mkdir(figure_dir)
    figure_name = figure_name + '.' + figure_format
    mp.pyplot.savefig(os.path.join(figure_dir, figure_name))


def _plot_vector(vector, labels, subplot, title, show_y_labels=True, indices_red=None, sharey=None):
    ax = mp.pyplot.subplot(subplot, sharey=sharey)
    mp.pyplot.title(title)
    n_vector = labels.shape[0]

    y_ticks = numpy.array(range(n_vector), dtype=numpy.int32)
    color = 'k'
    colors = numpy.repeat([color], n_vector)
    if indices_red is None:
        indices_red = y_ticks
        coldif = False
    else:
        colors[indices_red] = 'r'
        coldif = True
    if len(vector.shape) == 1:
        ax.barh(y_ticks, vector, color=colors, align='center')
    else:
        ax.barh(y_ticks, vector[0, :], color=colors, align='center')
    # ax.invert_yaxis()
    ax.grid(True, color='grey')
    ax.set_yticks(y_ticks)
    if show_y_labels:
        region_labels = numpy.array(["%d. %s" % l for l in zip(range(n_vector), labels)])
        ax.set_yticklabels(region_labels)
        if coldif:
            labels = ax.yaxis.get_ticklabels()
            for ids in indices_red:
                labels[ids].set_color('r')
            ax.yaxis.set_ticklabels(labels)
    else:
        ax.set_yticklabels([])

    ax.autoscale(tight=True)
    return ax


def _plot_regions2regions(adj, labels, subplot, title, show_y_labels=True, show_x_labels=True,
                          indices_red_x=None, sharey=None):
    ax = mp.pyplot.subplot(subplot, sharey=sharey)
    mp.pyplot.title(title)

    y_color = 'k'
    adj_size = adj.shape[0]
    y_ticks = numpy.array(range(adj_size), dtype=numpy.int32)
    if indices_red_x is None:
        indices_red_x = y_ticks
        x_ticks = indices_red_x
        x_color = y_color
    else:
        x_color = 'r'
        x_ticks = range(len(indices_red_x))
    region_labels = numpy.array(["%d. %s" % l for l in zip(range(adj_size), labels)])
    cmap = mp.pyplot.set_cmap('autumn_r')
    img = ax.imshow(adj[indices_red_x, :].T, cmap=cmap, interpolation='none')
    ax.set_xticks(x_ticks)
    ax.grid(True, color='grey')

    if show_y_labels:
        region_labels = numpy.array(["%d. %s" % l for l in zip(range(adj_size), labels)])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(region_labels)
        if not (x_color == y_color):
            labels = ax.yaxis.get_ticklabels()
            for idx in indices_red_x:
                labels[idx].set_color('r')
            ax.yaxis.set_ticklabels(labels)
    else:
        ax.set_yticklabels([])

    if show_x_labels:
        ax.set_xticklabels(region_labels[indices_red_x], rotation=90, color=x_color)
    else:
        ax.set_xticklabels([])

    ax.autoscale(tight=True)

    # make a color bar
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    mp.pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0

    return ax


def _show_projections_dict(head, sensors_dict, current_count=1, show_flag=SHOW_FLAG,
                           save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
    for sensors, projection in sensors_dict.iteritems():
        if projection is None:
            continue
        _plot_projection(projection, head.connectivity, sensors,
                         title=str(current_count) + " - " + sensors.s_type + " - Projection",
                         show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format)

        current_count += 1
    return current_count


def _plot_projection(proj, connectivity, sensors, figure=None, title="Projection",
                     y_labels=1, x_labels=1, x_ticks=None, y_ticks=None, show_flag=SHOW_FLAG,
                     save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name=''):
    if not (isinstance(figure, mp.pyplot.Figure)):
        figure = mp.pyplot.figure(title, figsize=LARGE_SIZE)

    n_sensors = sensors.number_of_sensors
    n_regions = connectivity.number_of_regions
    if x_ticks is None:
        x_ticks = numpy.array(range(n_sensors), dtype=numpy.int32)
    if y_ticks is None:
        y_ticks = numpy.array(range(n_regions), dtype=numpy.int32)

    cmap = mp.pyplot.set_cmap('autumn_r')
    img = mp.pyplot.imshow(proj[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
    mp.pyplot.grid(True, color='black')
    if y_labels > 0:
        region_labels = numpy.array(["%d. %s" % l for l in zip(range(n_regions), connectivity.region_labels)])
        mp.pyplot.yticks(y_ticks, region_labels[y_ticks])
    else:
        mp.pyplot.yticks(y_ticks)
    if x_labels > 0:
        sensor_labels = numpy.array(["%d. %s" % l for l in zip(range(n_sensors), sensors.labels)])
        mp.pyplot.xticks(x_ticks, sensor_labels[x_ticks], rotation=90)
    else:
        mp.pyplot.xticks(x_ticks)

    ax = figure.get_axes()[0]
    ax.autoscale(tight=True)
    mp.pyplot.title(title)

    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    mp.pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0

    if save_flag:
        if figure_name == '':
            figure_name = title.replace(" ", "").replace("\t", "_")
        _save_figure(figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)

    _check_show(show_flag)
    return figure


def plot_nullclines_eq(hypothesis,region_labels, special_idx=None, model="2d", zmode=numpy.array("lin"),
                       x0ne=X0_DEF, x0e=X0_CR_DEF, figure_name='Nullclines and equilibria',
                       show_flag=SHOW_FLAG, save_flag=False, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT,
                       figsize=SMALL_SIZE):

    add_name = " " + "Epileptor " + model + " z-" + str(zmode)
    figure_name = hypothesis.name + " " + figure_name + add_name

    #Fixed parameters for all regions:
    x1eq = numpy.mean(hypothesis.x1EQ)
    yc = numpy.mean(hypothesis.yc)
    Iext1 = numpy.mean(hypothesis.Iext1)
    x0cr = numpy.mean(hypothesis.x0cr)  # Critical x0
    r = numpy.mean(hypothesis.rx0)
    if model != "2d" or zmode != numpy.array("lin"):
        x0cr, r = calc_x0cr_r(yc, Iext1, epileptor_model=model, zmode=zmode, x1_rest=X1_DEF,
                                              x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF)

    x1lin0 = numpy.mean(hypothesis.x1LIN)  # The point of the linear approximation (1st order Taylor expansion)
    x1sq0 = numpy.mean(hypothesis.x1SQ)  # The point of the square (parabolic) approximation (2nd order Taylor expansion)
    
    # Lines:

    # x1 nullcline:
    if model == "2d":
        x1 = numpy.expand_dims(numpy.linspace(-2.0, 2.0 / 3.0, 100), 1).T
        zX1 = calc_fx1(x1, z=0, y1=yc, Iext1=Iext1, x1_neg=None) #yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2
        # approximations:
        # linear:
        x1lin = numpy.expand_dims(numpy.linspace(-5.5 / 3.0, -3.5 / 3, 30), 1).T
        # x1 nullcline after linear approximation
        zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, yc=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
                                    x1_neg=None, order=2)  # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
        # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1 nullcline after linear approximation
        # center point without approximation:
        # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
        # square:
        x1sq = numpy.expand_dims(numpy.linspace(-5.0 / 3, -1.0, 30), 1).T
        # x1 nullcline after parabolic approximation
        zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, yc=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
                                   x1_neg=None, order=3)  # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2

        # z nullcline:
        zZe = calc_fz(x1, z=0.0, x0=x0e, x0cr=x0cr, r=r, zmode=zmode)  # for epileptogenic regions
        zZne = calc_fz(x1, z=0.0, x0=x0ne, x0cr=x0cr, r=r, zmode=zmode)  # for non-epileptogenic regions

    else:
        x1 = numpy.expand_dims(numpy.linspace(-2.0*10, 2.0*10 / 3.0, 100), 1).T
        zX1 = calc_fx1(x1, z=0.0, y1=calc_eq_y1(x1eq, yc, d=5.0), Iext1=Iext1, model="2d", x1_neg=None) #y1eq + Iext1 - x1 ** 3 + 3.0 * x1 ** 2

        # z nullcline:
        zZe = calc_fz(x1, z=0.0, x0=x0e, zmode=zmode, model="2d")   # for epileptogenic regions
        zZne = calc_fz(x1, z=0.0, x0=x0ne, zmode=zmode, model="2d")  # for non-epileptogenic regions

    fig = mp.pyplot.figure(figure_name, figsize=figsize)
    x1null, = mp.pyplot.plot(x1[0, :], zX1[0, :], 'b-', label='x1 nullcline', linewidth=1)
    ax = mp.pyplot.gca()
    ax.axes.hold(True)
    zE1null, = mp.pyplot.plot(x1[0, :], zZe[0, :], 'g-', label='z nullcline at critical point (E=1)', linewidth=1)
    zE2null, = mp.pyplot.plot(x1[0, :], zZne[0, :], 'g--', label='z nullcline for E=0', linewidth=1)
    if model == "2d":
        sq, = mp.pyplot.plot(x1sq[0, :], zX1sq[0, :], 'm--', label='Parabolic local approximation', linewidth=2)
        lin, = mp.pyplot.plot(x1lin[0, :], zX1lin[0, :], 'c--', label='Linear local approximation', linewidth=2)
        mp.pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])
    else:
        mp.pyplot.legend(handles=[x1null, zE1null, zE2null])
    
    ii=range(hypothesis.n_regions)
    if special_idx is None:
        ii = numpy.delete(ii, special_idx)
        
    points =[]    
    for i in ii:
        point, = mp.pyplot.plot(hypothesis.x1EQ[0,i], hypothesis.zEQ[0,i], '*', mfc='k', mec='k', ms=10,  alpha=0.3,
                                label=str(i)+'.'+region_labels[i])
        points.append(point)
    if special_idx is None:
        for i in special_idx:
            point, = mp.pyplot.plot(hypothesis.x1EQ[0,i], hypothesis.zEQ[0,i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
                                    label=str(i)+'.'+region_labels[i])
            points.append(point)
    #ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
    #ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'E=0.0', fontsize=10, color='r')
    #ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
    #ax.axes.text(x1sq0, zsq0 - 0.2, 'E=1.0', fontsize=10, color='m')
    if model == "2d":
        ax.set_title("Equilibria, nullclines and Taylor series approximations \n at the x1-z phase plane of the" +
                     add_name + " for x1<0")
    else:
        ax.set_title("Equilibria, nullclines at the x1-z phase plane of the" + add_name + " for x1<0")
    ax.axes.autoscale(tight=True)
    ax.axes.set_xlabel('x1')
    ax.axes.set_ylabel('z')
    ax.axes.set_ylim(2.0, 5.0)    
    if MOUSEHOOVER:
        #datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'), 
        #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
        HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'), 
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )

    if save_flag:
        if len(fig.get_label())==0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(" ", "_").replace("\t", "_")
        _save_figure(figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    _check_show(show_flag)


def plot_hypothesis(hypothesis, region_labels, figure_name='', show_flag=SHOW_FLAG,
                    save_flag=False, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=LARGE_SIZE):
    fig = mp.pyplot.figure('Hypothesis ' + hypothesis.name, frameon=False, figsize=figsize)
    mp.gridspec.GridSpec(1, 7, width_ratios=[1, 1, 1, 1, 1, 2, 1])

    ax0 = _plot_vector(hypothesis.x0, region_labels, 171, 'Excitabilities x0',
                       show_y_labels=False, indices_red=hypothesis.seizure_indices)

    _plot_vector(hypothesis.E, region_labels, 172, 'Epileptogenicities E',
                 show_y_labels=False, indices_red=hypothesis.seizure_indices, sharey=ax0)

    _plot_vector(hypothesis.x1EQ, region_labels, 173, 'x1 Equilibria',
                 show_y_labels=False, indices_red=hypothesis.seizure_indices, sharey=ax0)

    _plot_vector(hypothesis.zEQ, region_labels, 174, 'z Equilibria',
                 show_y_labels=True, indices_red=hypothesis.seizure_indices, sharey=ax0)

    _plot_vector(hypothesis.Ceq, region_labels, 175, 'Total afferent coupling \n at equilibrium',
                 show_y_labels=False, indices_red=hypothesis.seizure_indices, sharey=ax0)

    if hypothesis.n_seizure_nodes > 0:
        _plot_regions2regions(hypothesis.weights, region_labels, 176,
                              'Afferent connectivity \n from seizuring regions',
                              show_y_labels=False, show_x_labels=True,
                              indices_red_x=hypothesis.seizure_indices, sharey=ax0)

    if len(hypothesis.lsa_ps) > 0:
        _plot_vector(hypothesis.lsa_ps, region_labels, 177, 'LSA Propagation Strength',
                     show_y_labels=False, indices_red=hypothesis.seizure_indices, sharey=ax0)

    _set_axis_labels(fig, 121, hypothesis.n_regions, region_labels, hypothesis.seizure_indices, 'r')
    _set_axis_labels(fig, 122, hypothesis.n_regions, region_labels, hypothesis.seizure_indices, 'r', 'right')

    if save_flag:
        if figure_name == '':
            figure_name = fig.get_label()
        _save_figure(figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    _check_show(show_flag)
    
    plot_nullclines_eq(hypothesis,region_labels,special_idx=hypothesis.seizure_indices, model="2d",
                       zmode=numpy.array("lin"), figure_name='Nullclines and equilibria', show_flag=show_flag,
                       save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format, figsize=figsize)

    plot_nullclines_eq(hypothesis, region_labels, special_idx=hypothesis.seizure_indices, model="2d",
                       zmode=numpy.array("sig"), figure_name='Nullclines and equilibria', show_flag=show_flag,
                       save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format, figsize=figsize)

    plot_nullclines_eq(hypothesis, region_labels, special_idx=hypothesis.seizure_indices, model="6d",
                       zmode=numpy.array("lin"), figure_name='Nullclines and equilibria', show_flag=show_flag,
                       save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format, figsize=figsize)

    plot_nullclines_eq(hypothesis, region_labels, special_idx=hypothesis.seizure_indices, model="6d",
                      zmode=numpy.array("sig"), figure_name='Nullclines and equilibria', show_flag=show_flag,
                      save_flag=save_flag, figure_dir=figure_dir, figure_format=figure_format, figsize=figsize)


                       

def _set_axis_labels(fig, sub, n_regions, region_labels, indices2emphasize, color='k', position='left'):
    y_ticks = range(n_regions)
    region_labels = numpy.array(["%d. %s" % l for l in zip(y_ticks, region_labels)])
    big_ax = fig.add_subplot(sub, frameon=False)
    if position == 'right':
        big_ax.yaxis.tick_right()
        big_ax.yaxis.set_label_position("right")
    big_ax.set_yticks(y_ticks)
    big_ax.set_yticklabels(region_labels, color='k')
    if not (color == 'k'):
        labels = big_ax.yaxis.get_ticklabels()
        for idx in indices2emphasize:
            labels[idx].set_color(color)
        big_ax.yaxis.set_ticklabels(labels)
    big_ax.invert_yaxis()
    big_ax.axes.get_xaxis().set_visible(False)
    big_ax.set_axis_bgcolor('none')


def plot_timeseries(time, data_dict, special_idx=None, title='Time Series', show_flag=SHOW_FLAG,
                    save_flag=False, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name='TimeSeries',labels=None,figsize=LARGE_SIZE):
                       
    mp.pyplot.figure(title, figsize=figsize)
    no_rows = len(data_dict)
    lines = []
    for i, subtitle in enumerate(data_dict):
        ax = mp.pyplot.subplot(no_rows, 1, i + 1)
        mp.pyplot.hold(True)
        if i == 0:
            mp.pyplot.title(title)
        data = data_dict[subtitle]
        nTS = data.shape[1]
        if labels is None: 
            labels = numpy.array(range(nTS)).astype(str)
        lines.append([])
        if special_idx is None:
            for iTS in range(nTS):
                line, = mp.pyplot.plot(time, data[:,iTS], 'k', alpha=0.3, label = labels[iTS])
                lines[i].append(line)
        else:
            mask = numpy.array(range(nTS))
            mask = numpy.delete(mask,special_idx)
            for iTS in special_idx:
                line, = mp.pyplot.plot(time, data[:, iTS], 'r', alpha=0.7, label = labels[iTS])
                lines[i].append(line)
            for iTS in mask:
                line, = mp.pyplot.plot(time, data[:, iTS], 'k', alpha=0.3, label = labels[iTS])
                lines[i].append(line)
        mp.pyplot.ylabel(subtitle)
        ax.set_autoscalex_on(False)
        ax.set_xlim([time[0], time[-1]])
        if MOUSEHOOVER:
            #datacursor( lines[i], formatter='{label}'.format, bbox=dict(fc='white'), 
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(lines[i], formatter='{label}'.format, bbox=dict(fc='white'), 
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )
    mp.pyplot.xlabel("Time (ms)")

    if save_flag:
        fig = mp.pyplot.gcf() 
        if len(fig.get_label())==0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(" ", "_").replace("\t", "_")
        _save_figure(figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    _check_show(show_flag)


def plot_raster(time, data_dict, special_idx=None, title='Time Series', offset=3.0, show_flag=SHOW_FLAG,
                    save_flag=False, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name='TimeSeries',labels=None,figsize=LARGE_SIZE):
                       
    mp.pyplot.figure(title, figsize=figsize)
    no_rows = len(data_dict)
    lines = []
    for i, subtitle in enumerate(data_dict):
        ax = mp.pyplot.subplot(1, no_rows, i + 1)
        mp.pyplot.hold(True)
        if i == 0:
            mp.pyplot.title(title)
        data = data_dict[subtitle]
        data = zscore(data,axis=None)
        nTS = data.shape[1]
        if labels is None: 
            labels = numpy.array(range(nTS)).astype(str)
        lines.append([])
        if special_idx is None:
            for iTS in range(nTS):
                line, = mp.pyplot.plot(time, data[:,iTS]+offset*iTS, 'k', label = labels[iTS])
                lines[i].append(line)
        else:
            mask = numpy.array(range(nTS))
            mask = numpy.delete(mask,special_idx)
            for iTS in special_idx:
                line, = mp.pyplot.plot(time, data[:, iTS]+offset*iTS, 'r', label = labels[iTS])
                lines[i].append(line)
            for iTS in mask:
                line, = mp.pyplot.plot(time, data[:, iTS]+offset*iTS, 'k', label = labels[iTS])
                lines[i].append(line)
        mp.pyplot.ylabel(subtitle)
        ax.set_autoscalex_on(False)
        ax.set_xlim([time[0], time[-1]])
        ax.invert_yaxis()
        if MOUSEHOOVER:
            #datacursor( lines[i], formatter='{label}'.format, bbox=dict(fc='white'), 
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(lines[i], formatter='{label}'.format, bbox=dict(fc='white'), 
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )
    mp.pyplot.xlabel("Time (ms)")

    if save_flag:
        fig = mp.pyplot.gcf() 
        if len(fig.get_label())==0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(" ", "_").replace("\t", "_")
        _save_figure(figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    _check_show(show_flag)


def plot_trajectories(data_dict, special_idx=None, title='State space trajectories', show_flag=SHOW_FLAG,
                    save_flag=False, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name='Trajectories',labels=None,figsize=LARGE_SIZE):
                       
    mp.pyplot.figure(title, figsize=figsize)
    ax = mp.pyplot.subplot(111)
    mp.pyplot.hold(True)
    no_dims = len(data_dict)
    if no_dims>2:
        from mpl_toolkits.mplot3d import Axes3D
        ax = mp.pyplot.subplot(111,projection='3d')
    else:
        ax = mp.pyplot.subplot(111)
    lines = []
    ax_labels=[]
    data=[]
    for i, var in enumerate(data_dict):  
        if i == 0:
            mp.pyplot.title(title)
        ax_labels.append(var)    
        data.append(data_dict[var])
    nTS = data[0].shape[1]
    if labels is None: 
        labels = numpy.array(range(nTS)).astype(str)
    lines.append([])
    if special_idx is None:
        for iTS in range(nTS):
            if no_dims>2:
                line, = mp.pyplot.plot(data[0][:,iTS], data[1][:,iTS], data[2][:,iTS],  'k', alpha=0.3, label = labels[iTS])
            else:
                line, = mp.pyplot.plot(data[0][:,iTS], data[1][:,iTS], 'k', alpha=0.3, label = labels[iTS])
            lines.append(line)
    else:
        mask = numpy.array(range(nTS))
        mask = numpy.delete(mask,special_idx)
        for iTS in special_idx:
            if no_dims>2:
                line, = mp.pyplot.plot(data[0][:,iTS], data[1][:,iTS], data[2][:,iTS], 'r', alpha=0.7, label = labels[iTS])
            else:
                line, = mp.pyplot.plot(data[0][:,iTS], data[1][:,iTS], 'r', alpha=0.7, label = labels[iTS])
            lines.append(line)
        for iTS in mask:
            if no_dims>2:
                line, = mp.pyplot.plot(data[0][:,iTS], data[1][:,iTS], data[2][:,iTS], 'k', alpha=0.3, label = labels[iTS])
            else:
                line, = mp.pyplot.plot(data[0][:,iTS], data[1][:,iTS], 'k', alpha=0.3, label = labels[iTS])
            lines.append(line)
    mp.pyplot.xlabel(ax_labels[0])        
    mp.pyplot.ylabel(ax_labels[1])
    if no_dims>2:
        mp.pyplot.ylabel(ax_labels[2])
    if MOUSEHOOVER:
        #datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'), 
        #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
        HighlightingDataCursor(lines[0], formatter='{label}'.format, bbox=dict(fc='white'), 
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )

    if save_flag:
        fig = mp.pyplot.gcf() 
        if len(fig.get_label())==0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(" ", "_").replace("\t", "_")
        _save_figure(figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
    _check_show(show_flag)


def plot_sim_results(model, hyp, head, res, sensorsSEEG):

    if isinstance(model, EpileptorDP2D):
        plot_timeseries(res['time'], {'x1': res['x1'], 'z(t)': res['z']},
                        hyp.seizure_indices, title=" Simulated TAVG for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
    else:
        plot_timeseries(res['time'], {'LFP(t)': res['lfp'], 'z(t)': res['z']},
                        hyp.seizure_indices, title=" Simulated LFP-z for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x1(t)': res['x1'], 'y1(t)': res['y1']},
                        hyp.seizure_indices, title=" Simulated pop1 for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x2(t)': res['x2'], 'y2(t)': res['y2'], 'g(t)': res['g']}, hyp.seizure_indices,
                        title=" Simulated pop2-g for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        start_plot = int(numpy.round(0.01 * res['hpf'].shape[0]))
        plot_raster(res['time'][start_plot:], {'hpf': res['hpf'][start_plot:, :]}, hyp.seizure_indices,
                    title=" Simulated hfp rasterplot for " + hyp.name, offset=10.0,
                    save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                    labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)

    if isinstance(model, EpileptorDPrealistic):
        plot_timeseries(res['time'], {'1/(1+exp(-10(z-3.03))': 1 / (1 + numpy.exp(-10 * (res['z'] - 3.03))),
                                      'slope': res['slopeTS'], 'Iext2': res['Iext2ts']},
                        hyp.seizure_indices, title=" Simulated controlled parameters for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)
        plot_timeseries(res['time'], {'x0': res['x0ts'], 'Iext1':  res['Iext1ts'] , 'K': res['Kts']},
                        hyp.seizure_indices, title=" Simulated parameters for " + hyp.name,
                        save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                        labels=head.connectivity.region_labels, figsize=VERY_LARGE_SIZE)

    for i in range(len(sensorsSEEG)):
        start_plot = int(numpy.round(0.01*res['seeg'+str(i)].shape[0]))
        plot_raster(res['time'][start_plot:], {'SEEG': res['seeg'+str(i)][start_plot:, :]},
                    title=" Simulated SEEG" + str(i) + " raster plot for " + hyp.name,
                    offset=10.0, save_flag=SAVE_FLAG, show_flag=SHOW_FLAG, figure_dir=FOLDER_FIGURES,
                    labels=sensorsSEEG[i].labels, figsize=VERY_LARGE_SIZE)

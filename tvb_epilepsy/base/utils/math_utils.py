# Some math tools

from itertools import product

import numpy as np
from matplotlib import pyplot

from tvb_epilepsy.base.constants.module_constants import WEIGHTS_NORM_PERCENT, INTERACTIVE_ELBOW_POINT
from tvb_epilepsy.base.utils.log_error_utils import warning, initialize_logger


def weighted_vector_sum(weights, vectors, normalize=True):
    if isinstance(vectors, np.ndarray):
        vectors = list(vectors.T)
    if normalize:
        weights /= np.sum(weights)
    vector_sum = weights[0] * vectors[0]
    for iv in range(1, len(weights)):
        vector_sum += weights[iv] * vectors[iv]
    return np.array(vector_sum)


def normalize_weights(weights, percentile=WEIGHTS_NORM_PERCENT):  # , max_w=1.0
    # Create the normalized connectivity weights:
    if len(weights) > 0:
        normalized_w = np.array(weights)
        # Remove diagonal elements
        n_regions = normalized_w.shape[0]
        normalized_w *= 1 - np.eye(n_regions)
        # Normalize with the 95th percentile
        # if np.max(normalized_w) - max_w > 1e-6:
        normalized_w = np.array(normalized_w / np.percentile(normalized_w, percentile))
        #    else:
        #        normalized_w = np.array(weights)
        # normalized_w[normalized_w > max_w] = max_w
        return normalized_w
    else:
        return np.array([])


def compute_in_degree(weights):
    return np.expand_dims(np.sum(weights, axis=1), 1).T


def compute_projection(locations1, locations2, normalize=95, ceil=True):
    n1 = locations1.shape[0]
    n2 = locations2.shape[0]
    projection = np.zeros((n1, n2))
    dist = np.zeros((n1, n2))
    for i1, i2 in product(range(n1), range(n2)):
        dist[i1, i2] = np.abs(np.sum((locations1[i1, :] - locations2[i2, :]) ** 2))
        projection[i1, i2] = 1 / dist[i1, i2]
    if normalize:
        projection /= np.percentile(projection, normalize)
    if ceil:
        if ceil is True:
            ceil = 1.0
        projection[projection > ceil] = ceil
    return projection


def get_greater_values_array_inds(values, n_vals=1):
    return np.argsort(values)[::-1][:n_vals]


def select_greater_values_array_inds(values, threshold=None, verbose=False):
    if isinstance(threshold, np.float):
        return np.where(values > threshold)[0]
    else:
        if verbose:
            warning("Switching to curve elbow point method since threshold=" + str(threshold))
        elbow_point = curve_elbow_point(values)
        return get_greater_values_array_inds(values, elbow_point + 1)


def curve_elbow_point(vals, interactive=INTERACTIVE_ELBOW_POINT):
    logger = initialize_logger(__name__)
    vals = np.array(vals).flatten()
    if np.any(vals[0:-1] - vals[1:] < 0):
        vals = np.sort(vals)
        vals = vals[::-1]
    cumsum_vals = np.cumsum(vals)
    grad = np.gradient(np.gradient(np.gradient(cumsum_vals)))
    elbow = np.argmax(grad)
    if interactive:
        pyplot.ion()
        fig, ax = pyplot.subplots()
        xdata = range(len(vals))
        lines=[]
        lines.append(ax.plot(xdata, cumsum_vals, 'g*', picker=None, label="values' cumulative sum")[0])
        lines.append(ax.plot(xdata, vals, 'bo', picker=None, label="values in descending order")[0])
        lines.append(ax.plot(elbow, vals[elbow], "rd",
                             label="suggested elbow point (maximum of third central difference)")[0])
        lines.append(ax.plot(elbow, cumsum_vals[elbow], "rd")[0])
        pyplot.legend(handles=lines[:2])

        class MyClickableLines(object):

            def __init__(self, fig, ax, lines):
                self.x = None
                #self.y = None
                self.ax = ax
                title = "Mouse lef-click please to select the elbow point..." + \
                        "\n...or click ENTER to continue accepting our automatic choice in red..."
                self.ax.set_title(title)
                self.lines = lines
                self.fig = fig

            def event_loop(self):
                self.fig.canvas.mpl_connect('button_press_event', self.onclick)
                self.fig.canvas.mpl_connect('key_press_event', self.onkey)
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(timeout=-1)
                return

            def onkey(self, event):
                if event.key == "enter":
                    self.fig.canvas.stop_event_loop()
                return

            def onclick(self, event):
                if event.inaxes != self.lines[0].axes: return
                dist = np.sqrt((self.lines[0].get_xdata() - event.xdata) ** 2.0)  # + (self.lines[0].get_ydata() - event.ydata) ** 2.)
                self.x = np.argmin(dist)
                self.fig.canvas.stop_event_loop()
                return

        click_point = MyClickableLines(fig, ax, lines)
        click_point.event_loop()
        if click_point.x is not None:
            elbow = click_point.x
            logger.info("\nmanual selection: " + str(elbow))
        else:
            logger.info("\nautomatic selection: " + str(elbow))
        return elbow
    else:
        return elbow



import os
from shutil import copyfile, rmtree, copytree
import zipfile
import glob
import h5py
import numpy as np
from scipy.stats import zscore
from matplotlib import pyplot as plt
from tvb_fit.io.h5_reader import H5Reader
from tvb_fit.plot.base_plotter import BasePlotter
from tvb_fit.tvb_epilepsy.base.constants.config import Config


def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        rmtree(to_path)
    copytree(from_path, to_path)


def unzip_folder(zippath, outdir=None):
    zippath = zippath.split(".zip")[0]
    if outdir == None:
        outdir = zippath
    zip_ref = zipfile.ZipFile(zippath+".zip", 'r')
    zip_ref.extractall(outdir)
    zip_ref.close()


def hypo2hypo(orig_hypos_path, orig_head, new_head, orig_gain, new_gain):
    sensors_labels = orig_head.get_sensors_id().labels
    orig_hypos = {}
    new_hypos = {}
    for orig_hyp_path in orig_hypos_path.items():

        new_hyp_path = orig_hyp_path.replace("Head"+orig_head.name, "Head"+new_head.name)
        copy_and_overwrite(orig_hyp_path, new_hyp_path)

        hyp_name = orig_hyp_path.split("/")[-2]

        orig_hypos[hyp_name] = {}
        new_hypos[hyp_name] = {}

        orig_hyp_file = h5py.File(os.path.join(orig_hyp_path, hyp_name), 'r')
        hyp_values = orig_hyp_file["values"][()]
        orig_hyp_file.close()

        orig_hypos[hyp_name]["regions_indices"] = np.where(hyp_values>0)[0]
        orig_hypos[hyp_name]["regions"] = orig_head.connectivity.region_labels[orig_hypos[hyp_name]["regions_indices"]]
        orig_hypos[hyp_name]["sensors_indices"] = orig_hypos[hyp_name]["regions_indices"]
        orig_hypos[hyp_name]["sensors_labels"] = []

        new_hypos[hyp_name]["regions_indices"] = orig_hypos[hyp_name]["regions_indices"]

        for ii, reg_ind in enumerate(orig_hypos[hyp_name]["regions_indices"]):
            sens_ind = np.argmax(orig_gain[:, reg_ind])
            orig_hypos[hyp_name]["sensors_indices"][ii] = sens_ind
            orig_hypos[hyp_name]["sensors_labels"].append(sensors_labels[sens_ind])
            new_hypos[hyp_name]["regions_indices"][ii] = np.argmax(new_gain[sens_ind, :])

        new_hypos[hyp_name]["regions"] = new_head.connectivity.region_labels[new_hypos[hyp_name]["regions_indices"]]
        new_hypos[hyp_name]["sensors_indices"] = orig_hypos[hyp_name]["regions_indices"]
        orig_hypos[hyp_name]["sensors_labels"] = np.array(orig_hypos[hyp_name]["sensors_labels"])
        new_hypos[hyp_name]["sensors_labels"] = orig_hypos[hyp_name]["sensors_labels"]

        new_hyp_file = h5py.File(os.path.join(new_hyp_path, hyp_name), 'r+')
        new_hyp_file.attrs["Number_of_nodes"] = new_head.number_of_regions
        new_hyp_file["values"][()][new_hypos[hyp_name]["regions_indices"]] = \
            hyp_values[orig_hypos[hyp_name]["regions_indices"]]
        orig_hyp_file.close()

    return orig_hypos, new_hypos


if __name__ == "__main__":

    datapath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/data/CC"
    respath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC"
    figspath = os.path.join(respath, 'testing_heads/figs')
    conn_figs_path = os.path.join(figspath, 'connectomes')
    subjects = (np.array(range(0, 30)) + 1).tolist()
    subjects = ["TVB%s" % subject for subject in subjects]

    atlases = ["default", "a2009s"]

    config = Config()
    config.out.FOLDER_FIGURES = figspath
    plotter = BasePlotter(config)
    h5_reader = H5Reader()

    connectomes = {"DK": np.nan*np.ones((len(subjects), 87, 87)), "D": np.nan*np.ones((len(subjects), 167, 167))}
    tracts = {"DK": np.nan * np.ones((len(subjects), 87, 87)), "D": np.nan * np.ones((len(subjects), 167, 167))}
    areas = {"DK": np.nan * np.ones((len(subjects), 87)), "D": np.nan * np.ones((len(subjects), 167))}
    orientations = {"DK": np.nan * np.ones((len(subjects), 87)), "D": np.nan * np.ones((len(subjects), 167))}
    hypos = {}

    for isubject, subject in enumerate(subjects):

        print(subject)

        heads = {}
        gains = {}
        hypos[subject] = {}
        hypospaths = {}
        orig = ""
        new = ""
        for atlas, atlas_suffix, head_suffix, def_nregions in zip([atlases, ["", ".a2009s"], ["DK", "D"], [87, 167]]):

            print(atlas)

            subject_path = os.path.join(respath, subject)


            head_path = os.path.join(subject_path, "Head"+head_suffix)
            tvbpath = os.path.join(subject_path, "tvb")
            atlaspath = os.path.join(tvbpath, atlas)

            heads[head_suffix] = h5_reader.read_head(head_path, head_suffix)

            n_sensors = heads[head_suffix].get_sensors_id().locations.shape[0]
            sensors_filename = os.path.join(head_path, "SensorsSEEG_"+str(n_sensors)+"_distance.h5")
            sensors_file = h5py.File(sensors_filename, 'r+')
            gains[head_suffix] = sensors_file["/gain_matrix"][()]
            sensors_file.close()

            if heads[head_suffix].number_of_regions == def_nregions:
                areas[head_suffix][isubject] = heads[head_suffix].connectivity.areas
                orientations[head_suffix][isubject] = heads[head_suffix].connectivity.orientations
                connectomes[head_suffix][isubject] = heads[head_suffix].connectivity.normalized_weights
                tracts[head_suffix][isubject] = heads[head_suffix].connectivity.tract_lengths

            hypospaths[head_suffix] = glob.glob(head_path + "/*/")
            if len(hypospaths[head_suffix]) > 0:
                orig = head_suffix
            else:
                new = head_suffix
        hypos[subject][orig], hypos[subject][new] = \
            hypo2hypo(hypospaths[orig], heads[orig], heads[new], gains[orig], gains[new])

    connectomes_zscore = {}
    tracts_zscore = {}
    areas_zscore = {}
    orientations_zscore = {}
    for head_suffix in ["DK", "D"]:
        connectomes_zscore[head_suffix] = zscore(connectomes[head_suffix], axis=0)
        tracts_zscore[head_suffix] = zscore(tracts[head_suffix], axis=0)
        areas_zscore[head_suffix] = zscore(areas[head_suffix], axis=0)
        orientations_zscore[head_suffix] = zscore(orientations[head_suffix], axis=0)

    for isubject, subject in enumerate(subjects):
        print(subject)
        subject_path = os.path.join(respath, subject)
        for head_suffix, def_nregions in zip(["DK", "D"], [87, 167]):
            print(head_suffix)
            head_path = os.path.join(subject_path, "Head" + head_suffix)
            heads[head_suffix] = h5_reader.read_head(head_path, head_suffix)
            if heads[head_suffix].number_of_regions == def_nregions:
                regions_ticks = np.array(range(def_nregions))
                fig, axes = plt.subplots(1, 2, figsize=(40, 20))
                axes = np.reshape(axes, (axes.size,))
                axes[0] = plotter.plot_regions2regions(connectomes_zscore[head_suffix][isubject],
                                                       heads[head_suffix].connectivity.region_labels, 121,
                                                       "weights zscore", show_x_labels=True, show_y_labels=True,
                                                       x_ticks=regions_ticks, y_ticks=regions_ticks)[0]
                axes[1] = plotter.plot_regions2regions(tracts_zscore[head_suffix][isubject],
                                                       heads[head_suffix].connectivity.region_labels, 122,
                                                       "tracts zscore", show_x_labels=True, show_y_labels=True,
                                                       x_ticks=regions_ticks, y_ticks=regions_ticks)[0]

                plt.savefig(os.path.join(figspath, subject + "_" + head_suffix + "_conn.png"), orientation='landscape')

                fig, axes = plt.subplots(1, 2, figsize=(20, 40))
                axes = np.reshape(axes, (axes.size,))

import os
from shutil import rmtree, copytree #, copyfile
import zipfile
import glob
import h5py
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from tvb_fit.base.computations.math_utils import normalize_weights
from tvb_fit.io.h5_reader import H5Reader
from tvb_fit.io.h5_writer_base import H5WriterBase as H5Writer
from tvb_fit.plot.base_plotter import BasePlotter


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


def correct_TVB28(hypopath="/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC/TVB28/HeadDK",
                  oldHeadPath="/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC/old/TVB28/Head",
                  newConnPath="/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC/TVB28/tvb/default/connectivity"):
    # Old TVB20 hypothesis is defined on a DK parcellation with only 74 regions.
    # This scipt is meant to map it to the normal one with all 87 regions.
    # The mapping will happen via the region labels
    old_conn = h5py.File(os.path.join(oldHeadPath, "Connectivity.h5"), "r")
    old_regions = old_conn["region_labels"][()]
    old_conn.close()
    unzip_folder(newConnPath)
    new_regions = np.loadtxt(os.path.join(newConnPath, "centers.txt"), usecols=0, dtype="str")
    rmtree(newConnPath)
    for hypo in ["preseeg", "postseeg"]:
        print(hypo)
        hypofile = h5py.File(os.path.join(hypopath, hypo, hypo+".h5"), "r+")
        old_values = hypofile["values"][()]
        old_comments = hypofile["comments"][()]
        del hypofile["values"]
        del hypofile["comments"]
        old_reg_inds = np.where(old_values > 0)[0]
        old_values = old_values[old_reg_inds]
        new_values = np.zeros((new_regions.shape[0], ), dtype=old_values.dtype)
        new_comments = np.zeros((new_regions.shape[0], 2), dtype=old_comments.dtype)
        for iregion in range(new_regions.shape[0]):
            new_comments[iregion, 0] = new_regions[iregion]
        for ihyp, (old_reg_ind, value) in enumerate(zip(old_reg_inds, old_values)):
            print(ihyp)
            region = old_regions[old_reg_ind]
            print("old: %s. %s: %s" % (str(old_reg_ind), region, str(value)))
            new_reg_ind = np.where(region == new_regions)[0][0]
            new_values[new_reg_ind] = value
            print("new: %s. %s: %s" % (str(new_reg_ind), region, str(value)))
            old_comment_ind = np.where(region == old_comments[:, 0])[0][0]
            print("old: %s. %s: %s" %
                    (str(old_comment_ind), old_comments[old_comment_ind, 0], old_comments[old_comment_ind, 1]))
            new_comments[new_reg_ind, 1] = old_comments[old_comment_ind, 1]
            print("new: %s. %s: %s" %
                  (str(new_reg_ind), new_comments[new_reg_ind, 0], new_comments[new_reg_ind, 1]))
        hypofile.create_dataset("values", data=new_values)
        hypofile.create_dataset("comments", data=new_comments)
        hypofile.attrs["Number_of_nodes"] = new_regions.shape[0]
        hypofile.close()


def hypo2hypo(orig_hypos_path, orig_head, new_head, orig_gain, new_gain):
    sensors_labels = orig_head.get_sensors_id().labels
    orig_hypos = OrderedDict()
    new_hypos = OrderedDict()
    for orig_hyp_path in orig_hypos_path:

        new_hyp_path = orig_hyp_path.replace("Head"+orig_head.name, "Head"+new_head.name)
        copy_and_overwrite(orig_hyp_path, new_hyp_path)

        hyp_name = orig_hyp_path.split("/")[-2]

        orig_hypos[hyp_name] = OrderedDict()
        new_hypos[hyp_name] = OrderedDict()

        orig_hyp_file = h5py.File(os.path.join(orig_hyp_path, hyp_name+'.h5'), 'r+')
        hyp_values = orig_hyp_file["values"][()]
        comments_unsort = orig_hyp_file["comments"][()]
        comments = np.array(comments_unsort).astype(comments_unsort.dtype)
        for comment in comments_unsort:
            ind = np.where(comment[0] == orig_head.connectivity.region_labels)[0]
            comments[ind] = comment
        del orig_hyp_file["comments"]
        orig_hyp_file.create_dataset("comments", data=comments)
        orig_hyp_file.close()

        orig_reg_inds = np.where(hyp_values > 0)[0]
        hyp_values = hyp_values[orig_reg_inds]
        sort_inds = np.argsort(hyp_values)[::-1]
        orig_reg_inds = orig_reg_inds[sort_inds]
        hyp_values = hyp_values[sort_inds]
        orig_hypos[hyp_name]["regions_indices"] = np.array(orig_reg_inds)
        orig_hypos[hyp_name]["values"] = np.array(hyp_values)
        orig_hypos[hyp_name]["regions"] = \
            np.array(orig_head.connectivity.region_labels[orig_hypos[hyp_name]["regions_indices"]])
        orig_hypos[hyp_name]["sensors_indices"] = np.array(orig_hypos[hyp_name]["regions_indices"])
        orig_hypos[hyp_name]["sensors"] = []
        orig_hypos[hyp_name]["comments"] = []

        new_hypos[hyp_name]["regions_indices"] = np.array(orig_hypos[hyp_name]["regions_indices"])
        new_hypos[hyp_name]["regions"] = np.array(orig_hypos[hyp_name]["regions"])
        new_hypos[hyp_name]["values"] = np.array(orig_hypos[hyp_name]["values"])
        new_hypos[hyp_name]["sensors_indices"] = np.array(orig_hypos[hyp_name]["sensors_indices"])
        new_hypos[hyp_name]["sensors"] = []
        new_hypos[hyp_name]["comments"] = []

        new_comments = np.array(["" for _ in range(new_head.number_of_regions)]).astype(comments.dtype)
        new_comments = \
            np.concatenate([np.array(new_head.connectivity.region_labels).astype(comments.dtype)[:, np.newaxis],
                            new_comments[:, np.newaxis]], axis=1)

        for ii, reg_ind in enumerate(orig_hypos[hyp_name]["regions_indices"]):
            sens_ind = np.argmax(orig_gain[:, reg_ind])
            orig_hypos[hyp_name]["sensors_indices"][ii] = sens_ind
            orig_hypos[hyp_name]["sensors"].append(sensors_labels[sens_ind])
            new_hypos[hyp_name]["regions_indices"][ii] = \
                np.argmin(np.sum(np.abs(orig_head.connectivity.centres[reg_ind]-new_head.connectivity.centres), axis=1))
            new_hypos[hyp_name]["regions"][ii] = \
                new_head.connectivity.region_labels[new_hypos[hyp_name]["regions_indices"][ii]]
            new_hypos[hyp_name]["sensors_indices"][ii] = \
                np.argmax(new_gain[:, new_hypos[hyp_name]["regions_indices"][ii]])
            new_hypos[hyp_name]["sensors"].append(sensors_labels[new_hypos[hyp_name]["sensors_indices"][ii]])
            new_comments[new_hypos[hyp_name]["regions_indices"][ii]] = comments[reg_ind]
            orig_hypos[hyp_name]["comments"].append(comments[reg_ind])
            new_hypos[hyp_name]["comments"].append(comments[reg_ind])
            if orig_hypos[hyp_name]["sensors_indices"][ii] != new_hypos[hyp_name]["sensors_indices"][ii]:
                print("Hypothesis (%s) region (DK: %d. %s, D: %s. %s) linked to different sensors "
                        "\n for DD (%d. %s) \n and D parcellations (%d. %s)!" %
                        (hyp_name, reg_ind, orig_hypos[hyp_name]["regions"][ii],
                         new_hypos[hyp_name]["regions_indices"][ii], new_hypos[hyp_name]["regions"][ii],
                         sens_ind, orig_hypos[hyp_name]["sensors"][ii],
                         new_hypos[hyp_name]["sensors_indices"][ii], new_hypos[hyp_name]["sensors"][ii]))
        orig_hypos[hyp_name]["sensors"] = np.array(orig_hypos[hyp_name]["sensors"])
        new_hypos[hyp_name]["sensors"] = np.array(new_hypos[hyp_name]["sensors"])
        orig_hypos[hyp_name]["comments"] = np.array(orig_hypos[hyp_name]["comments"])
        new_hypos[hyp_name]["comments"] = np.array(new_hypos[hyp_name]["comments"])

        if len(np.unique(new_hypos[hyp_name]["regions_indices"])) < len(hyp_values):
            print("New hypothesis (%s) for parcellation %s leading to less regions:"
                    "\n %s \nthan the original one for parcellation %s: %s" %
                    (hyp_name, new_head.name,
                     str(["%d. %s" % (ind, name) for (ind, name) in
                          zip(new_hypos[hyp_name]["regions_indices"], new_hypos[hyp_name]["regions"])]),
                     orig_head.name, str(["%d. %s" % (ind, name) for (ind, name) in
                          zip(orig_hypos[hyp_name]["regions_indices"], orig_hypos[hyp_name]["regions"])])))

        new_hyp_file = h5py.File(os.path.join(new_hyp_path, hyp_name+'.h5'), 'r+')
        new_hyp_file.attrs["Number_of_nodes"] = new_head.number_of_regions
        del new_hyp_file["values"]
        del new_hyp_file["comments"]
        new_hyp_file.create_dataset("comments", data=new_comments)
        new_values = np.zeros((new_head.number_of_regions, ))
        new_values[new_hypos[hyp_name]["regions_indices"]] = np.array(hyp_values)
        new_hyp_file.create_dataset("values", data=new_values)
        new_hyp_file.close()

    return orig_hypos, new_hypos


def hypos_DK_to_D_mapping(hypos, centers):
    new_hypos = OrderedDict()
    for hyp_name, hypoDK in hypos["DK"].items():
        new_hypos[hyp_name] = OrderedDict()
        if len(np.unique(hypos["DK"][hyp_name]["regions_indices"])) != len(hypos["DK"][hyp_name]["regions_indices"]) \
            or len(np.unique(hypos["D"][hyp_name]["regions_indices"])) != len(hypos["D"][hyp_name]["regions_indices"]):
            print("WTF?")
        for attr in hypos["D"][hyp_name].keys():
            new_hypos[hyp_name][attr] = []
        for iiDK, reg_ind_DK in enumerate(hypoDK["regions_indices"]):
            sort_inds = np.argsort(np.sum(np.abs(centers["DK"][reg_ind_DK]-centers["D"]), axis=1))
            iiD = -1
            ii = 0
            while iiD < 0 and ii < len(sort_inds):
                reg_ind_D = sort_inds[ii]
                try:
                    iiD = np.where(hypos["D"][hyp_name]["regions_indices"] == reg_ind_D)[0][0]
                except:
                    ii = ii + 1
            if iiD == -1:
                print("WTF?")
            else:
                for attr in new_hypos[hyp_name].keys():
                    new_hypos[hyp_name][attr].append(hypos["D"][hyp_name][attr][iiD])
        for attr in new_hypos[hyp_name].keys():
            new_hypos[hyp_name][attr] = np.array(new_hypos[hyp_name][attr])

        if len(new_hypos[hyp_name]) != 6:
            print("WTF?")

    hypos["D"] = new_hypos
    return hypos


def read_hypo(hypos_paths, heads, gains):

    sensors_labels = heads["DK"].get_sensors_id().labels
    hypos = {"DK": OrderedDict(), "D": OrderedDict()}

    centers = OrderedDict()

    for head_suffix in ["DK", "D"]:
        centers[head_suffix] = heads[head_suffix].connectivity.centres

        for hyp_path in hypos_paths[head_suffix]:
            hyp_name = hyp_path.split("/")[-2]
            hypos[head_suffix][hyp_name] = OrderedDict()
            hyp_file = h5py.File(os.path.join(hyp_path, hyp_name + '.h5'), 'r')
            hyp_values = hyp_file["values"][()]
            comments = hyp_file["comments"][()]
            hyp_file.close()
            reg_inds = np.where(hyp_values > 0)[0]
            hyp_values = hyp_values[reg_inds]
            sort_inds = np.argsort(hyp_values)[::-1]
            reg_inds = reg_inds[sort_inds]
            hyp_values = hyp_values[sort_inds]
            hypos[head_suffix][hyp_name]["regions_indices"] = np.array(reg_inds)
            hypos[head_suffix][hyp_name]["values"] = np.array(hyp_values)
            hypos[head_suffix][hyp_name]["regions"] = \
                np.array(heads[head_suffix].connectivity.region_labels[hypos[head_suffix][hyp_name]["regions_indices"]])
            hypos[head_suffix][hyp_name]["sensors_indices"] = np.array(hypos[head_suffix][hyp_name]["regions_indices"])
            hypos[head_suffix][hyp_name]["sensors"] = []
            hypos[head_suffix][hyp_name]["comments"] = []

            for ii, reg_ind in enumerate(hypos[head_suffix][hyp_name]["regions_indices"]):
                sens_ind = np.argmax(gains[head_suffix][:, reg_ind])
                hypos[head_suffix][hyp_name]["sensors_indices"][ii] = sens_ind
                hypos[head_suffix][hyp_name]["sensors"].append(sensors_labels[sens_ind])
                hypos[head_suffix][hyp_name]["comments"].append(comments[reg_ind])
            hypos[head_suffix][hyp_name]["sensors"] = np.array(hypos[head_suffix][hyp_name]["sensors"])
            hypos[head_suffix][hyp_name]["comments"] = np.array(hypos[head_suffix][hyp_name]["comments"])

            if len(hypos[head_suffix][hyp_name]) != 6:
                print("WTF?")
    return hypos_DK_to_D_mapping(hypos, centers)


def correct_head_size(head, areas, def_regions_nr=87, missing_region_ind=4):
    new_areas = \
        np.concatenate([areas[:missing_region_ind], np.zeros((1,), dtype=areas.dtype), areas[missing_region_ind:]])
    # old_orientations = head.connectivity.orientations
    # new_orientations = np.concatenate([old_orientations[:missing_region_ind], np.zeros((1,3), dtype=areas.dtype),
    #                                    old_orientations[missing_region_ind:]], axis=0)
    old_connectome = normalize_weights(head.connectivity.weights, percentile=99, ceil=False)
    new_connectome = \
        np.concatenate([old_connectome[:missing_region_ind], np.zeros((1, def_regions_nr-1), dtype=areas.dtype),
                                     old_connectome[missing_region_ind:]], axis=0)
    new_connectome = \
        np.concatenate([new_connectome[:, :missing_region_ind], np.zeros((def_regions_nr, 1), dtype=areas.dtype),
         new_connectome[:, missing_region_ind:]], axis=1)
    old_tracts = head.connectivity.tract_lengths
    new_tracts = np.concatenate(
        [old_tracts[:missing_region_ind], np.zeros((1, def_regions_nr-1), dtype=areas.dtype),
         old_tracts[missing_region_ind:]], axis=0)
    new_tracts = \
        np.concatenate([new_tracts[:, :missing_region_ind], np.zeros((def_regions_nr, 1), dtype=areas.dtype),
                        new_tracts[:, missing_region_ind:]], axis=1)
    return new_connectome, new_tracts, new_areas # , new_orientations


def read_hypos(hypos_path, head_suffixes, subjects=None):
    hypos = OrderedDict()
    hypos_file = h5py.File(hypos_path, "r")
    if subjects is None:
        subjects = hypos_file.keys()
    for subject in subjects:
        hypos[subject] = OrderedDict()
        if hypos_file.get(subject, None) is not None:
            for head_suffix in head_suffixes:
                hypos[subject][head_suffix] = OrderedDict()
                subj_head_suffix = os.path.join(subject, head_suffix)
                for hyp_name in hypos_file[subj_head_suffix].keys():
                    subj_head_suffix_hyp = os.path.join(subj_head_suffix, hyp_name)
                    hypos[subject][head_suffix][hyp_name] = OrderedDict()
                    for attr in hypos_file[subj_head_suffix_hyp].keys():
                        hypos[subject][head_suffix][hyp_name][attr] = \
                            np.array(hypos_file[os.path.join(subj_head_suffix_hyp, attr)][()])

    hypos_file.close()
    return hypos


def read_head_stats(head_stats_path, head_suffixes):
    head_stats = OrderedDict()
    head_stats_file = h5py.File(head_stats_path, "r")
    for head_suffix in head_suffixes:
        head_stats[head_suffix] = OrderedDict()
        for attr in head_stats_file[head_suffix].keys():
            head_stats[head_suffix][attr] = np.array(head_stats_file[os.path.join(head_suffix, attr)][()])
    head_stats_file.close()
    return head_stats


if __name__ == "__main__":

    datapath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/data/CC"
    respath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC"
    testheads_path = os.path.join(respath, 'testing_heads')
    figspath = os.path.join(testheads_path, 'figs')
    conn_figs_path = os.path.join(figspath, 'connectomes')
    hypos_path = os.path.join(testheads_path, "hypos.h5")
    head_stats_path = os.path.join(testheads_path, "head_stats.h5")
    # orientations_figs_path = os.path.join(figspath, 'orientations')

    READ_HYPOS_FLAG = False
    READ_HEAD_STATS_FLAG = False

    subjects = (np.array(range(0, 30)) + 1).tolist()
    subjects = ["TVB%s" % subject for subject in subjects]

    atlases = ["default", "a2009s"]
    head_suffixes = ["DK", "D"]
    atlas_def_nregions = [87, 167]

    plotter = BasePlotter()
    h5_reader = H5Reader()
    h5_writer = H5Writer()

    if READ_HYPOS_FLAG:
        try:
            hypos = read_hypos(hypos_path, head_suffixes, subjects)
        except:
            READ_HYPOS_FLAG = False
    if not READ_HYPOS_FLAG:
        hypos = OrderedDict()

    if READ_HEAD_STATS_FLAG:
        try:
            head_stats = read_head_stats(head_stats_path, head_suffixes)
        except:
            READ_HEAD_STATS_FLAG = False
    if not READ_HEAD_STATS_FLAG:
        connectomes = {"DK": np.nan*np.ones((len(subjects), 87, 87)), "D": np.nan*np.ones((len(subjects), 167, 167))}
        tracts = {"DK": np.nan * np.ones((len(subjects), 87, 87)), "D": np.nan * np.ones((len(subjects), 167, 167))}
        areas = {"DK": np.nan * np.ones((len(subjects), 87)), "D": np.nan * np.ones((len(subjects), 167))}
        # orientations = {"DK": np.nan * np.ones((len(subjects), 87, 3)), "D": np.nan * np.ones((len(subjects), 167, 3))}
        head_stats = OrderedDict()

    region_labels = OrderedDict()
    for head_suffix in head_suffixes:
        head = h5_reader.read_head(os.path.join(respath, "TVB1", "Head"+head_suffix), head_suffix)
        region_labels[head_suffix] = head.connectivity.region_labels

    hypos_string = ""

    for isubject, subject in enumerate(subjects):

        print(subject)

        if not READ_HYPOS_FLAG or not READ_HEAD_STATS_FLAG:
            heads = OrderedDict()
            gains = OrderedDict()
            if not (READ_HYPOS_FLAG):
                hypos[subject] = OrderedDict()
                hypospaths = OrderedDict()
                orig = ""
                new = ""
        for atlas, atlas_suffix, head_suffix, def_nregions in \
                zip(atlases, ["", ".a2009s"], head_suffixes, atlas_def_nregions):

            print(atlas)

            subject_path = os.path.join(respath, subject)

            head_path = os.path.join(subject_path, "Head"+head_suffix)
            tvbpath = os.path.join(subject_path, "tvb")
            atlaspath = os.path.join(tvbpath, atlas)
            conn_path = os.path.join(atlaspath, "connectivity")

            if not READ_HYPOS_FLAG or not READ_HEAD_STATS_FLAG:

                heads[head_suffix] = h5_reader.read_head(head_path, head_suffix)

                n_sensors = heads[head_suffix].get_sensors_id().locations.shape[0]
                try:
                    sensors_filename = os.path.join(head_path, "SensorsSEEG_"+str(n_sensors)+"_distance.h5")
                    sensors_file = h5py.File(sensors_filename, 'r+')
                    gains[head_suffix] = sensors_file["/gain_matrix"][()]
                    sensors_file.close()
                except:
                    gains[head_suffix] = heads[head_suffix].get_sensors_id().gain_matrix

                unzip_folder(conn_path)
                this_areas = np.genfromtxt(os.path.join(conn_path, "areas.txt"))
                rmtree(conn_path)
                if heads[head_suffix].number_of_regions != def_nregions:
                    # for TVB25 DK missing enthorinal cortex
                    connectomes[head_suffix][isubject], tracts[head_suffix][isubject], \
                    areas[head_suffix][isubject] = \
                        correct_head_size(heads[head_suffix], this_areas,
                                          def_regions_nr=87, missing_region_ind=4)
                    #, orientations[head_suffix][isubject] = \
                else:
                    connectomes[head_suffix][isubject] = normalize_weights(heads[head_suffix].connectivity.weights,
                                                                           percentile=99, ceil=False)
                    tracts[head_suffix][isubject] = heads[head_suffix].connectivity.tract_lengths
                    areas[head_suffix][isubject] = this_areas
                    # orientations[head_suffix][isubject] = heads[head_suffix].connectivity.orientations

                if not READ_HYPOS_FLAG:
                    hypospaths[head_suffix] = glob.glob(head_path + "/*/")
                    if len(hypospaths[head_suffix]) > 0:
                        orig = head_suffix
                    else:
                        new = head_suffix
        if not READ_HYPOS_FLAG:
            if len(orig) > 0 and len(new) > 0:
                # try:
                print("original parcellation is %s" % orig)
                hypos[subject][orig], hypos[subject][new] = \
                    hypo2hypo(hypospaths[orig], heads[orig], heads[new], gains[orig], gains[new])

            # except:
            #     print("Subject %s hypotheses' processing failed!")
            #     pass
            elif len(orig) > 0:
                # try:
                hypos[subject] = read_hypo(hypospaths, heads, gains)
                #except:
                #   print("Subject %s hypotheses' reading failed!")
                #   pass
        if len(hypos[subject]) == 2:
            this_hypos_string = '\n\n--------------------------------------------------------------------------------\n'
            this_hypos_string += '--------------------------------------------------------------------------------\n'
            this_hypos_string += "%s\n" % subject
            for hypo_name, hypo in hypos[subject]["DK"].items():
                this_hypos_string += "\nhypothesis: %s\n\n" % hypo_name
                for ii in range(len(hypos[subject]["DK"][hypo_name]["values"])):
                    if np.any(hypos[subject]["DK"][hypo_name]["values"][ii] >
                              hypos[subject]["DK"][hypo_name]["values"][:ii]) or \
                            np.any(hypos[subject]["D"][hypo_name]["values"][ii] >
                                   hypos[subject]["D"][hypo_name]["values"][:ii]):
                        print("WTF?")
                    this_hypos_string += "DK       : Region: %s. %s, Sensor: %s. %s" \
                                    "\nDestrieux: Region: %s. %s, Sensor: %s. %s" \
                                    "\nvalue = %s, comment: %s \n\n" % \
                                    (hypos[subject]["DK"][hypo_name]['regions_indices'][ii],
                                     hypos[subject]["DK"][hypo_name]['regions'][ii],
                                     hypos[subject]["DK"][hypo_name]['sensors_indices'][ii],
                                     hypos[subject]["DK"][hypo_name]['sensors'][ii],
                                     hypos[subject]["D"][hypo_name]['regions_indices'][ii],
                                     hypos[subject]["D"][hypo_name]['regions'][ii],
                                     hypos[subject]["D"][hypo_name]['sensors_indices'][ii],
                                     hypos[subject]["D"][hypo_name]['sensors'][ii],
                                     hypos[subject]["DK"][hypo_name]['values'][ii],
                                     hypos[subject]["DK"][hypo_name]['comments'][ii, 1])
                this_hypos_string += '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n'
            print(this_hypos_string)
            hypos_string += this_hypos_string

    with open(os.path.join(testheads_path, "hypos.txt"), "w") as text_file:
        text_file.write(hypos_string)
    if not(READ_HYPOS_FLAG):
        h5_writer.write_object_to_file(os.path.join(testheads_path, "hypos.h5"), hypos)

    if not READ_HEAD_STATS_FLAG :
        for head_suffix in ["DK", "D"]:
            head_stats[head_suffix] = {"connectomes_mean": np.nanmean(connectomes[head_suffix], axis=0)}
            head_stats[head_suffix].update({"tracts_mean": np.nanmean(tracts[head_suffix], axis=0)})
            head_stats[head_suffix].update({"areas_mean": np.nanmean(areas[head_suffix], axis=0)})
            # head_stats[head_suffix].update({"orientations_mean": np.nanmean(orientations[head_suffix], axis=0)})
            head_stats[head_suffix].update({"connectomes_std": np.nanstd(connectomes[head_suffix], axis=0)})
            head_stats[head_suffix].update({"tracts_std": np.nanstd(tracts[head_suffix], axis=0)})
            head_stats[head_suffix].update({"areas_std": np.nanstd(areas[head_suffix], axis=0)})
            # head_stats[head_suffix].update({"orientations_std": np.nanstd(orientations[head_suffix], axis=0)})

            head_stats[head_suffix].update({"connectomes_zscore": np.zeros(connectomes[head_suffix].shape)})
            head_stats[head_suffix].update({"tracts_zscore": np.zeros(tracts[head_suffix].shape)})
            head_stats[head_suffix].update({"areas_zscore": np.zeros(areas[head_suffix].shape)})
        # head_stats[head_suffix].update({"orientations_zscore": np.zeros(orientations[head_suffix].shape)})

    for isubject, subject in enumerate(subjects):
        print(subject)

        subject_path = os.path.join(respath, subject)
        for head_suffix, def_nregions in zip(head_suffixes, atlas_def_nregions):
            print(head_suffix)
            if not READ_HEAD_STATS_FLAG:
                head_stats[head_suffix]["connectomes_zscore"][isubject] = \
                                (connectomes[head_suffix][isubject] - head_stats[head_suffix]["connectomes_mean"]) / \
                                        head_stats[head_suffix]["connectomes_std"]
                head_stats[head_suffix]["tracts_zscore"][isubject] = \
                                        (tracts[head_suffix][isubject] - head_stats[head_suffix][ "tracts_mean"]) / \
                                                    head_stats[head_suffix]["tracts_std"]
                head_stats[head_suffix]["areas_zscore"][isubject] = \
                                            (areas[head_suffix][isubject] - head_stats[head_suffix]["areas_mean"]) / \
                                                        head_stats[head_suffix]["areas_std"]
                # head_stats[head_suffix]["orientations_zscore"][isubject] = \
                #              (orientations[head_suffix][isubject] - head_stats[head_suffix]["orientations_mean"]) / \
                #                                     head_stats[head_suffix]["orientations_std"]

            head_path = os.path.join(subject_path, "Head" + head_suffix)
            if head_stats[head_suffix]["connectomes_zscore"][isubject].squeeze().shape[0] == def_nregions:
                regions_ticks = np.array(range(def_nregions))
                fig, axes = plt.subplots(1, 2, figsize=(60, 30))
                axes = np.reshape(axes, (axes.size,))
                axes[0] = plotter.plot_regions2regions(head_stats[head_suffix]["connectomes_zscore"][isubject].squeeze(),
                                                       region_labels[head_suffix], 121,
                                                       "weights zscore", show_x_labels=True, show_y_labels=True,
                                                       x_ticks=regions_ticks, y_ticks=regions_ticks,
                                                       cmap="jet", vmin=-3, vmax=+3)[0]
                axes[1] = plotter.plot_regions2regions(head_stats[head_suffix]["tracts_zscore"][isubject].squeeze(),
                                                       region_labels[head_suffix], 122,
                                                       "tracts zscore", show_x_labels=True, show_y_labels=True,
                                                       x_ticks=regions_ticks, y_ticks=regions_ticks,
                                                       cmap="jet", vmin=-3, vmax=+3)[0]

                plt.savefig(os.path.join(conn_figs_path, subject + "_" + head_suffix + "_conn_zscore.png"),
                            orientation='landscape')
            else:
                print("WTF?")

    if not READ_HEAD_STATS_FLAG :
        h5_writer.write_object_to_file(os.path.join(testheads_path, "head_stats.h5"), head_stats)

    for head_suffix, def_nregions in zip(head_suffixes, atlas_def_nregions):
        this_areas_zscore = np.array(head_stats[head_suffix]["areas_zscore"])
        this_areas_zscore[np.isnan(this_areas_zscore)] = 0.0
        fig, axes = plt.subplots(1, 1, figsize=(60, 30))
        for isubject, subject in enumerate(subjects):
            subj_id = isubject+1
            for reg_ind in range(def_nregions):
                axes.text(float(reg_ind), float(this_areas_zscore[isubject, reg_ind]), str(subj_id))
        x = np.array(range(def_nregions))
        axes.plot(x, 3 * np.ones(x.shape), 'r')
        axes.plot(x, -3 * np.ones(x.shape), 'r')
        axes.set_ylim(-6, 6)
        axes.set_xticks(x )
        axes.set_xticklabels(region_labels[head_suffix], rotation=90)
        plt.savefig(os.path.join(figspath, "areas_zscore_%s.png" % head_suffix), orientation='landscape')

        # for isubject, subject in enumerate(subjects):
        #     fig, axes = plt.subplots(1, 1, figsize=(60, 30))
        #     img = axes.imshow(head_stats[head_suffix]["orientations_zscore"][isubject].squeeze().T,
        #                       cmap="jet", vmin=-3, vmax=3)
        #     axes.set_xticks(np.array(range(def_nregions)))
        #     axes.set_xticklabels(region_labels[head_suffix], rotation=90)
        #     plt.colorbar(img)
        #     plt.savefig(os.path.join(orientations_figs_path, "%s_%s_orientations_zscore.png" % (subject, head_suffix)),
        #                 orientation='landscape')

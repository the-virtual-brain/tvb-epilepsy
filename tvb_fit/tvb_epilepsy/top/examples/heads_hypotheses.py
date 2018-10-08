import os
from shutil import rmtree, copytree #, copyfile
import zipfile
import glob
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tvb_fit.base.utils.log_error_utils import warning
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


def hypo2hypo(orig_hypos_path, orig_head, new_head, orig_gain, new_gain):
    sensors_labels = orig_head.get_sensors_id().labels
    orig_hypos = {}
    new_hypos = {}
    for orig_hyp_path in orig_hypos_path:

        new_hyp_path = orig_hyp_path.replace("Head"+orig_head.name, "Head"+new_head.name)
        copy_and_overwrite(orig_hyp_path, new_hyp_path)

        hyp_name = orig_hyp_path.split("/")[-2]

        orig_hypos[hyp_name] = {}
        new_hypos[hyp_name] = {}

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
        orig_hypos[hyp_name]["regions_indices"] = np.array(orig_reg_inds)
        hyp_values = hyp_values[orig_reg_inds]
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
        orig_hyp_file.close()

    return orig_hypos, new_hypos


def read_hypo(hypos_path, head, gain):
    sensors_labels = head.get_sensors_id().labels
    hypos = {}

    for hyp_path in hypos_path:
        hyp_name = hyp_path.split("/")[-2]
        hypos[hyp_name] = {}
        hyp_file = h5py.File(os.path.join(hyp_path, hyp_name + '.h5'), 'r+')
        hyp_values = hyp_file["values"][()]
        comments = hyp_file["comments"][()]
        hyp_file.close()
        reg_inds = np.where(hyp_values > 0)[0]
        hypos[hyp_name]["regions_indices"] = np.array(reg_inds)
        hyp_values = hyp_values[reg_inds]
        hypos[hyp_name]["values"] = np.array(hyp_values)
        hypos[hyp_name]["regions"] = \
            np.array(head.connectivity.region_labels[hypos[hyp_name]["regions_indices"]])
        hypos[hyp_name]["sensors_indices"] = np.array(hypos[hyp_name]["regions_indices"])
        hypos[hyp_name]["sensors"] = []
        hypos[hyp_name]["comments"] = []

        for ii, reg_ind in enumerate(hypos[hyp_name]["regions_indices"]):
            sens_ind = np.argmax(gain[:, reg_ind])
            hypos[hyp_name]["sensors_indices"][ii] = sens_ind
            hypos[hyp_name]["sensors"].append(sensors_labels[sens_ind])
            hypos[hyp_name]["comments"].append(comments[reg_ind])
        hypos[hyp_name]["sensors"] = np.array(hypos[hyp_name]["sensors"])
        hypos[hyp_name]["comments"] = np.array(hypos[hyp_name]["comments"])

    return hypos

if __name__ == "__main__":

    datapath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/data/CC"
    respath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC"
    testheads_path = os.path.join(respath, 'testing_heads')
    figspath = os.path.join(testheads_path, 'figs')
    conn_figs_path = os.path.join(figspath, 'connectomes')
    orientations_figs_path = os.path.join(figspath, 'orientations')
    subjects = (np.array(range(0, 30)) + 1).tolist()
    subjects = ["TVB%s" % subject for subject in subjects]

    atlases = ["default", "a2009s"]

    plotter = BasePlotter()
    h5_reader = H5Reader()
    h5_writer = H5Writer()

    connectomes = {"DK": np.nan*np.ones((len(subjects), 87, 87)), "D": np.nan*np.ones((len(subjects), 167, 167))}
    tracts = {"DK": np.nan * np.ones((len(subjects), 87, 87)), "D": np.nan * np.ones((len(subjects), 167, 167))}
    areas = {"DK": np.nan * np.ones((len(subjects), 87)), "D": np.nan * np.ones((len(subjects), 167))}
    orientations = {"DK": np.nan * np.ones((len(subjects), 87, 3)), "D": np.nan * np.ones((len(subjects), 167, 3))}
    hypos = {}
    head_stats = {}

    hypos_string = ""
    for isubject, subject in enumerate(subjects):

        print(subject)

        heads = {}
        gains = {}
        hypos[subject] = {}
        hypospaths = {}
        orig = ""
        new = ""
        for atlas, atlas_suffix, head_suffix, def_nregions in zip(atlases, ["", ".a2009s"], ["DK", "D"], [87, 167]):

            print(atlas)

            subject_path = os.path.join(respath, subject)

            head_path = os.path.join(subject_path, "Head"+head_suffix)
            tvbpath = os.path.join(subject_path, "tvb")
            atlaspath = os.path.join(tvbpath, atlas)
            conn_path = os.path.join(atlaspath, "connectivity")
            heads[head_suffix] = h5_reader.read_head(head_path, head_suffix)

            n_sensors = heads[head_suffix].get_sensors_id().locations.shape[0]
            try:
                sensors_filename = os.path.join(head_path, "SensorsSEEG_"+str(n_sensors)+"_distance.h5")
                sensors_file = h5py.File(sensors_filename, 'r+')
                gains[head_suffix] = sensors_file["/gain_matrix"][()]
                sensors_file.close()
            except:
                gains[head_suffix] = heads[head_suffix].get_sensors_id().gain_matrix

            if heads[head_suffix].number_of_regions == def_nregions:
                unzip_folder(conn_path)
                areas[head_suffix][isubject] = np.genfromtxt(os.path.join(conn_path, "areas.txt"))
                rmtree(conn_path)
                orientations[head_suffix][isubject] = heads[head_suffix].connectivity.orientations
                connectomes[head_suffix][isubject] = heads[head_suffix].connectivity.normalized_weights
                tracts[head_suffix][isubject] = heads[head_suffix].connectivity.tract_lengths

            hypospaths[head_suffix] = glob.glob(head_path + "/*/")
            if len(hypospaths[head_suffix]) > 0:
                orig = head_suffix
            else:
                new = head_suffix
        if len(orig) > 0 and len(new) > 0:
            try:
                print("original parcellation is %s" % orig)
                hypos[subject][orig], hypos[subject][new] = \
                        hypo2hypo(hypospaths[orig], heads[orig], heads[new], gains[orig], gains[new])

            except:
                print("Subject %s hypotheses' processing failed!")
                pass
        elif len(orig) > 0:
            try:
                for head_suffix in ["DK", "D"]:
                    hypos[subject][head_suffix] = \
                        read_hypo(hypospaths[head_suffix], heads[head_suffix], gains[head_suffix])
            except:
                print("Subject %s hypotheses' processing failed!")
                pass
        if len(hypos[subject]["DK"]) > 0 and len(hypos[subject]["D"]) > 0:
            try:
                hypos_string += '\n\n------------------------------------------------------------------------------------\n'
                hypos_string += '------------------------------------------------------------------------------------\n'
                hypos_string += "%s\n" % subject
                for hypo_name, hypo in hypos[subject]["DK"].items():
                    hypos_string += "\nhypothesis: %s\n\n" % hypo_name
                    for ii in range(len(hypos[subject]["DK"][hypo_name]["values"])):
                        hypos_string += "DK       : Region: %s. %s, Sensor: %s. %s" \
                                        "\nDestrieux: Region: %s. %s, Sensor: %s. %s" \
                                        "\nvalue = %s, comment : %s \n\n" % \
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
                    hypos_string += '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - -\n'
            except:
                print("Subject %s hypotheses' printing failed!")

    with open(os.path.join(testheads_path, "hypos.txt"), "w") as text_file:
        text_file.write(hypos_string)
    h5_writer.write_object_to_file(os.path.join(testheads_path, "hypos.h5"), hypos)

    for head_suffix in ["DK", "D"]:
        head_stats[head_suffix] = {"connectomes_mean": np.nanmean(connectomes[head_suffix], axis=0)}
        head_stats[head_suffix].update({"tracts_mean": np.nanmean(tracts[head_suffix], axis=0)})
        head_stats[head_suffix].update({"areas_mean": np.nanmean(areas[head_suffix], axis=0)})
        head_stats[head_suffix].update({"orientations_mean": np.nanmean(orientations[head_suffix], axis=0)})
        head_stats[head_suffix].update({"connectomes_std": np.nanstd(connectomes[head_suffix], axis=0)})
        head_stats[head_suffix].update({"tracts_std": np.nanstd(tracts[head_suffix], axis=0)})
        head_stats[head_suffix].update({"areas_std": np.nanstd(areas[head_suffix], axis=0)})
        head_stats[head_suffix].update({"orientations_std": np.nanstd(orientations[head_suffix], axis=0)})

        head_stats[head_suffix].update({"connectomes_zscore": np.zeros(connectomes[head_suffix].shape)})
        head_stats[head_suffix].update({"tracts_zscore": np.zeros(tracts[head_suffix].shape)})
        head_stats[head_suffix].update({"areas_zscore": np.zeros(areas[head_suffix].shape)})
        head_stats[head_suffix].update({"orientations_zscore": np.zeros(orientations[head_suffix].shape)})

    for isubject, subject in enumerate(subjects):
        print(subject)

        subject_path = os.path.join(respath, subject)
        for head_suffix, def_nregions in zip(["DK", "D"], [87, 167]):
            print(head_suffix)

            head_stats[head_suffix]["connectomes_zscore"][isubject] = \
                                (connectomes[head_suffix][isubject] - head_stats[head_suffix]["connectomes_mean"]) / \
                                    head_stats[head_suffix]["connectomes_std"]
            head_stats[head_suffix]["tracts_zscore"][isubject] = \
                                        (tracts[head_suffix][isubject] - head_stats[head_suffix][ "tracts_mean"]) / \
                                                head_stats[head_suffix]["tracts_std"]
            head_stats[head_suffix]["areas_zscore"][isubject] = \
                                            (areas[head_suffix][isubject] - head_stats[head_suffix]["areas_mean"]) / \
                                                    head_stats[head_suffix]["areas_std"]
            head_stats[head_suffix]["orientations_zscore"][isubject] = \
                                (orientations[head_suffix][isubject] - head_stats[head_suffix]["orientations_mean"]) / \
                                                head_stats[head_suffix]["orientations_std"]

            head_path = os.path.join(subject_path, "Head" + head_suffix)
            heads[head_suffix] = h5_reader.read_head(head_path, head_suffix)
            if heads[head_suffix].number_of_regions == def_nregions:
                regions_ticks = np.array(range(def_nregions))
                fig, axes = plt.subplots(1, 2, figsize=(60, 30))
                axes = np.reshape(axes, (axes.size,))
                axes[0] = plotter.plot_regions2regions(head_stats[head_suffix]["connectomes_zscore"][isubject].squeeze(),
                                                       heads[head_suffix].connectivity.region_labels, 121,
                                                       "weights zscore", show_x_labels=True, show_y_labels=True,
                                                       x_ticks=regions_ticks, y_ticks=regions_ticks, cmap="jet")[0]
                axes[1] = plotter.plot_regions2regions(head_stats[head_suffix]["tracts_zscore"][isubject].squeeze(),
                                                       heads[head_suffix].connectivity.region_labels, 122,
                                                       "tracts zscore", show_x_labels=True, show_y_labels=True,
                                                       x_ticks=regions_ticks, y_ticks=regions_ticks, cmap="jet")[0]

                plt.savefig(os.path.join(conn_figs_path, subject + "_" + head_suffix + "_conn_zscore.png"),
                            orientation='landscape')

    h5_writer.write_object_to_file(os.path.join(testheads_path, "head_stats.h5"), head_stats)

    for head_suffix, def_nregions in zip(["DK", "D"], [87, 167]):
        fig, axes = plt.subplots(1, 1, figsize=(60, 30))
        for isubject, subject in enumerate(subjects):
            subj_id = isubject+1
            for reg_ind in range(def_nregions):
                axes.text(float(reg_ind), float(head_stats[head_suffix]["areas_zscore"][isubject, reg_ind]),
                           str(subj_id))
        x = np.array(range(def_nregions))
        axes.plot(x, 3 * np.ones(x.shape), 'r')
        axes.plot(x, -3 * np.ones(x.shape), 'r')
        axes.set_ylim(-6, 6)
        axes.set_xticks(x )
        axes.set_xticklabels(heads[head_suffix].connectivity.region_labels, rotation=90)
        plt.savefig(os.path.join(figspath, "areas_zscore_%s.png" % head_suffix), orientation='landscape')

        for isubject, subject in enumerate(subjects):
            fig, axes = plt.subplots(1, 1, figsize=(60, 30))
            img = axes.imshow(head_stats[head_suffix]["orientations_zscore"][isubject].squeeze().T,
                              cmap="jet", vmin=-3, vmax=3)
            axes.set_xticks(np.array(range(def_nregions)))
            axes.set_xticklabels(heads[head_suffix].connectivity.region_labels, rotation=90)
            plt.colorbar(img)
            plt.savefig(os.path.join(orientations_figs_path, "%s_%s_orientations_zscore.png" % (subject, head_suffix)),
                        orientation='landscape')

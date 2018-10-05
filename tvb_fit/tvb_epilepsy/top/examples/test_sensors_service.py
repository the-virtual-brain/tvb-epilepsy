import os
from shutil import copyfile, rmtree
import zipfile
import glob
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tvb_fit.base.utils.log_error_utils import warning
from tvb_fit.base.model.virtual_patient.sensors import SensorTypes
from tvb_fit.service.head_service import HeadService
from tvb_fit.io.h5_reader import H5Reader


def unzip_folder(zippath, outdir=None):
    zippath = zippath.split(".zip")[0]
    if outdir == None:
        outdir = zippath
    zip_ref = zipfile.ZipFile(zippath+".zip", 'r')
    zip_ref.extractall(outdir)
    zip_ref.close()


if __name__ == "__main__":

    datapath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/data/CC"
    respath = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC"
    gain_figs_path = os.path.join(respath, 'testing_heads/figs/seeg_gains')
    subjects = (np.array(range(1, 30)) + 1).tolist()
    # del subjects[14]
    subjects = ["TVB%s" % subject for subject in subjects]
    subjects += ["TVB4_mrielec", "TVB10_mrielec"]

    atlases = ["default", "a2009s"]

    head_service = HeadService()
    h5_reader = H5Reader()

    for subject in subjects:

        print(subject)

        for atlas in atlases:

            print(atlas)

            gain_matrices = {}

            subject_path = os.path.join(respath, subject)

            atlas_suffix = str(np.where(atlas == "default", "", ".a2009s"))
            head_suffix = str(np.where(atlas == "default", "DK", "D"))

            head_path = os.path.join(subject_path, "Head"+head_suffix)
            tvbpath = os.path.join(subject_path, "tvb")
            atlaspath = os.path.join(tvbpath, atlas)

            aparcaseg_path = os.path.join(atlaspath, "aparc+aseg-cor.nii.gz")
            os.rename(aparcaseg_path, aparcaseg_path.replace("-cor", ""))

            seeg_path = os.path.join(atlaspath, "seeg_%s_gain.txt")

            head = h5_reader.read_head(head_path, atlas)
            sensors = head.get_sensors_id(s_type=SensorTypes.TYPE_SEEG, sensor_ids=0)

            sensors_file = glob.glob(head_path+"/Sensors*")[0]
            sensors_name = sensors_file.split(".h5")[0]

            dp_sensors_filename = sensors_name + "_dp" + ".h5"
            os.rename(sensors_file, dp_sensors_filename)
            dp_sensors_file = h5py.File(dp_sensors_filename, 'r')
            gain_matrices["dp"] = dp_sensors_file["/gain_matrix"][()]
            dp_sensors_file.close()

            for method in ["dipole", "distance", "regions_distance"]:

                gain_matrices[method] = \
                    head_service.compute_gain_matrix(head, sensors, method, normalize=100)

                gain = np.genfromtxt(seeg_path % method)

                if np.abs(gain - gain_matrices[method]).max() > np.abs(gain).max()/100000:
                    warning("Error with gain matrix!")
                new_sensors_file = sensors_name + "_" + method + ".h5"
                copyfile(dp_sensors_filename, new_sensors_file)
                new_sensors_file = h5py.File(new_sensors_file, 'r+')
                new_sensors_file["/gain_matrix"][()] = gain_matrices[method]
                new_sensors_file.close()

            fig, axes = plt.subplots(2, 3, figsize=(30, 20))
            axes = np.reshape(axes, (axes.size, ))
            iax = -1
            for method in ["dipole", "distance", "regions_distance", "dp"]:

                if method == "dipole":

                    img = axes[0].imshow(gain_matrices[method], cmap="jet", vmin=-1, vmax=1)
                    axes[0].set_title(method)
                    plt.colorbar(img, cax=axes[0])
                    img = axes[1].imshow(np.abs(gain_matrices[method]), cmap="jet", vmin=0, vmax=1)
                    axes[1].set_title("|"+method+"|")
                    # plt.colorbar(img, cax=axes[1])
                    iax = 2

                else:
                    iax +=1
                    img = axes[iax].imshow(gain_matrices[method], cmap="jet", vmin=0, vmax=1)
                    axes[iax].set_title(method)
                    # plt.colorbar(img, cax=axes[iax])

            plt.savefig(os.path.join(gain_figs_path, subject + "_" + head_suffix + ".png"), orientation='landscape')

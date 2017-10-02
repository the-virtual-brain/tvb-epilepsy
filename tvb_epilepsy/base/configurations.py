import os
import platform
from datetime import datetime

USER_HOME = os.path.expanduser("~")
RUN_ENV = "local"
if "RUN_ENV" in os.environ:
    RUN_ENV = os.environ["RUN_ENV"]

if RUN_ENV == "test":
    DATA_TEST = "data"
    FOLDER_LOGS = os.path.join(os.getcwd(), "logs")
    FOLDER_RES = os.path.join(os.getcwd(), "res")
    FOLDER_FIGURES = os.path.join(os.getcwd(), "figs")

else:
    FOLDER_VEP_ONLINE = os.path.join(USER_HOME, 'Dropbox', 'Work', 'VBtech', 'DenisVEP', 'Results')
    FOLDER_VEP = FOLDER_VEP_ONLINE

    if platform.node() == 'dionperdMBP':
        SOFTWARE_PATH = os.path.join(USER_HOME, 'CBR', 'software', 'git')
        FOLDER_VEP_HOME = os.path.join(USER_HOME, 'CBR', 'VEP', 'tests')
        # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR', 'svn', 'episense', 'demo-data')
        DATA_TVB = os.path.join(USER_HOME, 'CBR', 'svn', 'tvb', 'tvb-data', 'tvb-data')
        # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/Results/PATI_HH')
        # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/JUNCH')
        DATA_CUSTOM = os.path.join(FOLDER_VEP, 'CC/TVB3')

    else:
        SOFTWARE_PATH = os.path.join(USER_HOME, 'VirtualVEP', 'software')
        FOLDER_VEP_HOME = os.path.join(USER_HOME, 'VEP', 'tests')
        # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR_software', 'svn-episense', 'demo-data')
        DATA_TVB = os.path.join(USER_HOME, 'CBR_software', 'svn-tvb', 'tvb-data', 'tvb-data')
        # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/Results/PATI_HH')
        # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/JUNCH')
        DATA_CUSTOM = os.path.join(FOLDER_VEP, 'CC/TVB3')

    if not (os.path.isdir(FOLDER_VEP_HOME)):
        os.mkdir(FOLDER_VEP_HOME)

    # Folder where input data will be
    # FOLDER_DATA = os. path.join(FOLDER_VEP, 'data')

    # Folder where logs will be written
    FOLDER_LOGS = os.path.join(FOLDER_VEP_HOME, 'logs' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))

    # Folder where results will be saved
    FOLDER_RES = os.path.join(FOLDER_VEP_HOME, 'results' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
    # Figures related settings:
    FOLDER_FIGURES = os.path.join(FOLDER_VEP_HOME, 'figures' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    if not (os.path.isdir(FOLDER_FIGURES)):
        os.mkdir(FOLDER_FIGURES)

    STATISTICAL_MODELS_PATH = os.path.join(SOFTWARE_PATH, "tvb-infer", "tvb_infer", "stan_epilepsy_models")

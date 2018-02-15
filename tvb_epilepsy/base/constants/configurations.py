import os
import platform
from datetime import datetime
import tvb_epilepsy


USER_HOME = os.path.expanduser("~")
TVB_EPILEPSY_PATH = os.path.dirname(tvb_epilepsy.__file__)
STATS_MODELS_PATH = os.path.join(TVB_EPILEPSY_PATH, "service", "model_inversion", "stan", "models")
CMDSTAN_PATH = os.path.join(USER_HOME, "ScientificSoftware/git/cmdstan")

RUN_ENV = "local"
if "RUN_ENV" in os.environ:
    RUN_ENV = os.environ["RUN_ENV"]
if RUN_ENV == "test":
    DATA_TEST = "data"
    FOLDER_LOGS = os.path.join(os.getcwd(), "logs")
    FOLDER_RES = os.path.join(os.getcwd(), "res")
    FOLDER_FIGURES = os.path.join(os.getcwd(), "figs")
    HEAD_FOLDER = os.path.join(DATA_TEST, "head2")

else:
    FOLDER_VEP_ONLINE = os.path.join(USER_HOME, 'Dropbox', 'Work', 'VBtech', 'VEP', 'results')
    VEP_SOFTWARE_PATH = os.path.join(USER_HOME, 'VEPtools', 'git')
    FOLDER_VEP = os.path.join(FOLDER_VEP_ONLINE, "CC")
    if platform.node() == 'dionperdMBP':
        # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR', 'svn', 'episense', 'demo-data')
        DATA_TVB = os.path.join(USER_HOME, 'CBR', 'svn', 'tvb', 'tvb-data', 'tvb-data')
    elif platform.node() == 'jupyter.cluster.thevirtualbrain.org':
        VEP_SOFTWARE_PATH = os.path.join(USER_HOME, 'VBtech/software/git')
        FOLDER_VEP_ONLINE = os.path.join(USER_HOME, 'VBtech/', 'results')
        FOLDER_VEP = os.path.join(FOLDER_VEP_ONLINE, 'CC')
        CMDSTAN_PATH = "/soft/stan/cmdstan-2.17.0"
    else:
        # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR_software', 'svn-episense', 'demo-data')
        DATA_TVB = os.path.join(USER_HOME, 'CBR_software', 'svn-tvb', 'tvb-data', 'tvb-data')
    DATA_CUSTOM = os.path.join(FOLDER_VEP, 'TVB3')
    FOLDER_VEP_TESTS = os.path.join(FOLDER_VEP_ONLINE, 'tests')
    if not (os.path.isdir(FOLDER_VEP_TESTS)):
        os.makedirs(FOLDER_VEP_TESTS)


    HEAD_FOLDER = os.path.join(DATA_CUSTOM, 'Head')
    # Folder where input data will be
    # FOLDER_DATA = os. path.join(FOLDER_VEP, 'data')
    # Folder where logs will be written
    FOLDER_LOGS = os.path.join(FOLDER_VEP_TESTS, 'logs' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    # Folder where results will be saved
    FOLDER_RES = os.path.join(FOLDER_VEP_TESTS, 'results' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
    # Figures related settings:
    FOLDER_FIGURES = os.path.join(FOLDER_VEP_TESTS, 'figures' + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
    if not (os.path.isdir(FOLDER_FIGURES)):
        os.mkdir(FOLDER_FIGURES)

VERY_LARGE_SIZE = (40, 20)
VERY_LARGE_PROTRAIT = (30, 50)
SUPER_LARGE_SIZE = (80, 40)
LARGE_SIZE = (20, 15)
SMALL_SIZE = (15, 10)
FIG_SIZE = SMALL_SIZE
FIG_FORMAT = 'png'
SAVE_FLAG = True
SHOW_FLAG = False
MOUSEHOOVER = False

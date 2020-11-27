__version__ = "0.0.1"

# Maintainer info
__author__ = "The FLIRT development team"

# Disable some annoying warnings
try:
    import warnings
    from tqdm.std import TqdmExperimentalWarning

    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)

    from astropy.utils.exceptions import AstropyWarning

    warnings.simplefilter('ignore', category=AstropyWarning)
except:
    pass

# Export main functions
from .acc import get_acc_features
from .eda import get_eda_features
from .hrv import get_hrv_features
from .bvp import get_bvp_features
from .temp import get_temp_features
from .stats import get_stat_features

# extra sub-packages
#from .simple import get_features_for_empatica_archive
from flirt.reader import empatica
#import flirt.with_

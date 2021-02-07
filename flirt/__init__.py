__version__ = "0.0.2"

# Maintainer info
__author__ = "The FLIRT development team"

# Dependencies


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
from .stats import get_stat_features
from .with_ import *

# extra sub-packages
# flirt.simple
# flirt.reader
# flirt.with_

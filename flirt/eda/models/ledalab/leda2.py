"""
Settings and current variables for current data structure. Emulates ledalab's "leda2" data structure and ledapreset.m
usage for proper emulation:
import leda2
(leda2.target.phasicData = xxx)  # the statement as would be in matlab
do this every time we need to access the objects
use the reset() method to reset the objects
"""


class Current:
    """
    Stores current values (old Ledalab's current.batchmode parameters)
    """
    def __init__(self):
        """
        Sets default values. Values such as filter_settings, smooth, downsample removed from this version
        """
        self.do_optimize = 2


class Analysis:
    """
    Stores all variables related to current analysis.
    """
    class Target:
        def __init__(self):
            self.tonicDriver = None
            self.t = None
            self.d = None
            self.sr = None
            self.poly = None  # interpolate.PchipInterpolator instance
            self.method = 'sdeco'
            self.groundtime = None
            self.groundlevel = None
            self.groundlevel_pre = None
            self.iif_t = None
            self.iif_data = None

    class Error:
        def __init__(self):
            self.MSE = None
            self.RMSE = None
            self.chi2 = None
            self.deviation = None
            self.discreteness = None
            self.negativity = None
            self.compound = None

    def __init__(self):
        self.target = Analysis.Target()
        self.error = Analysis.Error()
        self.tau = None
        self.driver = None
        self.tonicDriver = None
        self.driverSC = None
        self.remainder = None
        self.phasicData = None
        self.tonicData = None
        self.phasicDriverRaw = None
        self.tauMin = .001
        self.tauMinDiff = .01
        self.smoothwin = .2
        self.smoothwin_sdeco = .2
        self.segmWidth = 12
        self.amp = None
        self.method = 'sdeco'
        self.impulseOnset = None
        self.impulsePeakTime = None
        self.impulseAmp = None
        self.onset = None
        self.peakTime = None
        self.kernel = None


class Settings:
    """
    Stores default settings.
    """
    class InitVal:
        def __init__(self):
            self.hannWinWidth = .5
            self.signHeight = .01
            self.groundInterp = 'spline'  # 'pchip' keeps only S(x)' continuous

    def __init__(self):
        self.tauMin = .001  # .1
        self.tauMax = 100.0
        self.tauMinDiff = .01
        self.dist0_min = .001
        self.initVal = Settings.InitVal()
        self.tonicGridSize_sdeco = 10.0
        self.tau0_sdeco = [1.0, 3.75]   # see Benedek & Kaernbach, 2010, J Neurosc Meth
        self.d0Autoupdate_sdeco = 0.0
        self.smoothwin_sdeco = .2
        self.sigPeak = .001
        self.segmWidth = None


class Data:
    """
    data.conductance.data = data.conductance_data in this version
    data.time.data = data.time_data in this version
    """
    def __init__(self):
        self.time_data = None
        self.conductance_data = None
        self.conductance_smoothData = None
        self.conductance_smoothData_win = None
        self.conductance_error = None
        self.conductance_min = None
        self.conductance_max = None
        self.samplingrate = None
        self.timeoff = 0  # was leda2.data.time.timeoff
        self.N = None


class Trough2peakAnalysis:
    def __init__(self):
        self.onset = None
        self.peaktime = None
        self.onset_idx = None
        self.peaktime_idx = None
        self.amp = None

# globals used to pass settings across modules
settings = Settings()
analysis = Analysis()
analysis0 = Analysis()
current = Current()
data = Data()
trough2peakAnalysis = Trough2peakAnalysis()


def reset():
    """
    Call this to re-initialise the leda2 object
    """
    global settings, analysis, analysis0, current, data, trough2peakAnalysis
    settings = Settings()
    analysis = Analysis()
    analysis0 = Analysis()
    current = Current()
    data = Data()
    trough2peakAnalysis = Trough2peakAnalysis()
    

def delete_fit(self):
    global analysis0, analysis
    analysis0 = Analysis()
    analysis = Analysis()
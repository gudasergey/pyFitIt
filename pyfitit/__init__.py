import warnings, os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from . import utils
utils.fixDisplayError()
from . import adf, directMethod, exafs, fdmnes, feff, fileBrowser, geometry, curveFitting, ihs
from . import inverseMethod, ML, molecule, mpi, optimize,  plotting, sampling, smoothLib, descriptor, mixture, adaptiveSampling
from . import uiControls, msPathGroupManager, funcModel

from .adaptiveSampling import minlpe
from .curveFitting import autobk, subtractBaseAuto, subtractBase, mback
from .descriptor import addDescriptors, calcPreedge, descriptorQuality, efermiDescriptor, pcaDescriptor, plotDescriptors1d, plotDescriptors2d, relPcaDescriptor, stableExtrema
from .directMethod import Estimator as constructDirectEstimator
from .directMethod import compareDifferentMethods as compareDifferentDirectMethods
from .exafs import Exafs, convert2RSpace, waveletTransform, xanes2exafs, FT_Transform
from .fdmnes import parseOneFolder as parseFdmnesFolder
from .feff import parseOneFolder as parseFeffFolder
from .funcModel import FuncModel, ParamProperties
from .inverseMethod import Estimator as constructInverseEstimator
from .inverseMethod import compareDifferentMethods as compareDifferentInverseMethods
from .mixture import generateMixtureOfSample, tryAllMixtures
from .ML import Sample, readSample
from .molecule import Molecule, pi, norm, cross, dot, normalize
from .msPathGroupManager import MSPathGroupManager, ExafsPredictor, sampleExafs, fitExafsParams, fitExafsByStructureSliders, fitXanesAndExafsSimultaneously
from .plotting import plotSample, plotToFile, readPlottingFile
from .project import Project, loadProject, checkProject, saveAsProject, createPartialProject
from .sampling import generateInputFiles, calcSpectra, calcForAllxyzInFolder, collectResults, sampleAdaptively, checkSampleIsGoodByCount, checkSampleIsGoodByCVError
from .smoothLib import smoothInterpNorm, smoothDataFrame, smoothExp
from .uiControls import fitSmooth, smoothSliders, FuncModelSliders, SpectrumSliders, xanesSliders, SampleInspector
from .utils import Spectrum, readSpectrum, readExafs, readSpectra, SpectrumCollection, reloadPyfitit, loadData, initPyfitit, saveNotebook, saveAsScript, Cache

# used by user in project files
from types import MethodType

def openFile(*p): return fileBrowser.openFile('openFile',*p)


join = os.path.join


def getProjectFolder(): return os.getcwd()


def loadProject(*p, **q):
    initPyfitit()
    return project.loadProject(*p,**q)


def parseADFFolder(folder, makePiramids=False):
    spectrum, _ = adf.parse_one_folder(folder, makePiramids)
    return spectrum


utils.fixFlaskChmod()

# uncomment to see warnings source line
# warnings.filterwarnings('error')

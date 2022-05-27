import warnings, os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# uncomment to see warnings source line
# warnings.filterwarnings('error')
import numpy as np
from . import utils
utils.fixDisplayError()
from . import adf, directMethod, fdmnes, feff, fileBrowser, geometry, curveFitting, ihs
from . import inverseMethod, ML, molecule, optimize,  plotting, sampling, smoothLib, descriptor, mixture, adaptiveSampling
from . import uiControls, msPathGroupManager, funcModel

from .adaptiveSampling import minlpe
from .descriptor import stableExtrema, efermiDescriptor, pcaDescriptor, relPcaDescriptor, plotDescriptors1d, plotDescriptors2d, descriptorQuality, addDescriptors
from .directMethod import Estimator as constructDirectEstimator
from .directMethod import compareDifferentMethods as compareDifferentDirectMethods
from .fdmnes import parseOneFolder as parseFdmnesFolder
from .feff import parseOneFolder as parseFeffFolder
from .funcModel import FuncModel, ParamProperties
from .inverseMethod import Estimator as constructInverseEstimator
from .inverseMethod import compareDifferentMethods as compareDifferentInverseMethods
from .mixture import generateMixtureOfSample
from .ML import Sample
readSample = ML.Sample.readFolder
from .molecule import Molecule, pi, norm, cross, dot, normalize
from .msPathGroupManager import MSPathGroupManager, ExafsPredictor, convert2RSpace, sampleExafs, fitExafsParams, fitExafsByStructureSliders, fitXanesAndExafsSimultaneously
from .plotting import plotSample, plotToFile, readPlottingFile
from .project import Project, loadProject, checkProject, saveAsProject, createPartialProject
from .sampling import generateInputFiles, calcSpectra, collectResults, sampleAdaptively, checkSampleIsGoodByCount, checkSampleIsGoodByCVError
from .smoothLib import smoothInterpNorm
from .uiControls import fitSmooth, smoothSliders, FuncModelSliders, SpectrumSliders, xanesSliders, SampleInspector
from .utils import Spectrum, readSpectrum, readExafs, readSpectra, SpectrumCollection, reloadPyfitit, cacheInFile, loadData, initPyfitit, saveNotebook, saveAsScript

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




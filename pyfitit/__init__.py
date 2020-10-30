import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from . import utils
utils.fixDisplayError()
import numpy as np
import copy, json, warnings, traceback, gc, jupytext, nbformat, numbers
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Javascript, HTML
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from matplotlib.font_manager import FontProperties
from types import MethodType
from sklearn.ensemble import ExtraTreesRegressor

# we need to list all libraries, otherwise python can't find them in other imports
from . import adf, directMethod, factorAnalysis, fdmnes, feff, fileBrowser, geometry, curveFitting
from . import inverseMethod, ML, molecule, optimize,  plotting, sampling, smoothLib, descriptor, mixture, adaptiveSampling
from . import uiControls

from .project import Project, loadProject, checkProject, saveAsProject, createPartialProject
from .molecule import Molecule, pi, norm, cross, dot, normalize
from .utils import Spectrum, readSpectrum, readExafs, readSpectra, SpectrumCollection


join = os.path.join

debug = False


def getProjectFolder(): return os.getcwd()


# uncomment to see warnings source line
if debug:
    warnings.filterwarnings('error')

styles = '''<style>
    .container { width:100% !important; }
    .output_area {display:inline-block !important; }
    .cell > .output_wrapper > .output {margin-left: 14ex;}
    .out_prompt_overlay.prompt {min-width: 14ex;}
    .fitBySlidersOutput {flex-direction:row !important; }
    /*.fitBySlidersOutput .p-Widget.jupyter-widgets-output-area.output_wrapper {display:none} - прячет ошибки*/
    .fitBySlidersOutput div.output_subarea {max-width: inherit}
    .fitBySlidersExafsOutput {flex-direction:row !important; }
    /*.fitBySlidersExafsOutput .p-Widget.jupyter-widgets-output-area.output_wrapper {display:none} - прячет ошибки*/
    .fitBySlidersExafsOutput div.output_subarea {max-width: inherit}
    .pcaOutput { display:block; }
    .pcaOutput div.output_subarea {max-width: inherit}
    .pcaOutput .output_subarea .widget-hbox { place-content: initial!important; }
    .pcaOutput > .output_area:nth-child(2) {float:left !important; }
    .pcaOutput > .output_area:nth-child(3) {display:block !important; }
    .pcaOutput > .output_area:nth-child(4) { /* width:100%; */ }
    .status { white-space: pre; }

    .fileBrowser {margin-left:14ex; display:block; }
    .fileBrowser div.p-Panel {display:block; }
    .fileBrowser button.folder { font-weight:bold; }
    .fileBrowser button.parentFolder { font-size:200%; }

    .widget-inline-hbox .widget-label {width:auto}
    </style>'''


def initPyfitit():
    if utils.isJupyterNotebook():
        display(HTML(styles))


initPyfitit()


def saveNotebook():
    if not utils.isJupyterNotebook(): return
    display(Javascript('IPython.notebook.save_notebook();'))


def loadProject(*p, **q):
    initPyfitit()
    return project.loadProject(*p,**q)


def parseFdmnesFolder(folder):
    spectrum = fdmnes.parse_one_folder(folder)
    return spectrum


def parseADFFolder(folder, makePiramids = False):
    spectrum, _ = adf.parse_one_folder(folder, makePiramids)
    return spectrum


parseFeffFolder = feff.parse_one_folder

# sampling
readSample = ML.Sample.readFolder
generateInputFiles = sampling.generateInputFiles
calcSpectra = sampling.calcSpectra
collectResults = sampling.collectResults
constructDirectEstimator = directMethod.Estimator
constructInverseEstimator = inverseMethod.Estimator
krigingSampling = sampling.krigingSampling
sampleAdaptively = adaptiveSampling.sampleAdaptively

smoothInterpNorm = smoothLib.smoothInterpNorm

plotSample = plotting.plotSample

def saveAsScript(fileName):
    if not utils.isJupyterNotebook(): return
    fileName = utils.fixPath(fileName)
    notebook_path = utils.this_notebook()
    if notebook_path is None:
        print('Can\'t find notebook file. Do you use non standard connection to jupyter server? Save this file as script in main menu')
        return
    with open(notebook_path, 'r', encoding='utf-8') as fp:
        notebook = nbformat.read(fp, as_version=4)
    script = jupytext.writes(notebook, ext='.py', fmt='py')
    script_path = fileName
    if script_path[-3:] != '.py': script_path += '.py'
    with open(script_path, 'w', encoding='utf-8') as fp: fp.write(script)


compareDifferentInverseMethods = inverseMethod.compareDifferentMethods
compareDifferentDirectMethods = directMethod.compareDifferentMethods


# for PCA
def openFile(*p): return fileBrowser.openFile('openFile',*p)
calcSVD = factorAnalysis.calcSVD
MalinowskyParameters = factorAnalysis.MalinowskyParameters
plotTestStatistic = factorAnalysis.plotTestStatistic
recommendPCnumber = factorAnalysis.recommendPCnumber


def saveToFile(fileName, obj):
    fileName = utils.fixPath(fileName)
    folder = os.path.dirname(fileName)
    os.makedirs(folder, exist_ok=True)
    if isinstance(obj, np.ndarray):
        np.savetxt(fileName, obj, delimiter = ' ')
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv(fileName, sep=' ', index=False)
    else:
        with open(fileName, 'w') as f: json.dump(obj, f)

# fitting
fitSmooth = uiControls.FitSmooth

# params: sample, project, debug=False, defaultParams=None, theoryProcessingPipeline=None
fitBySliders = uiControls.SpectrumSliders

# params: sample, project, debug=True, defaultParams=None
fitBySlidersExafs = uiControls.ExafsSliders

# params: sampleList, projectList, defaultParams=None, debug=False
fitBySlidersMixture = uiControls.XanesMixtureFittingBackend

# params: sample, debug=True, customProcessor=None, customPlotter=None
SampleInspector = uiControls.SampleInspector

stableExtrema = descriptor.stableExtrema
efermiDescriptor = descriptor.efermiDescriptor
pcaDescriptor = descriptor.pcaDescriptor
relPcaDescriptor = descriptor.relPcaDescriptor
plot_descriptors_1d = descriptor.plot_descriptors_1d
plot_descriptors_2d = descriptor.plot_descriptors_2d
descriptor_quality = descriptor.descriptor_quality
generateMixtureOfSample = mixture.generateMixtureOfSample

def plot_data(energy,data):
    _=plt.plot(energy,data)
    axes = plt.gca()
    axes.set_xlim([min(energy),max(energy)])
    plt.xlabel('Energy',fontweight='bold')
    plt.ylabel('Absorption',fontweight='bold')
    plt.title('Experimental Data')

def interpolation(energy,data,step):
    e_valor=np.arange(min(energy),max(energy),step)
    dat_valor=[]
    col=np.shape(data)[1]
    for i in range(col):
        d=np.interp(e_valor,energy,data[:,i])
        dat_valor.append(d)
    data=np.transpose(dat_valor)
    energy=e_valor
    return energy, data

def normalization(energy,data):
    scaled=np.zeros(np.shape(data)[1])
    for i in range(np.shape(data)[1]):
        scaled[i]=np.sqrt((1./((1./(np.max(energy)-np.min(energy)))*(np.trapz((data[:,i])**2,energy)))))
    for i in range(np.shape(data)[1]):
        data[:,i]=data[:,i]*scaled[i]
    return data

plotPCAcomponentsParams = {}
def plotPCAcomponents(energy, principal_components):
    global plotPCAcomponentsParams
    # Not Normalized Components
    n_col=np.shape(principal_components)[1]
    principal_components_v=copy.copy(principal_components)
    components=pd.DataFrame(principal_components)
    components.index=energy
    # Normalized Components
    val_N=pd.DataFrame(principal_components_v)
    val=np.transpose(val_N.values)
    for i in range(np.shape(val)[0]):
        val[i]=val[i]/(np.trapz(val[i],energy))
    dv=pd.DataFrame(np.transpose(val))
    dv.index=energy
    fig, ax = plt.subplots(figsize=plotting.figsize)
    plotPCAcomponentsParams["Switching"] = 'Not Normalized'
    plotPCAcomponentsParams["Component"] = 1

    def redraw():
        ax.clear()
        switching = plotPCAcomponentsParams["Switching"]
        valor=components if switching=='Not Normalized' else dv
        control = plotPCAcomponentsParams["Component"]
        for j in range(control):
            ax.plot(energy, valor.iloc[:,j],linewidth=2) ####
        plt.xlabel('Energy',fontweight='bold')
        plt.ylabel('Absorption',fontweight='bold')
        plt.title('Abstract Components')
        plt.xlim([min(energy),max(energy)])
        #plt.grid()
        filename = "N_abstract_components.dat"
        if switching=='Not Normalized': filename = 'Not_'+filename
        saveToFile('results'+os.sep+filename, valor.iloc[:,0:control])

    radio = widgets.RadioButtons( options=['Not Normalized', 'Normalized'], description='Switching:', disabled=False)
    def switch(change):
        global plotPCAcomponentsParams
        if change['type'] != 'change' or change['name'] != 'value': return
        plotPCAcomponentsParams["Switching"] = change['new']
        redraw()
    radio.observe(switch)

    componentWidget = widgets.BoundedIntText(value=1, min=1, max=n_col, step=1, description='Component:', disabled=False)
    def changeComp(change):
        global plotPCAcomponentsParams
        if change['type'] != 'change' or change['name'] != 'value': return
        plotPCAcomponentsParams["Component"] = change['new']
        redraw()
    componentWidget.observe(changeComp)

    uif = widgets.HBox((radio, componentWidget))
    uif.layout.flex_flow = 'row wrap'
    uif.layout.justify_content = 'space-between'
    uif.layout.align_items = 'flex-start'
    uif.layout.align_content = 'flex-start'
    display(uif)
    display(Javascript('$(this.element).addClass("pcaOutput");'))
    redraw()

PCAplot={}
def PCAcomparison(energy,data):
    global PCAplot
    n_col=np.shape(data)[1]
    u,s,vT=np.linalg.svd(data, full_matrices=False)
    PCAplot['column']=0
    PCAplot['number']=1
    fig, ax = plt.subplots(figsize=plotting.figsize)
    inset_ax = fig.add_axes([0.53, 0.25, 0.35, 0.2])
    def redrawf():
        ax.clear()
        inset_ax.clear()
        control1=PCAplot['column']
        control2=PCAplot['number']
        u_red = u[:,:control2]
        s_red = np.diag(s[:control2])
        v_red = vT[:control2,:]
        dataR=np.dot(np.dot(u_red,s_red),v_red)
        residuals=data-dataR
        ax.set_title('Experimental vs Reconstructed spectra')
        ax.set_xlabel('Energy',fontweight='bold')
        ax.set_ylabel('Absorption',fontweight='bold')
        ax.plot(energy,data[:,control1],linewidth=3,color='red',label='Exp.')
        ax.set_xlim([min(energy),max(energy)])
        ax.plot(energy,dataR[:,control1],linewidth=2,ls='--',color='black',label='Reconstructed')
        ax.legend()
        ax.legend()
        inset_ax.set_xlim([min(energy),max(energy)])
        inset_ax.axhline(0)
        inset_ax.plot(energy,residuals[:,control1])
        inset_ax.set_title('Residuals Plot')
    componentWidget1 = widgets.BoundedIntText(value=0, min=0, max=n_col-1, step=1, description='Spectrum:', disabled=False, continuous_update=True)
    def changeComp1(change1):
        global PCAplot
        if change1['type'] != 'change' or change1['name'] != 'value': return
        PCAplot["column"] = change1['new']
        redrawf()
    componentWidget1.observe(changeComp1)

    componentWidget2 = widgets.BoundedIntText(value=1, min=1, max=n_col, step=1, description='Component:', disabled=False, continuous_update=True)
    def changeComp2(change2):
        global PCAplot
        if change2['type'] != 'change' or change2['name'] != 'value': return
        PCAplot["number"] = change2['new']
        redrawf()
    componentWidget2.observe(changeComp2)
    uif = widgets.HBox((componentWidget1,componentWidget2))

    uif.layout.flex_flow = 'row wrap'
    uif.layout.justify_content = 'space-between'
    uif.layout.align_items = 'flex-start'
    uif.layout.align_content = 'flex-start'
    display(uif)
    display(Javascript('$(this.element).addClass("pcaOutput");'))
    redrawf()










def targetTransformationPCA_old(energy, data):
    n_row=np.shape(data)[0]
    n_col=np.shape(data)[1]
    def q(NumFactors):
        lisT=np.zeros(NumFactors)
        for i in range(NumFactors-1):
            lisT[i]=(0.5*2.**(1.-i))
        lisT=np.sort(lisT)
        for i in range(len(lisT)):
            lisT[i]=int(np.round(n_col*lisT[i]))
        lisT[len(lisT)-1]=lisT[len(lisT)-1]-1
        # Reference Matrix
        references= np.zeros((n_row,NumFactors))
        for i in range(len(lisT)):
            references[:,i]=data[:,np.int(lisT[i])]
        # Initialization
        mat_X = np.zeros((NumFactors, NumFactors))
        spectra = factorAnalysis.Spectra(data, references, NumFactors)
        n_sliders = NumFactors**2

        fig = plt.figure(figsize=(10, 7))
        axs = fig.add_subplot(121)
        axc = fig.add_subplot(122)

        def update_N(**xvalor):
            xvalor=[]
            for i in range(n_sliders):
                xvalor.append(controls[i].value)
            values=np.reshape(xvalor,(NumFactors,NumFactors))
            mat_X=np.transpose(values)
            sp,conc=spectra.get_spectrum(mat_X)
            new_xanes=pd.DataFrame(sp)
            new_xanes.index=energy
            new_concentrations=pd.DataFrame(conc)
            saveToFile("results"+os.sep+"Pure_XANES_Spectra.dat",new_xanes.values)
            saveToFile("results"+os.sep+"Pure_XANES_Concentrations.dat",new_concentrations.values)

            #plot figures
            axs.clear()
            axs.set_xlim([min(energy),max(energy)])
            axs.set_xlabel("Energy (eV)",fontweight='bold')
            axs.set_ylabel("Absorbance",fontweight='bold')
            axs.set(title="Pure XANES")
            axc.clear()
            axc.set_xlim([0,n_col-1])
            axc.set_ylim([-0.05,1.2])
            axc.set_xlabel("Scan Index",fontweight='bold')
            axc.set_ylabel("Fraction of Pure Components",fontweight='bold')
            axc.set(title="Pure Concentrations")
            new_xanes.plot(ax=axs, linewidth=3)
            new_concentrations.plot(ax=axc,linewidth=3)

        # Setup Widgets
        controls=[]
        o='vertical'
        for i in range(n_sliders):
            title="x%i%i" % (i%NumFactors+1, i//NumFactors+1)
            sl=widgets.FloatSlider(description=title,min=-5.0, max=5.0, step=0.1, orientation=o, continuous_update=False)
            controls.append(sl)
        controlsDict = {}
        for c in controls:
            controlsDict[c.description] = c
        uif = widgets.HBox(tuple(controls))

        outf = widgets.interactive_output(update_N,controlsDict)
        display(uif, outf)
        return NumFactors

    widgets.interact(q,NumFactors=widgets.BoundedIntText(
            value=2,
            min=2,
            max=n_col-1,
            step=1,
            description='PCs:',
            disabled=False))

class targetTransformationPCA:
    def __init__(self, energy, data, sign, min_val, max_val, step_val):
        self.params = {}
        self.fig = plt.figure()
        self.pureSpectra = None
        self.pureConcentrations = None

        def n_pc(components):
            NumFactors=components
            self.params['PCs'] = components
            u,s,vT=np.linalg.svd(data, full_matrices=False)
            u_red = u[:,:NumFactors]
            s_red = np.diag(s[:NumFactors])
            v_red = vT[:NumFactors,:]
            us=np.dot(u_red,s_red)
            n_col=np.shape(data)[1]
            # scaling
            scale=sign*(np.sqrt((1./((1./(np.max(energy)-np.min(energy)))*(np.trapz((us[:,0])**2,energy)))))) # Ho tolto -1 davanti a scale
            scale_M=[]
            for i in range(NumFactors):
                scale_M.append(scale)
        #mat_X_0 Not Normalized
            mat_X = np.zeros((NumFactors, NumFactors))
        ##############################################
        #mat_X_N Normalized
            mat_X_N = np.zeros((NumFactors, NumFactors))
            mat_X_N[0,:]=scale_M
            n_sliders = NumFactors**2
            n_sliders_N=(NumFactors**2)-NumFactors
        ##############################################
        #mat_X_1s first spectrum fixed
            col_1s = np.linalg.lstsq(us,data[:,0],rcond=None)[0]
            mat_X_1s=np.zeros((NumFactors, NumFactors))
            mat_X_1s[:,0]=col_1s
        #mat_X_L last spectrum fixed
            col_L=np.linalg.lstsq(us,data[:,np.shape(data)[1]-1],rcond=None)[0]
            mat_X_L=np.zeros((NumFactors, NumFactors))
            mat_X_L[:,NumFactors-1]=col_L
        #mat_X_1s_L first and last spectrum fixed
            mat_X_1s_L=np.zeros((NumFactors, NumFactors))
            mat_X_1s_L[:,0]=col_1s
            mat_X_1s_L[:,NumFactors-1]=col_L
            n_sliders_1s_L=(NumFactors**2)-2*NumFactors
        ###############################################
        #mat_X_N_1s Normalized and first spectrum fixed
            mat_X_N_1s = np.zeros((NumFactors, NumFactors))
            mat_X_N_1s[0,:]=scale_M
            mat_X_N_1s[:,0]=col_1s
            n_sliders_N_1s=(NumFactors-1)**2
        ###############################################
        #mat_X_N_L Normalized and Last spectrum fixed
            mat_X_N_L = np.zeros((NumFactors, NumFactors))
            mat_X_N_L[0,:]=scale_M
            mat_X_N_L[:,NumFactors-1]=col_L
            n_sliders_N_L=(NumFactors-1)**2
        #################################################
        #mat_X_N_L Normalized  1st and Last spectrum fixed
            mat_X_N_1st_L = np.zeros((NumFactors, NumFactors))
            mat_X_N_1st_L[0,:]=scale_M
            mat_X_N_1st_L[:,0]=col_1s
            mat_X_N_1st_L[:,NumFactors-1]=col_L
            n_sliders_N_1s_L=(NumFactors-1)*(NumFactors-2)

            def nature(control):
                ntype = control
                self.params['Switching'] = control
                if ntype=="Case: 1":
                    def assign(control_1):
                        s_spectrum=control_1
                        self.params['Constraints'] = control_1
                        if s_spectrum=='No Constraints':
                            matrix=mat_X
                            sliders=n_sliders
                        elif s_spectrum=="1st spectrum fixed":
                            matrix=mat_X_1s
                            sliders=n_sliders_N
                        elif s_spectrum=="Last spectrum fixed":
                            matrix=mat_X_L
                            sliders=n_sliders_N
                        elif s_spectrum=='1st and Last spectrum fixed':
                            matrix=mat_X_1s_L
                            sliders=n_sliders_1s_L
                        factorAnalysis.unNorm(s_spectrum,us,v_red,matrix,NumFactors,sliders,data,energy,min_val,max_val,step_val, self)

                    widgets.interact(assign, control_1 = widgets.RadioButtons(
                    options=['No Constraints','1st spectrum fixed', 'Last spectrum fixed','1st and Last spectrum fixed'],
                    description='Constraints:',
                    disabled=False))

                elif ntype=="Case: 2":
                    def assign2(control_2):
                        n_spectrum=control_2
                        self.params['Constraints'] = control_2
                        if n_spectrum=="Normalization":
                            matrix=mat_X_N
                            sliders=n_sliders_N
                        elif n_spectrum=="Norm. and 1st spectrum":
                            matrix=mat_X_N_1s
                            sliders=n_sliders_N_1s
                        elif n_spectrum=="Norm. and Last spectrum":
                            matrix=mat_X_N_L
                            sliders=n_sliders_N_L
                        elif n_spectrum=="Norm., 1s and last spectrum":
                            matrix=mat_X_N_1st_L
                            sliders=n_sliders_N_1s_L
                        factorAnalysis.Norm(n_spectrum,us,v_red,matrix,NumFactors,sliders,data,energy,min_val,max_val,step_val, self)

                    widgets.interact(assign2, control_2 = widgets.RadioButtons(
                    options=['Normalization','Norm. and 1st spectrum','Norm. and Last spectrum','Norm., 1s and last spectrum'],
                    description='Constraints:',
                    disabled=False))

            widgets.interact(nature, control = widgets.RadioButtons(
            options=['Case: 1', 'Case: 2'],
            description='Switching:',
            disabled=False))

        n_col=np.shape(data)[1]
        widgets.interact(n_pc,components=widgets.BoundedIntText(
            value=2,
            min=2,
            max=n_col-1,
            step=1,
            description='PCs:',
            disabled=False))

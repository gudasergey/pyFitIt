import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display
import pandas as pd
from scipy.stats import f
import os
import scipy.integrate as integrate

refSelection={}
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def interpolation(energy, intensity, stepVal):
    energy_interpolation=np.arange(np.min(energy),np.max(energy),stepVal)
    columns=np.shape(intensity)[1] ;rows=len(energy_interpolation)
    intensity_interpolation=np.zeros((rows,columns))
    for i in range(columns):intensity_interpolation[:,i]=np.interp(energy_interpolation,energy,intensity[:,i])
    return energy_interpolation,intensity_interpolation 


def normalization(energy, intensity):
    scaled=np.zeros(np.shape(intensity)[1])
    for i in range(np.shape(intensity)[1]):
        scaled[i]=np.sqrt(1./((1./(np.max(energy)-np.min(energy)))*(np.trapz((intensity[:,i])**2,energy))))
    for i in range(np.shape(intensity)[1]):
        intensity[:,i]=intensity[:,i]*scaled[i]
    return intensity


def sliderDescription(npc, minVal=-5, maxVal=5, stepVal=0.1):
    controls=[]
    for i in range(npc**2):
        rg=widgets.FloatRangeSlider(value=[minVal, maxVal], min=minVal, max=maxVal, step=stepVal, description='Range t'+str(i), disabled=False, continuous_update=False,orientation='horizontal',readout=True,readout_format='.1f')
        sl=widgets.FloatSlider(value=0.5*( minVal+maxVal),min= minVal, max= maxVal,step= 0.05, description='t '+str(i), disabled=False, continuous_update=False)
        stp=widgets.BoundedFloatText(value=stepVal, min=0, max=maxVal, description='Step t'+str(i), disabled=False)
        controls.append([rg,stp,sl])
    return controls


def malinowsky(l, nrow, ncol):
    ind=np.zeros(ncol-1)
    ie=np.zeros(ncol-1)
    index=range(1,ncol)
    for i in range(0,ncol-1):
        ind[i]=(np.sqrt((np.sum(l[i+1:ncol]))/(nrow*(ncol-index[i]))))/(ncol-index[i])**2
        ie[i]=np.sqrt(index[i]*(np.sum(l[i+1:ncol]))/(nrow*ncol*(ncol-index[i])))
    return ind, ie

def fisherFunction(l, nrow, ncol):
    pc=np.arange(1, ncol+1, 1)
    p=np.zeros(len(pc))
    for i in range(0,len(pc)):
        p[i]=(nrow-pc[i]+1)*(ncol-pc[i]+1)
    s1=np.zeros(len(pc));s2=np.zeros(len(pc))
    fi=np.zeros(len(pc)-1); fisher=np.zeros(len(pc)-1)
    a=pc+1
    for i1 in range(0,ncol-1):s1[i1]=np.sum((nrow-a[i1:np.size(pc)]+1)*(ncol-a[i1:np.size(pc)]+1))
    for i2 in range(0,ncol-1):s2[i2]=np.sum(l[i2+1:np.size(pc)+1])
    for i3 in range(0,ncol-1):fi[i3]=(s1[i3]/p[i3])*(l[i3]/s2[i3])
    for i4 in range(0,ncol-1):fisher[i4]=((integrate.quad(lambda x: f.pdf(x, 1, (ncol-1)-i4), fi[i4], np.inf))[0])*100
    return fisher

def xanesRfactor(intensity, pcfit):
    ncol=np.shape(intensity)[1]
    rlist=np.zeros(ncol)
    for i in range(ncol):
        rlist[i]=100*sum((intensity[:,i]-pcfit[:,i])**2)/sum((intensity[:,i])**2)
    return rlist


plotcontrol={}
def plotStatistic(s, ind, ie, fisher, folder):
    global plotcontrol
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(8,7))
    plotcontrol['Log10 scale']=False
    statistic=pd.DataFrame({'SV':s[:len(s)-1],'IND': ind, 'IE': ie, 'F':fisher})
    statisticLog=pd.DataFrame({'SV (log10)':np.log10(s[:len(s)-1]),'IND (log10)': np.log10(ind), 'IE (log10)': np.log10(ie), 'F':fisher})
    statistic.to_csv(os.path.join(folder,'Statistic.csv'), index = True, header=False)
    statisticLog.to_csv(os.path.join(folder,'StatisticLog.csv'), index = True, header=False)
    def redraw():
        ax1.clear();ax2.clear();ax3.clear();ax4.clear()
        switching = plotcontrol['Log10 scale']
        if switching==False:
            ax1.plot(s,'-o',color='green')
            ax1.set_ylabel('SV',size=12)
            ax1.set_title('Scree',size=15)
            ax2.plot(ind,'-o',color='blue')
            ax2.set_title('IND plot',size=15)
            ax2.set_ylabel('IND',size=12)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax3.plot(fisher,'-o',color='red')
            ax3.set_title('F-test',size=15)
            ax3.set_ylabel('F variable',size=12)
            ax3.set_xlabel('Scan number',size=12)
            ax4.plot(ie,'-o',color='tab:blue')
            ax4.set_title('IE plot',size=15)
            ax4.set_ylabel('IE',size=12)
            ax4.set_xlabel('Scan number',size=12)
            ax4.yaxis.set_label_position("right")
            ax4.yaxis.tick_right()
        elif switching==True:
            ax1.plot(np.log10(s),'-o',color='green')
            ax1.set_ylabel('Variance',size=12)
            ax1.set_title('Scree',size=15)
            ax2.plot(np.log10(ind),'-o',color='blue')
            ax2.set_title('IND plot',size=15)
            ax2.set_ylabel('IND',size=12)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax3.plot(fisher,'-o',color='red')
            ax3.set_title('F-test',size=15)
            ax3.set_ylabel('F variable',size=12)
            ax3.set_xlabel('Scan number',size=12)
            ax4.plot(np.log10(ie),'-o',color='tab:blue')
            ax4.set_title('IE plot',size=15)
            ax4.set_ylabel('IE',size=12)
            ax4.yaxis.set_label_position("right")
            ax4.yaxis.tick_right()
            ax4.set_xlabel('Scan number',size=12)
    def reprint():
        switching = plotcontrol['Log10 scale']
        if switching==False: print(statistic) 
        elif switching==False: print(statisticLog)
        
    cb1=widgets.Checkbox(value=False,description='Log10 scale',disabled=False,indent=False)
    cb2=widgets.Checkbox(value=False,description='Show data',disabled=False,indent=False)
    def outputType(**args):
        global plotcontrol
        plotcontrol['Log10 scale']=cb1.value
        redraw()
        if cb1.value==True and cb2.value==True: print(statisticLog)
        elif cb1.value==False and cb2.value==True:print(statistic)
        elif cb2.value==False: print('')
    ui=HBox([cb1,cb2])
    out=widgets.interactive_output(outputType,{cb1.description: cb1,cb2.description:cb2})
    display(ui,out)

def makePCAfit(energy,intensity, u,s,v,folder):
    ncol=np.shape(intensity)[1]
    spectrum=widgets.BoundedIntText(value=0,min=0,max=ncol-1,step=1,description='Spectrum:',disabled=False)
    pcNumber=widgets.BoundedIntText(value=1,min=1,max=ncol,step=1,description='PCs:',disabled=False)
    cbsave=widgets.Checkbox(value=False,description='Save PCA fit',disabled=False,indent=False)
    fig=plt.figure(figsize=(8,7))
    gs=GridSpec(2,1,height_ratios=[5, 1])
    ax1=fig.add_subplot(gs[0])
    ax2=fig.add_subplot(gs[1])
    ax2.set_xlabel('Energy',size=12)
    ax1.set_ylabel('Intensity',size=12)
    ax1.set_title('PCA Fit',size=15)
    pcfit_initial=np.dot(u[:,0:1],np.dot(np.diag(s[0:1]),v[0:1,:]))
    line,=ax1.plot(energy,intensity[:,0],color='black',label='Spectrum 0')
    line1,=ax1.plot(energy,pcfit_initial[:,0],color='blue',label='PCs 1')
    line2,=ax2.plot(energy,intensity[:,0]-pcfit_initial[:,0])
    ax1.legend()
    def updateCurves(**args):
        npc=pcNumber.value
        sv=spectrum.value
        pcfit=np.dot(u[:,0:npc],np.dot(np.diag(s[0:npc]),v[0:npc,:]))
        residuals=intensity[:,sv]-pcfit[:,sv]
        line.set_ydata(intensity[:,sv])
        line.set_label('Spectrum '+str(sv))
        line1.set_ydata(pcfit[:,sv])
        line1.set_label('PCs '+str(npc))
        line2.set_ydata(residuals)
        ax1.legend()
        if cbsave.value==True:
            spectrum.disabled=True
            pcNumber.disabled=True
            data=np.zeros((len(residuals),3))
            data[:,0]=intensity[:,sv]
            data[:,1]=pcfit[:,sv]
            data[:,2]=residuals
            df=pd.DataFrame(data)
            df.index=energy
            df.to_csv(os.path.join(folder,'fit_PCA.csv'), index = True, header=False)
            print('Saved PCA fit')
        elif cbsave.value==False:
            spectrum.disabled=False
            pcNumber.disabled=False
            print(' ')
    ui1=VBox([spectrum,pcNumber])
    ui2=HBox([ui1,cbsave])
    out=widgets.interactive_output(updateCurves,{spectrum.description: spectrum,pcNumber.description:pcNumber,cbsave.description:cbsave})
    display(ui2,out)

def plotRfactor(intensity, u,s,v,folder):
    ncol=np.shape(intensity)[1]
    pcNumber=widgets.BoundedIntText(value=1,min=1,max=ncol,step=1,description='PCs:',disabled=False)
    cbsave=widgets.Checkbox(value=False,description='Save R-factor trend',disabled=False,indent=False)
    pcfit_initial=np.dot(u[:,0:1],np.dot(np.diag(s[0:1]),v[0:1,:]))
    resvalue=xanesRfactor(intensity, pcfit_initial)
    fig,ax=plt.subplots(figsize=(8,7))
    ax.bar(np.arange(ncol),resvalue)
    def updateBarPlot(**args):
        pc=pcNumber.value
        pcfit=np.dot(u[:,0:pc],np.dot(np.diag(s[0:pc]),v[0:pc,:]))
        ax.clear()
        ax.set_xlabel('Scan Number',size=12)
        ax.set_ylabel('Intensity',size=12)
        ax.set_title('%Rfactor trend',size=15)
        resvalue=xanesRfactor(intensity, pcfit)
        ax.bar(np.arange(ncol),resvalue)
        if cbsave.value==True:
            pcNumber.disabled=True
            df=pd.DataFrame(resvalue)
            df.index=np.arange(ncol)
            df.to_csv(os.path.join(folder,'Rfactor_trend.csv'), index = True, header=False)
            print('Saved R-factor trend')
        elif cbsave.value==False:
            pcNumber.disabled=False
            print(' ')
    ui=HBox([pcNumber,cbsave])
    out=widgets.interactive_output( updateBarPlot,{pcNumber.description:pcNumber,cbsave.description:cbsave})
    display(ui,out)

def leastSq(r,ref,pc):
    lsq=[]
    for i in range(len(ref)):
        lsq.append(list(np.dot(np.linalg.pinv(r[:,0:pc]),ref[i])))
    return lsq

def normalize(energy, r):
    emin=np.min(energy)
    emax=np.max(energy)
    return np.sqrt(1./((1./(emax-emin)*(np.trapz((r[:,0])**2,energy)))))

def makeTmMatrix(npc,sliders,nrow,ncol,tref=None,norm=None):
    tm=np.zeros((npc,npc))
    tm_sub=sliders.reshape((nrow,ncol))
    if type(tref) !=type(None):
        nref=np.shape(tref)[1]
        tm[:,:nref]=tref
        if norm !=None: tm[0,:]=norm*np.ones(npc)
    elif type(tref)==type(None):tm[0,:]=norm*np.ones(npc)
    
    if npc==nrow and npc==ncol:
        tm=tm_sub
    elif npc !=nrow and npc==ncol:
        tm[1:,:]=tm_sub
    elif npc==nrow and npc !=ncol:
        tm[:,npc-ncol:]=tm_sub
    elif npc !=nrow and npc !=ncol:
        tm[1:,npc-ncol:]=tm_sub
    return tm


def getPureSpectra(tmMatrix,r,v):
    npc=np.shape(tmMatrix)[0]
    return np.dot(r[:,:npc],tmMatrix), np.transpose(np.dot(np.linalg.pinv(tmMatrix),v[:npc,:]))

def makeSVD(intensity):
    u,s,v=np.linalg.svd(intensity,full_matrices=False)
    return u,s,v

def wmat(c,imp,irank,jvar):
    dm=np.zeros((irank+1, irank+1))
    dm[0,0]=c[jvar,jvar]	
    for k in range(irank):
        kvar=int(imp[k])
        dm[0,k+1]=c[jvar,kvar]
        dm[k+1,0]=c[kvar,jvar] 			
        for kk in range(irank):
            kkvar=int(imp[kk])
            dm[k+1,kk+1]=c[kvar,kkvar]
    return dm

def simplisma(intensity, npc, noise):
    # Adapted from A. Clark's work: https://github.com/usnistgov/pyMCR/pull/10
    nrow,ncol=intensity.shape
    dl = np.zeros((nrow, ncol))
    imp = np.zeros(npc)
    mp = np.zeros(npc)
    w = np.zeros((npc, ncol))
    p = np.zeros((npc, ncol))
    s = np.zeros((npc, ncol))
    noise=noise/100
    mean=np.mean(intensity, axis=0)
    noise=np.max(mean)*noise
    s[0,:]=np.std(intensity, axis=0)
    w[0,:]=(s[0,:]**2)+(mean**2)
    p[0,:]=s[0,:]/(mean+noise)
    imp[0] = int(np.argmax(p[0,:]))
    mp[0] = p[0,:][int(imp[0])]
    l=np.sqrt((s[0,:]**2)+((mean+noise)**2))
    for j in range(ncol):
        dl[:,j]=intensity[:,j]/l[j]
    c=np.dot(dl.T,dl)/nrow
    w[0,:]=w[0,:]/(l**2)
    p[0,:]=w[0,:]*p[0,:]
    s[0,:]=w[0,:]*s[0,:]
    positions=[int(imp[0]+1)]
    purity=[mp[0]]
    for i in range(npc-1):
        for j in range(ncol):
            dm=wmat(c,imp,i+1,j)
            w[i+1,j]=np.linalg.det(dm)
            p[i+1,j]=w[i+1,j]*p[0,j]
            s[i+1,j]=w[i+1,j]*s[0,j]
        imp[i+1] = int(np.argmax(p[i+1,:]))
        mp[i+1] = p[i+1,int(imp[i+1])]
        positions.append(int(imp[i+1]+1))
        purity.append(mp[i+1])
    s_guess=np.zeros((nrow, npc))		
    for i in range(npc):
        s_guess[0:nrow,i]=intensity[0:nrow,int(imp[i])]
    c_guess = np.transpose(np.dot(np.linalg.pinv(s_guess), intensity))
    return s_guess, c_guess, purity, positions


def purestPlot(energy, intensity, npc,noise=5):
    u,sl,v=makeSVD(intensity)
    s=np.diag(sl)
    r=np.dot(u,s)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,7))
    ax1.set_xlabel('Energy',size=12)
    ax1.set_ylabel('Intensity',size=12)
    ax1.set_title('Purest Spectra',size=15)
    ax2.set_xlabel('Scan number',size=12)
    ax2.set_ylabel('Fractions',size=12)
    ax2.set_title('Purest Concentrations',size=15)
    ax2.yaxis.tick_right()
    s,c, purity, position=simplisma(intensity, npc,noise)
    for i in range(np.shape(s)[1]):
        ax1.plot(energy,s[:,i],label='Scan: '+str(position[i])+'\nPurity: \n'+str(trunc(purity[i],4)))
    for j in range(np.shape(c)[1]):
        ax2.plot(c[:,j])
    tref=leastSq(r,[s],npc)
    print('Guessed Transformation Matrix:')
    print(np.array(tref).reshape(npc,npc))
    ax1.legend()
    

class Dataset:
    def __init__(self,xanes,folder_to_save='results'):
        self.original_energy=xanes[:,0] #original energy
        self.original_intensity=xanes[:,1:] #original xanes coefficients
        self.energy=self.original_energy
        self.intensity=self.original_intensity
        self.references=None
        self.manipulated_energy=self.energy #to be used for the interpolation and normalization
        self.manipulated_intensity=self.intensity #to be used for the interpolation and normalization
        cwd=os.getcwd()
        self.folder=os.path.join(cwd,folder_to_save)
        if not os.path.exists(self.folder):os.makedirs(self.folder)
            

    def selectRange(self):
        range_slider = widgets.FloatRangeSlider(value=[np.min(self.original_energy),np.max(self.original_energy)],min=np.min(self.original_energy), max=np.max(self.original_energy), step=0.01,description='Energy:',continuous_update=False)
        cbsave=widgets.Checkbox(value=False,description='Select Range',disabled=False,indent=False)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,7))
        curves1=[];curves2=[]
        ax1.set_xlabel('Energy',size=12)
        ax1.set_ylabel('Intensity',size=12)
        ax1.set_title('Exp. Spectra',size=15)
        ax2.set_xlabel('Energy',size=12)
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Intensity',size=12)
        ax2.set_title('Selected Region',size=15)
        ax2.yaxis.tick_right()
        for i in range(np.shape(self.original_intensity)[1]):
            ax1.plot(self.original_energy,self.original_intensity[:,i],color='tab:blue')
            line1,=ax1.plot(self.original_energy,self.original_intensity[:,i]);line2,=ax2.plot(self.original_energy,self.original_intensity[:,i])
            curves1.append(line1); curves2.append(line2)
        line1=ax1.axvline(np.min(self.original_energy),color='tab:blue',linestyle='--'); line2=ax1.axvline(np.max(self.original_energy),color='tab:blue',linestyle='--')
        def plot_range(**args):
            xlim=range_slider.value
            emin=np.min(xlim); emax=np.max(xlim)
            e1=min(self.original_energy, key=lambda x:abs(x-emin)) 
            e2=min(self.original_energy,key=lambda x:abs(x-emax))
            p1=list(self.original_energy).index(e1);p2=list(self.original_energy).index(e2)
            for k in range(np.shape(self.original_intensity)[1]):
                curves1[k].set_ydata(self.original_intensity[p1:p2,k]); curves1[k].set_xdata(self.original_energy[p1:p2])
                curves2[k].set_ydata(self.original_intensity[p1:p2,k]); curves2[k].set_xdata(self.original_energy[p1:p2])
            line1.set_xdata(self.original_energy[p1]); line2.set_xdata(self.original_energy[p2])
            if cbsave.value==True:
                range_slider.disabled=True
                df=pd.DataFrame(self.original_intensity[p1:p2,:])
                df.index=self.original_energy[p1:p2]
                df.to_csv(os.path.join(self.folder,'selected_xanes.csv'), index = True, header=False)
                self.intensity=self.original_intensity[p1:p2,:]
                self.energy=self.original_energy[p1:p2]
                print('Selected spectra in range (eV/keV):',self.original_energy[p1],':',self.original_energy[p2])
            elif cbsave.value==False:
                range_slider.disabled=False
                print(' ')
        ui = widgets.HBox([range_slider,cbsave])
        out2=widgets.interactive_output(plot_range,{range_slider.description:range_slider,cbsave.description:cbsave})
        display(ui,out2)
        
        
    def manipulateXanes(self, minVal=0.01,maxVal=0.5,stepVal=0.005):
        assert(minVal<=maxVal), "The lower limit of the range is higher of the maximum value"
        assert(stepVal>=0), "The variation step is lower than 0"
        try:
            step_int_slider=widgets.FloatSlider(value=0.5*(maxVal+minVal),min=minVal, max=maxVal, step=stepVal,description='Energy:',continuous_update=False)
            cb1=widgets.Checkbox(value=False,description='Interpolation',disabled=False,indent=False)
            cb2=widgets.Checkbox(value=False,description='Set step',disabled=False,indent=False)
            cb3=widgets.Checkbox(value=False,description='Normalization',disabled=False,indent=False)
            cb4=widgets.Checkbox(value=False,description='Remove corrections',disabled=False,indent=False)
            def energy_options(**args):
                if cb4.value==True:
                    cb1.disabled=True
                    cb3.disabled=True
                    self.manipulated_energy=self.energy
                    self.manipulated_intensity=self.intensity
                elif cb4.value==False:
                    cb1.disabled=False
                    cb3.disabled=False
                    if cb1.value==True:
                        cb4.disabled=True
                        cb3.disabled=True
                        def energy_interpolation(**args):
                            if cb2.value==True:
                                step_int_slider.disabled=True
                                cb3.disabled=False
                                self.manipulated_energy,self.manipulated_intensity=interpolation(self.energy,self.intensity, step_int_slider.value)
                                print('Normalized XANES with energy step: ', step_int_slider.value); print(self.manipulated_intensity)
                                if cb3.value==True: self.manipulated_intensity=normalization(self.manipulated_energy,self.manipulated_intensity); print('Normalized XANES'); print(self.manipulated_intensity)
                            elif cb2.value==False:step_int_slider.disabled=False
                        tab1=HBox(children=[step_int_slider,cb2])    
                        out1=widgets.interactive_output(energy_interpolation,{cb2.description:cb2,step_int_slider.description:step_int_slider})
                        display(tab1,out1)
                    elif cb1.value==False: 
                        cb3.disabled=False
                        if cb3.value==True: self.manipulated_intensity=normalization(self.manipulated_energy,self.manipulated_intensity); print('Normalized XANES'); print(self.manipulated_intensity); cb1.disabled=True;cb4.disabled=True 
                        elif cb3.value==False:cb1.disabled=False; cb4.disabled=False
            tab2=HBox(children=[cb1,cb3,cb4])  
            out2=widgets.interactive_output(energy_options,{cb1.description:cb1,cb3.description:cb3,cb4.description:cb4})
            display(tab2,out2)
        except AssertionError as error:print(error)


    def setReferences(self,energy,intensity,npc):
        assert (npc>0), "The number of Pcs is lower than 1"
        assert(type(npc)==int), "The PCs number is not an integer"
        assert(len(energy)==np.shape(intensity)[0]), "The number of energy points is different from the number of absorption coefficients"
        try: self.selectRef(energy,intensity,npc,self.folder)
        except AssertionError as error:print(error)
        
    
    def selectRef(self, energy, intensity, npc, folder):
        nrow,ncol=np.shape(intensity)
        global refSelection
        fig, ax = plt.subplots(figsize=(8,7))
        refSelection["Number"] = 1
        cNumber = widgets.BoundedIntText(value=1, min=1, max=npc, step=1, description='Reference:', disabled=False)
        cb=widgets.Checkbox(value=False,description='Set these spectra',disabled=False,indent=False)
        def redraw():
            ax.clear()
            ax.set_xlabel('Energy',size=12)
            ax.set_ylabel('Intensity',size=12)
            ax.set_title('References',size=15)
            number = refSelection["Number"]
            curve=[]; controls=[]
            for i in range(number):
                line,=ax.plot(energy,intensity[:,0],label='Spectrum 0')
                curve.append(line)
                controls.append(widgets.BoundedIntText(value=0,min=0,max=ncol,step=1,description='Spectrum:'+str(i),disabled=False))
            ax.legend()
            def changeSpectra(**args):
                global refSelection
                for j1 in range(number):
                    curve[j1].set_ydata(intensity[:,controls[j1].value])
                    curve[j1].set_label('Spectrum '+str(controls[j1].value))
                ax.legend()
                if cb.value==True:
                    cNumber.disabled=True
                    refSpectra=np.zeros((len(energy),number+1))
                    for j2 in range(number):
                        refSpectra[:,j2+1]=intensity[:,controls[j2].value]
                        controls[j2].disabled=True
                    refSpectra[:,0]=energy
                    self.references=refSpectra
                    df=pd.DataFrame(refSpectra)
                    df.to_csv(os.path.join(self.folder,'Selected_references.csv'), index = False, header=False)
                elif cb.value==False:
                    cNumber.disabled=False
                    for j3 in range(number):
                        controls[j3].disabled=False;
                    self.references=None
            controlsDict = {}
            for cc in controls: controlsDict[cc.description] = cc
            ui = widgets.HBox(tuple(controls))
            ui.layout.flex_flow = 'row wrap'
            ui.layout.justify_content = 'space-between'
            ui.layout.align_items = 'flex-start'
            ui.layout.align_content = 'flex-start'
            out1 = widgets.interactive_output(changeSpectra, controlsDict)
            out1b = widgets.interactive_output(changeSpectra,{cb.description:cb})
            display(ui,cb,out1b,out1)
        def changeNumber(**args):
            global refSelection
            refSelection["Number"]=cNumber.value
            redraw()
        out2=widgets.interactive_output(changeNumber, {cNumber.description:cNumber})
        display(cNumber,out2)
        


class PCA:
    def __init__(self,folder_to_save='results'):
        self.components=None
        self.e_pca=None
        self.s_pca=None
        self.c_pca=None
        cwd=os.getcwd()
        self.folder=os.path.join(cwd,folder_to_save)
        if not os.path.exists(self.folder):os.makedirs(self.folder)

    def plotAbstracts(self,energy,abstract,abstractScaled,folder):
        global plotPCAcomponents
        plotPCAcomponents={}
        ncol=np.shape(abstract)[1]
        fig, ax = plt.subplots(figsize=(8,7))
        plotPCAcomponents["Switching"] = "Not Weighted"
        plotPCAcomponents["Component"] = 1
        radio = widgets.RadioButtons( options=['Not Weighted', 'Weighted'], description='Switching:', disabled=False)
        componentWidget = widgets.BoundedIntText(value=1, min=1, max=ncol, step=1, description='Component:', disabled=False)
        cbsave=widgets.Checkbox(value=False,description='Save components',disabled=False,indent=False)
        def redraw():
            ax.clear()
            ax.set_xlabel('Energy',size=12)
            ax.set_ylabel('Intensity',size=12)
            ax.set_title('Abstract Components',size=15)
            switching = plotPCAcomponents["Switching"]
            components=abstract if switching=='Not Weighted' else abstractScaled
            control = plotPCAcomponents["Component"]
            pcName=[]
            for i in range(control):
                pcName.append('PC '+str(i+1))
            for j in range(control):
                ax.plot(energy, components[:,j],label=pcName[j])
            self.components=components[:,:control]
            ax.legend()
        def switch(**args):
            global plotPCAcomponents
            plotPCAcomponents["Switching"] = radio.value
            redraw()
        def changeComp(**args):
            global plotPCAcomponents
            plotPCAcomponents["Component"] = componentWidget.value
            redraw()
        def saveComponent(**args):
            if cbsave.value==True:
                radio.disabled=True
                componentWidget.disabled=True
                df=pd.DataFrame(self.components)
                df.index=energy
                if  radio.value=='Not Weighted':
                    df.to_csv(os.path.join(folder,'Not_weighted_components.csv'), index = True, header=False)
                    print('Saved un-weighted components')
                elif radio.value=='Weighted':
                    df.to_csv(os.path.join(folder,'Weighted_components.csv'), index = True, header=False)
                    print('Saved weighted components')
            elif cbsave.value==False:
                radio.disabled=False
                componentWidget.disabled=False
                print(' ')
        out1=widgets.interactive_output(switch, {radio.description:radio})     
        out2=widgets.interactive_output(changeComp, {componentWidget.description:componentWidget})
        out3=widgets.interactive_output(saveComponent, {cbsave.description:cbsave})
        ui = widgets.HBox([radio, componentWidget,cbsave])
        display(ui,out1,out2,out3)


    def componentAnalysis(self,energy,intensity):
        assert(len(energy)==np.shape(intensity)[0]), "The number of energy points is different from the number of absorption coefficients"
        try:
            u,s,v=makeSVD(intensity)
            sdiag=np.diag(s)
            if np.shape(intensity)[0]<np.shape(intensity)[1]:
                abstract=np.transpose(v)
                abstractScaled=np.transpose(np.dot(sdiag,v))
            else:
                abstract=u
                abstractScaled=np.dot(abstract,sdiag)
            self.plotAbstracts(energy,abstract,abstractScaled,self.folder)
        except AssertionError as error:print(error)

    def getStatistic(self,intensity):
        u,s,v=makeSVD(intensity)
        if np.shape(intensity)[0]<np.shape(intensity)[1]: intensity=np.transpose(intensity)
        nrow,ncol=np.shape(intensity)
        l=(s**2)/(nrow-1)
        ind,ie=malinowsky(l,nrow,ncol)
        fisher=fisherFunction(l,nrow,ncol)
        plotStatistic(s,ind,ie,fisher,self.folder)
        
    def pcaFit(self,energy,intensity):
        assert(len(energy)==np.shape(intensity)[0]), "The number of energy points is different from the number of absorption coefficients"
        try: u,s,v=makeSVD(intensity); makePCAfit(energy,intensity,u,s,v,self.folder)
        except AssertionError as error:print(error)

    def rfactorTrend(self,intensity):
        u,s,v=makeSVD(intensity)
        plotRfactor(intensity, u,s,v,self.folder)
        
    
    def guessTransformationMatrix(self,energy,intensity,npc):
        assert (npc>0), "The number of Pcs is lower than 1"
        assert(type(npc)==int), "The PCs number is not an integer"
        assert(len(energy)==np.shape(intensity)[0]), "The number of energy points is different from the number of absorption coefficients"
        try: purestPlot(energy, intensity, npc)
        except AssertionError as error:print(error)
        
    
    def transformationMatrix(self,energy,intensity,npc,references=None,minVal=-5,maxVal=5,stepVal=0.1):
        assert (npc>0), "The number of Pcs is lower than 1"
        assert(type(npc)==int), "The PCs number is not an integer"
        assert(len(energy)==np.shape(intensity)[0]), "The number of energy points is different from the number of absorption coefficients"
        assert(minVal<=maxVal), "The lower limit of the range is higher of the maximum value"
        assert(stepVal>=0), "The variation step is lower than 0"
        if type(references) != type(None):
            assert (np.shape(references)[1]-1<=npc), "The number of Pcs is lower than the selected number of references"
            assert(np.shape(references)[0]==np.shape(intensity)[0]), "Spectra and references are interpolated over two different energy grids"
        try: self.tmPlot(energy,intensity,npc,references,minVal,maxVal,stepVal)
        except AssertionError as error:print(error)
        

    def tmPlot(self, energy,intensity,npc,ref_list,minVal,maxVal,stepVal):
        u,sl,v=makeSVD(intensity)
        s=np.diag(sl)
        r=np.dot(u,s)
        w=v
        self.e_pca=energy
        cb=widgets.Checkbox(value=False,description='Normalization',disabled=False,indent=False)
        cbflip=widgets.Checkbox(value=False,description='Flip',disabled=False,indent=False)
        cbFirst=widgets.Checkbox(value=False,description='First Spectrum',disabled=False,indent=False)
        cbLast=widgets.Checkbox(value=False,description='Last Spectrum',disabled=False,indent=False)
        cbSave=widgets.Checkbox(value=False,description='Save Components',disabled=False,indent=False)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,7))
        ax1.set_xlabel('Energy',size=12)
        ax1.set_ylabel('Intensity',size=12)
        ax1.set_title('XANES',size=15)
        ax2.set_title('Concentrations',size=15)
        ax2.set_xlabel('Scan number',size=12)
        ax2.set_ylabel('Fractions',size=12)
        ax2.yaxis.tick_right()
        s_initial=np.zeros((len(energy),npc))
        c_initial=np.zeros((np.shape(intensity)[1],npc))
        curves1=[]; curves2=[]
        pcName=[]
        for n in range(npc):
            pcName.append('Component '+str(n+1))
        for i in range(npc):
            line1,=ax1.plot(energy,s_initial[:,i],label=pcName[i])
            line2,=ax2.plot(c_initial[:,i])
            curves1.append(line1); curves2.append(line2)
        ax1.legend()
        items=sliderDescription(npc, minVal=minVal, maxVal=maxVal,stepVal=stepVal)
        itemMatrix=[items[i:i+npc] for i in range(len(items))[::npc]]
        def updateCurves(**args):
            for i in range(npc):
                for j in range(npc):
                    itemMatrix[i][j][2].min = itemMatrix[i][j][0].value[0]
                    itemMatrix[i][j][2].max = itemMatrix[i][j][0].value[1]
                    itemMatrix[i][j][2].step = itemMatrix[i][j][1].value
                    
            if type(ref_list)==type(None):
                if cb.value==True and cbFirst.value==False and cbLast.value==False:
                    cbflip.disabled=False
                    nrow=npc-1; ncol=npc
                    norm=normalize(energy, r)
                    tref=None
                    if cbflip.value==True: 
                        norm=-1*norm  
                    sliders=[]
                    for i in range(1,npc):
                        for j in range(npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=tref,norm=norm)
                    for k1 in range(npc):
                         itemMatrix[0][k1][0].layout.visibility = "hidden"
                         itemMatrix[0][k1][1].layout.visibility = "hidden"
                         itemMatrix[0][k1][2].layout.visibility = "hidden"
                         
                    for k2 in range(1,npc): 
                        itemMatrix[k2][0][0].layout.visibility = "visible"
                        itemMatrix[k2][0][1].layout.visibility = "visible"
                        itemMatrix[k2][0][2].layout.visibility = "visible"
                        itemMatrix[k2][1][0].layout.visibility = "visible"
                        itemMatrix[k2][1][1].layout.visibility = "visible"
                        itemMatrix[k2][1][2].layout.visibility = "visible"
       
                if cb.value==False and cbFirst.value==True and cbLast.value==False: #first spectrum fixed
                    nrow=npc; ncol=npc-1
                    norm=None
                    first=intensity[:,0]
                    tref=leastSq(r,[first],npc)
                    sliders=[]
                    for i in range(npc):
                        for j in range(1,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=tref,norm=norm)
                    if cbflip.value==True:
                        tm[0,:]=-1*tm[0,:]
                    for k in range(npc):
                        itemMatrix[k][0][0].layout.visibility = "hidden"
                        itemMatrix[k][0][1].layout.visibility = "hidden"
                        itemMatrix[k][0][2].layout.visibility = "hidden"
                        itemMatrix[k][1][0].layout.visibility = "visible"
                        itemMatrix[k][1][1].layout.visibility = "visible"
                        itemMatrix[k][1][2].layout.visibility = "visible"

                if cb.value==False and cbLast.value==True and cbFirst.value==False: #last spectrum fixed
                    nrow=npc; ncol=npc-1
                    norm=None
                    last=intensity[:,np.shape(intensity)[1]-1]
                    tref=leastSq(r,[last],npc)
                    sliders=[]
                    for i in range(npc):
                        for j in range(1,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=np.array(tref),norm=norm)
                    if cbflip.value==True:
                        tm[0,:]=-1*tm[0,:]
                    for k in range(npc):
                        itemMatrix[k][0][0].layout.visibility = "hidden"
                        itemMatrix[k][0][1].layout.visibility = "hidden"
                        itemMatrix[k][0][2].layout.visibility = "hidden"
                        itemMatrix[k][1][0].layout.visibility = "visible"
                        itemMatrix[k][1][1].layout.visibility = "visible"
                        itemMatrix[k][1][2].layout.visibility = "visible" 

                if cb.value==False and cbFirst.value==True and cbLast.value==True: #first and last spectra fixed
                    nrow=npc; ncol=npc-2
                    norm=None
                    first=intensity[:,0]
                    last=intensity[:,np.shape(intensity)[1]-1]
                    tref=leastSq(r,[first,last],npc)
                    sliders=[]
                    for i in range(npc):
                        for j in range(2,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=np.transpose(tref),norm=norm)
                    if cbflip.value==True:
                        tm[0,:]=-1*tm[0,:]
                    for k in range(npc):
                        itemMatrix[k][0][0].layout.visibility = "hidden"
                        itemMatrix[k][0][1].layout.visibility = "hidden"
                        itemMatrix[k][0][2].layout.visibility = "hidden"
                        itemMatrix[k][1][0].layout.visibility = "hidden"
                        itemMatrix[k][1][1].layout.visibility = "hidden"
                        itemMatrix[k][1][2].layout.visibility = "hidden" 
                        
                if cb.value==True and cbFirst.value==True and cbLast.value==False: #normalization and first spectrum fixed
                    cbflip.disabled=False
                    nrow=npc-1; ncol=npc-1
                    norm=normalize(energy, r)
                    first=intensity[:,0]
                    tref=leastSq(r,[first],npc)
                    if cbflip.value==True: 
                        norm=-1*norm
                    sliders=[]
                    for i in range(1,npc):
                        for j in range(1,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=tref,norm=norm)
                    for k1 in range(npc):
                        itemMatrix[0][k1][0].layout.visibility = "hidden"
                        itemMatrix[0][k1][1].layout.visibility = "hidden"
                        itemMatrix[0][k1][2].layout.visibility = "hidden"
                    for k2 in range(1,npc):
                        itemMatrix[k2][0][0].layout.visibility = "hidden"
                        itemMatrix[k2][0][1].layout.visibility = "hidden"
                        itemMatrix[k2][0][2].layout.visibility = "hidden"
                        itemMatrix[k2][1][0].layout.visibility = "visible"
                        itemMatrix[k2][1][1].layout.visibility = "visible"
                        itemMatrix[k2][1][2].layout.visibility = "visible" 

                if cb.value==True and cbLast.value==True and cbFirst.value==False:
                     cbflip.disabled=False 
                     nrow=npc-1; ncol=npc-1
                     norm=normalize(energy, r)
                     last=intensity[:,np.shape(intensity)[1]-1]
                     tref=leastSq(r,[last],npc)
                     if cbflip.value==True: 
                         norm=-1*norm
                     sliders=[]
                     for i in range(1,npc):
                         for j in range(1,npc): sliders.append(itemMatrix[i][j][2].value)
                     tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=np.array(tref),norm=norm)
                     for k1 in range(npc):
                         itemMatrix[0][k1][0].layout.visibility = "hidden"
                         itemMatrix[0][k1][1].layout.visibility = "hidden"
                         itemMatrix[0][k1][2].layout.visibility = "hidden"
                     for k2 in range(1,npc):
                         itemMatrix[k2][0][0].layout.visibility = "hidden"
                         itemMatrix[k2][0][1].layout.visibility = "hidden"
                         itemMatrix[k2][0][2].layout.visibility = "hidden" 
                         itemMatrix[k2][1][0].layout.visibility = "visible"
                         itemMatrix[k2][1][1].layout.visibility = "visible"
                         itemMatrix[k2][1][2].layout.visibility = "visible" 
                         
                if cb.value==True and cbFirst.value==True and cbLast.value==True:
                    cbflip.disabled=False
                    nrow=npc-1; ncol=npc-2
                    norm=normalize(energy, r)
                    first=intensity[:,0]
                    last=intensity[:,np.shape(intensity)[1]-1]
                    tref=leastSq(r,[first,last],npc)
                    if cbflip.value==True: 
                        norm=-1*norm
                    sliders=[]
                    for i in range(1,npc):
                        for j in range(2,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref=np.transpose(tref),norm=norm)
                    for k1 in range(npc):
                        itemMatrix[0][k1][0].layout.visibility = "hidden"
                        itemMatrix[0][k1][1].layout.visibility = "hidden"
                        itemMatrix[0][k1][2].layout.visibility = "hidden"
                    for k2 in range(1,npc):
                        itemMatrix[k2][0][0].layout.visibility = "hidden"
                        itemMatrix[k2][0][1].layout.visibility = "hidden"
                        itemMatrix[k2][0][2].layout.visibility = "hidden"
                        itemMatrix[k2][1][0].layout.visibility = "hidden"
                        itemMatrix[k2][1][1].layout.visibility = "hidden"
                        itemMatrix[k2][1][2].layout.visibility = "hidden"
                        
                if cb.value==False and cbFirst.value==False and cbLast.value==False: #all free
                    nrow=npc; ncol=npc
                    norm=None;tref=None
                    sliders=[]
                    for i in range(npc):
                        for j in range(npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=np.array(sliders).reshape((npc,npc))
                    if cbflip.value==True:
                        tm[0,:]=-1*tm[0,:]
                    for k in range(npc):
                        itemMatrix[0][k][0].layout.visibility = "visible"
                        itemMatrix[0][k][1].layout.visibility = "visible"
                        itemMatrix[0][k][2].layout.visibility = "visible"
                        itemMatrix[k][0][0].layout.visibility = "visible"
                        itemMatrix[k][0][1].layout.visibility = "visible"
                        itemMatrix[k][0][2].layout.visibility = "visible"
                        itemMatrix[k][1][0].layout.visibility = "visible"
                        itemMatrix[k][1][1].layout.visibility = "visible"
                        itemMatrix[k][1][2].layout.visibility = "visible"                            

                
            ### with references selected in a separate routine 
            elif type(ref_list) != type(None):
                references=ref_list[:,1:]
                if cb.value==False:
                    nrow=npc; ncol=npc-np.shape(references)[1]
                    norm=None
                    tref=leastSq(r,[references],npc)
                    cbFirst.disabled=True; cbLast.disabled=True
                    sliders=[]
                    for i in range(npc):
                        for j in range(npc-ncol,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref[0],norm=norm)
                    if cbflip.value==True:
                        tm[0,:]=-1*tm[0,:]
                for k1 in range(npc):
                    for k2 in range(np.shape(references)[1]):
                        itemMatrix[k1][k2][0].layout.visibility = "hidden"
                        itemMatrix[k1][k2][1].layout.visibility = "hidden"
                        itemMatrix[k1][k2][2].layout.visibility = "hidden"
                
                if cb.value==True:
                    nrow=npc-1; ncol=npc-np.shape(references)[1]
                    norm=normalize(energy, r)
                    tref=leastSq(r,[references],npc)
                    if cbflip.value==True:
                        norm=-1*norm
                    sliders=[]
                    for i in range(1,npc):
                        for j in range(npc-ncol,npc): sliders.append(itemMatrix[i][j][2].value)
                    tm=makeTmMatrix(npc,np.array(sliders),nrow,ncol,tref[0],norm=norm)
                    
                    for k1 in range(1,npc):
                        for k2 in range(np.shape(references)[1]):
                            itemMatrix[k1][k2][0].layout.visibility = "hidden"
                            itemMatrix[k1][k2][1].layout.visibility = "hidden"
                            itemMatrix[k1][k2][2].layout.visibility = "hidden"
                    for k3 in range(npc):
                        itemMatrix[0][k3][0].layout.visibility = "hidden"
                        itemMatrix[0][k3][1].layout.visibility = "hidden"
                        itemMatrix[0][k3][2].layout.visibility = "hidden"

            print('Transformation Matrix:')
            print(tm)
            print('Rank:',np.linalg.matrix_rank(tm))
            s,c=getPureSpectra(tm,r,w)
            self.s_pca=s
            self.c_pca=c
            
            if cbSave.value==True:
                dfs=pd.DataFrame(self.s_pca)
                dfc=pd.DataFrame(self.c_pca)
                dfs.index=self.e_pca
                dfc.index=np.arange(0,np.shape(self.c_pca)[0])
                dfs.to_csv(os.path.join(self.folder,'components_spectra.csv'), index = True, header=False)
                dfc.to_csv(os.path.join(self.folder,'components_concentrations.csv'), index = True, header=False)
                for i in range(npc):
                    for j in range(npc):
                        itemMatrix[i][j][0].disabled=True
                        itemMatrix[i][j][1].disabled=True
                        itemMatrix[i][j][2].disabled=True
                if type(ref_list)== type(None): 
                    cb.disabled=True
                    cbflip.disabled=True
                    cbFirst.disabled=True
                    cbLast.disabled=True
                elif type(ref_list) != type(None):
                    cb.disabled=True
                    cbflip.disabled=True
                
                #if cbflip.disabled==False:cbflip.disabled=True

            elif cbSave.value==False:
                for i in range(npc):
                    for j in range(npc):
                        itemMatrix[i][j][0].disabled=False
                        itemMatrix[i][j][1].disabled=False
                        itemMatrix[i][j][2].disabled=False
                if type(ref_list)== type(None):
                    cb.disabled=False
                    cbflip.disabled=False
                    cbFirst.disabled=False
                    cbLast.disabled=False
                elif type(ref_list) != type(None):
                    cb.disabled=False
                    cbflip.disabled=False

            maxs=np.max(s); mins=np.min(s)
            maxc=np.max(c); minc=np.min(c)
            for k in range(npc):
                curves1[k].set_ydata(s[:,k])
                curves2[k].set_ydata(c[:,k])
                ax1.set_ylim(mins-0.01*mins-0.01,maxs+0.01*maxs+0.01)
                ax2.set_ylim(minc-0.01*minc-0.01,maxc+0.01*maxc+0.01)
        allItems=[]
        for l1 in range(npc):
            for l2 in range(npc):
                allItems.append(HBox(itemMatrix[l1][l2]))
        ui=VBox(allItems)
        ui.layout.flex_flow = 'row wrap'
        ui.layout.justify_content = 'space-between'
        ui.layout.align_items = 'flex-start'
        ui.layout.align_content = 'flex-start'
        cbChildren=HBox([cb,cbflip,cbFirst,cbLast,cbSave])
        controlsDict = {cb.description:cb,cbflip.description:cbflip,cbFirst.description:cbFirst,cbLast.description:cbLast,cbSave.description:cbSave}
        for l3 in range(npc):
            for l4 in range(npc):
                for cc in itemMatrix[l3][l4]: controlsDict[cc.description] = cc
        out=widgets.interactive_output(updateCurves,controlsDict)
        ch=VBox([cbChildren,ui])
        display(ch,out)
    

        
        


                

        

                

        

            

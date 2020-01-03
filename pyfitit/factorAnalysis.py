import numpy as np
import pandas as pd
import matplotlib, os
if (os.name != 'nt') and ('DISPLAY' not in os.environ): matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import f
from ipywidgets import interact
from ipywidgets import widgets #
from IPython.display import display, Javascript
from IPython.display import Math, Latex
from ipywidgets import Button
from numpy.linalg import matrix_rank


figureSize = None

class Spectra:
    def __init__(self, xanes, reference, num):
        # Load Data
        self.data = xanes
        # Load References
        ref = reference
        # Number of factors
        numFactors=num
        ###########################################################
        #                           PCA                           #
        ###########################################################

        u, s, v = np.linalg.svd(self.data, full_matrices=False)
        self.u_red = u[:,:numFactors]
        s_red = np.diag(s[:numFactors])
        v_red = v[:numFactors,:]
        self.concM = np.dot(s_red, v_red)
        self.mat_T0 = np.linalg.lstsq(self.u_red, ref)[0]
        invT = np.linalg.inv(self.mat_T0)

    def rotation(self, mat_X):
        mat_T = self.mat_T0 + mat_X
        print(' ')
        print('Rotational Matrix: ')
        print(mat_T)
        print(' ')
        sp = np.dot(self.u_red, mat_T)
        inv_mat_T = np.linalg.inv(mat_T)
        conc = np.dot(inv_mat_T, self.concM)
        sp = np.transpose(sp)
        return sp, conc

    def norm_rotation(self,mat_X):
        mat_TN=mat_X
        print(' ')
        print('Normalized Rotational Matrix: ')
        print(mat_TN)
        print(' ')
        spN=np.dot(self.u_red, mat_TN)
        inv_mat_TN = np.linalg.inv(mat_TN)
        concN = np.dot(inv_mat_TN, self.concM)
        spN=np.transpose(spN)
        return spN, concN

    def get_spectrum(self, mat_X):
        sp, conc = self.rotation(mat_X)
        sp=np.transpose(sp)
        conc=np.transpose(conc)
        return sp, conc

    def get_spectrumN(self,mat_X):
        spN,concN=self.norm_rotation(mat_X)
        spN=np.transpose(spN)
        concN=np.transpose(concN)
        return spN, concN

def calcSVD(data):
    n_row=np.shape(data)[0]; #numbers of rows
    n_col=np.shape(data)[1]; #numbers of columns
    u,s,vT=np.linalg.svd(data,full_matrices=False)
    uz,sz,vTz=np.linalg.svd(data,full_matrices=False)
    v= vT.T #  Principal Directions (each column is an eigenvector)
    s_S = np.diag(s)
    l=(s**2)/(n_row-1)  # Eigenvalues of the correlation matrix
    principal_components=np.dot(u,s_S)
    #principal_components_v=np.dot(u,s_S)
    return principal_components,l

def MalinowskyParameters(data, l):
    if np.shape(data)[0]<np.shape(data)[1]:
        data=np.transpose(data)
    n_row=np.shape(data)[0]; #numbers of rows
    n_col=np.shape(data)[1]; #numbers of columns
    ind=np.zeros(n_col-1)
    ie=np.zeros(n_col-1)
    index=range(1,n_col)
    for i in range(0,n_col-1):
        ind[i]=(np.sqrt((np.sum(l[i+1:n_col]))/(n_row*(n_col-index[i]))))/(n_col-index[i])**2
        ie[i]=np.sqrt(index[i]*(np.sum(l[i+1:n_col]))/(n_row*n_col*(n_col-index[i])))
    pc=np.arange(1., n_col+1, 1) # maximum number of cumponents (i.e number of spectra)
    p=np.zeros(np.size(pc))
    for i in range(0,np.size(pc)):
        p[i]=(n_row-pc[i]+1)*(n_col-pc[i]+1)
    s1=np.zeros(np.size(pc))
    s2=np.zeros(np.size(pc))
    fi=np.zeros(np.size(l)-1)
    result=np.zeros(np.size(l)-1)
    a=pc+1
    for i in range(0,n_col-1):
           s1[i]=np.sum((n_row-a[i:np.size(pc)]+1)*(n_col-a[i:np.size(pc)]+1))
    for j in range(0,n_col-1):
           s2[j]=np.sum(l[j+1:np.size(pc)+1])
    for i in range(0,n_col-1):
         fi[i]=(s1[i]/p[i])*(l[i]/s2[i])
    for i in range(0,n_col-1):
          result[i]=((integrate.quad(lambda x: f.pdf(x, 1, (n_col-1)-i), fi[i], np.inf))[0])*100
    statistic=pd.DataFrame({'IND': ind, 'IE': ie, 'F': result})
    statistic.index=statistic.index+1
    return statistic, pc

def plotTestStatistic(statistic, pc, l):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(8,5))
    ax1.semilogy(pc,l,'o-',color='green')
    #ax1.set_xlabel('PCs',fontweight='bold')
    ax1.set_ylabel('Variance',fontweight='bold')
    ax1.set(title='Scree')
    ax1.grid()

    ax2.semilogy(statistic['IND'], 'o-')
    #ax2.set_xlabel('PCs',fontweight='bold')
    ax2.set(title='IND Plot')
    ax2.set_ylabel('IND',fontweight='bold')
    ax2.yaxis.tick_right()
    ax2.grid()

    ax3.plot(statistic['F'],'o-',color='red')
    ax3.set_xlabel('PCs',fontweight='bold')
    ax3.set_ylabel('S.L. %',fontweight='bold')
    ax3.set(title='F')
    ax3.grid()

    ax4.semilogy(statistic['IE'], 'o-',color='blue')
    ax4.set_xlabel('PCs',fontweight='bold')
    ax4.set_ylabel('IE',fontweight='bold')
    ax4.yaxis.tick_right()
    ax4.set(title='IE')
    ax4.grid()

    # fig.tight_layout()
    # plt.show()

def recommendPCnumber(statistic):
    print(" ")
    print("Min IND value: ",statistic["IND"].min())
    print("Number of PCs suggested by IND-factor: ",statistic["IND"].idxmin(), "PC")
    print(" ")
    print("Highest SL(%) < 5%: ",max(statistic.loc[statistic.F<5,'F']))
    print("Number of PCs suggested by F-Test: ", statistic.loc[statistic.F<5,'F'].idxmax(), "PC")
    print(" ")




def Norm(n_spectrum,us,vt,mat_X,NumFactors,n_sliders,data,energy,min_val,max_val,step_val, guiClass):

    def update_Norm(**xvalor):
        xvalor=[]
        guiClass.params['x'] = {}
        for i in range(n_sliders):
            xvalor.append(controls[i].value)
            guiClass.params['x'][controls[i].description] = controls[i].value
        if n_spectrum=="Normalization":
            mat_X[1:,:] = np.reshape(xvalor,(NumFactors-1,NumFactors))
        elif n_spectrum=="Norm. and 1st spectrum":
            mat_X[1:,1:] = np.reshape(xvalor,(NumFactors-1,NumFactors-1))
        elif n_spectrum=="Norm. and Last spectrum":
            mat_X[1:,:NumFactors-1] = np.reshape(xvalor,(NumFactors-1,NumFactors-1))
        elif n_spectrum=="Norm., 1s and last spectrum":
            mat_X[1:,1:NumFactors-1] = np.reshape(xvalor,(NumFactors-1,NumFactors-2))

        print('ROTATIONAL MATRIX:')
        print(' ')
        print(mat_X)
        sp=np.dot(us,mat_X)
        nc=np.shape(sp)[1]
        rank=np.linalg.matrix_rank(mat_X)
        print('RANK:',rank)
        if rank<NumFactors:
            #ar,ac = mat_X.shape
            #i = np.eye(ac, ac)
            #mat_X_inv=np.linalg.lstsq(mat_X, i,rcond=None)[0]
            mat_X_inv=np.linalg.pinv(mat_X)
        elif rank==NumFactors:
            mat_X_inv=np.linalg.inv(mat_X)
        conc=np.transpose(np.dot(mat_X_inv,vt))
        new_xanes=pd.DataFrame(sp)
        new_xanes.index=energy
        new_concentrations=pd.DataFrame(conc)
        axs.clear()
        axs.set_xlim([min(energy),max(energy)])
        axs.set_xlabel("Energy (eV)",fontweight='bold')
        axs.set_ylabel("Absorption",fontweight='bold')
        axs.set(title="Pure XANES")
        pcname=[]
        for i in range(len(new_xanes.columns)):
            pcn="PC%i" % (i%len(new_xanes.columns)+1)
            pcname.append(pcn)
        new_xanes.columns=pcname
        new_concentrations.columns=pcname
        axc.clear()
        axc.set_xlabel("Scan Index",fontweight='bold')
        axc.set_ylabel("Fraction of Pure Components",fontweight='bold')
        axc.yaxis.set_label_position("right")
        axc.set(title="Pure Concentrations")
        new_xanes.plot(ax=axs, linewidth=2)
        new_concentrations.plot(ax=axc,linewidth=2)
        guiClass.pureSpectra = sp
        guiClass.pureConcentrations = conc

    if guiClass.fig is not None: plt.close(guiClass.fig)
    fig = plt.figure(figsize=figureSize)
    guiClass.fig = fig
    axs = fig.add_subplot(121)
    axc = fig.add_subplot(122)
    # Setup Widgets
    controls=[]
    o='vertical'
    for i in range(n_sliders):
        title="t%i" % (i%n_sliders+1)
        sl=widgets.FloatSlider(description=title,min=min_val, max=max_val, step=step_val,orientation=o,continuous_update=False) #change range
        controls.append(sl)
    controlsDict = {}
    for c in controls:
        controlsDict[c.description] = c
    uif = widgets.HBox(tuple(controls))
    outf = widgets.interactive_output(update_Norm,controlsDict)
    uif.layout.flex_flow = 'row wrap'
    uif.layout.justify_content = 'space-between'
    uif.layout.align_items = 'flex-start'
    uif.layout.align_content = 'flex-start'
    display(uif, outf)
    display(Javascript('$(this.element).addClass("pcaOutput");'))


def unNorm(s_spectrum,us,vt,mat_X_initial,NumFactors,n_sliders,data,energy,min_val,max_val,step_val, guiClass):
    def update_Norm(**arg):
        global mat_X
        mat_X = mat_X_initial
        xvalor=[]
        guiClass.params['x'] = {}
        for i in range(n_sliders):
            xvalor.append(controls[i].value)
            guiClass.params['x'][controls[i].description] = controls[i].value
        xvalor = np.array(xvalor)
        if s_spectrum=="No Constraints":
            values=np.reshape(xvalor,(NumFactors,NumFactors))
            mat_X=np.transpose(values)
        elif s_spectrum=="1st spectrum fixed":
            values=np.reshape(xvalor,(NumFactors,NumFactors-1))
            mat_X[:,1:] = values
        elif s_spectrum=="Last spectrum fixed":
            xvalor=np.reshape(xvalor,(NumFactors,NumFactors-1))
            mat_X[:,0:NumFactors-1]=xvalor
        elif s_spectrum=="1st and Last spectrum fixed":
            values=np.reshape(xvalor,(NumFactors,NumFactors-2))
            mat_X[:,1:NumFactors-1]=values
        print('ROTATIONAL MATRIX:')
        print(' ')
        print(mat_X)
        sp=np.dot(us,mat_X)
        nc=np.shape(sp)[1]
        rank=np.linalg.matrix_rank(mat_X)
        print('RANK:',rank)
        if rank==0:
            print(' ')
            print('NULL MATRIX')
            print(' ')
        else:
            if rank<NumFactors:
                #ar,ac = mat_X.shape
                #i = np.eye(ac, ac)
                #mat_X_inv=np.linalg.lstsq(mat_X, i,rcond=None)[0]
                mat_X_inv=np.linalg.pinv(mat_X)
            elif rank==NumFactors:
                mat_X_inv=np.linalg.inv(mat_X)
            conc=np.transpose(np.dot(mat_X_inv,vt))
            new_xanes=pd.DataFrame(sp)
            new_xanes.index=energy
            new_concentrations=pd.DataFrame(conc)
            axs.clear()
            axs.set_xlim([min(energy),max(energy)])
            axs.set_xlabel("Energy (eV)",fontweight='bold')
            axs.set_ylabel("Absorption",fontweight='bold')
            axs.set(title="Pure XANES")
            pcname=[]
            for i in range(len(new_xanes.columns)):
            	pcn="PC%i" % (i%len(new_xanes.columns)+1)
            	pcname.append(pcn)
            new_xanes.columns=pcname
            new_concentrations.columns=pcname
            axc.clear()
            axc.set_xlabel("Scan Index",fontweight='bold')
            axc.set_ylabel("Fraction of Pure Components",fontweight='bold')
            axc.yaxis.set_label_position("right")
            axc.set(title="Pure Concentrations")
            new_xanes.plot(ax=axs, linewidth=2)
            new_concentrations.plot(ax=axc,linewidth=2)
            guiClass.pureSpectra = sp
            guiClass.pureConcentrations = conc

    if guiClass.fig is not None: plt.close(guiClass.fig)
    fig = plt.figure(figsize=figureSize)
    guiClass.fig = fig
    axs = fig.add_subplot(121)
    axc = fig.add_subplot(122)
    # Setup Widgets
    controls=[]
    o='vertical'
    for i in range(n_sliders):
        title="t%i" % (i%n_sliders+1)
        sl=widgets.FloatSlider(description=title,min=min_val, max=max_val, step=step_val,orientation=o,continuous_update=False) #change range
        controls.append(sl)
    controlsDict = {}
    for c in controls:
        controlsDict[c.description] = c
    uif = widgets.HBox(tuple(controls))
    outf = widgets.interactive_output(update_Norm,controlsDict)
    uif.layout.flex_flow = 'row wrap'
    uif.layout.justify_content = 'space-between'
    uif.layout.align_items = 'flex-start'
    uif.layout.align_content = 'flex-start'
    display(uif, outf)
    display(Javascript('$(this.element).addClass("pcaOutput");'))









#if __name__ == "__main__":
#    sp = Spectra("for_PCA.dat", "references.dat")

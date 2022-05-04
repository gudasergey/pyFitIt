![Logo](http://hpc.nano.sfedu.ru/pyfitit/assets/logo.png)
# pyFitIt
Python implementation of FitIt software to fit X-ray absorption near edge structure (XANES) and other spectra. The python version is extended with additional features: machine learning, automatic component analysis, joint convolution fitting and others.

[PyFitIt website](http://hpc.nano.sfedu.ru/pyfitit/)

## Features
- Uses ipywidgets to construct the portable GUI
- Calculates spectra by [FDMNES](http://neel.cnrs.fr/spip.php?rubrique1007&lang=en) or [FEFF](http://monalisa.phys.washington.edu/) or [ADF](https://www.scm.com/product/adf/) or [pyGDM](https://wiechapeter.gitlab.io/pygdm/2018-01-15-pygdm/)
- Interpolates spectra in order to speedup fitting. Support different types of interpolation point generation: grid, random, [IHS](http://people.sc.fsu.edu/~jburkardt/cpp_src/ihs/ihs.html), adaptive and various interpolation methods including machine learning algorithms
- Using multidimensional interpolation approximation you can vary parameters by sliders and see immediately theoretical spectrum, which corresponds to this geometry. Fitting can be performed on the basis of visual comparison with experiment or using automatic procedure and quantitative criteria.
- Supports direct prediction of geometry parameters by machine learning
- Includes automatic and semi-automatic component analysis

## Installation
pip install pyfitit

## Usage
See examples folder in this repository.

### This project is developing thanks to
- Grigory Smolentsev
- Sergey Guda
- Alexandr Soldatov
- Carlo Lamberti
- Alexander Guda
- Oleg Usoltsev
- Yury Rusalev
- Andrea Martini
- Alexandr Algasov
- Danil Pashkov
- Aram Bugaev
- Mikhail Soldatov

### If you like the software acknowledge it using the references below:

[A.Martini, A.A. Guda, S.A. Guda, A.L. Bugaev, O.V. Safonova, A.V. Soldatov  Machine Learning Powered by Principal Component Descriptors as the Key for Sorted Structural Fit of XANES // Phys. Chem. Chem. Phys., 2021](https://doi.org/10.1039/D1CP01794B)

[A. Martini, A. L. Bugaev, S. A. Guda, A. A. Guda, E. Priola, E. Borfecchia, S. Smolders, K. Janssens, D. De Vos, and A. V. Soldatov
Revisiting the Extended X-ray Absorption Fine Structure Fitting Procedure through a Machine Learning-Based Approach // The Journal of Physical Chemistry A Article ASAP](https://doi.org/10.1021/acs.jpca.1c03746)

[Guda A.A., Guda S.A., Martini A., Kravtsova A.N., Algasov A., Bugaev A., Kubrin S.P., Guda L.V., Šot P., van Bokhoven J.A., Copéret C., Soldatov A.V. Understanding X-ray absorption spectra by means of descriptors and machine learning algorithms (2021) npj Computational Materials, 7 (1), art. no. 203](https://doi.org/10.1038/s41524-021-00664-9)

[Kozyr E.G., Bugaev A.L., Guda S.A., Guda A.A., Lomachenko K.A., Janssens K., Smolders S., De Vos D., Soldatov A.V. Speciation of Ru Molecular Complexes in a Homogeneous Catalytic System: Fingerprint XANES Analysis Guided by Machine Learning (2021) Journal of Physical Chemistry C, 125 (50), pp. 27844 - 27852](https://doi.org/10.1021/acs.jpcc.1c09082)

[A. Martini, A.A. Guda, S.A. Guda, A.L. Bugaev, O.V. Safonova, A.V. Soldatov "Machine Learning Powered by Principal Component Descriptors as the Key for Sorted Structural Fit of XANES" // Phys. Chem. Chem. Phys., 2021 DOI: 10.1039/D1CP01794B](https://doi.org/10.1039/D1CP01794B)

[A. Martini, A. A. Guda, S. A. Guda, A. Dulina, F. Tavani, P. D’Angelo, E. Borfecchia, and A. V. Soldatov. Estimating a Set of Pure XANES Spectra from Multicomponent Chemical Mixtures Using a Transformation Matrix-Based Approach //  In: Di Cicco A., Giuli G., Trapananti A. (eds) Synchrotron Radiation Science and Applications. Springer Proceedings in Physics, vol 220. Springer, Cham. DOI: 10.1007/978-3-030-72005-6_6](https://doi.org/10.1007/978-3-030-72005-6_6)

[A. Martini, S. A. Guda, A. A. Guda, G. Smolentsev, A. Algasov, O. Usoltsev, M. A. Soldatov, A. Bugaev, Yu. Rusalev, C. Lamberti, A. V. Soldatov "PyFitit: the software for quantitative analysis of XANES spectra using machine-learning algorithms" Computer Physics Communications. 2019. DOI: 10.1016/j.cpc.2019.107064](https://doi.org/10.1016/j.cpc.2019.107064)

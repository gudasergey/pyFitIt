![Logo](http://www.nano.sfedu.ru/upload/medialibrary/37a/Logo_2.png)
# pyFitIt
Python implementation of FitIt software to fit X-ray absorption near edge structure (XANES). The python version is extended with additional features: machine learning, automatic component analysis, joint convolution fitting and others.

## Features
- Uses ipywidgets to construct the portable GUI
- Calculates XANES by [FDMNES](http://neel.cnrs.fr/spip.php?rubrique1007&lang=en) or [FEFF](http://monalisa.phys.washington.edu/)
- Interpolates XANES in order to speedup fitting. Support different types of interpolation point generation: grid, random, [IHS](http://people.sc.fsu.edu/~jburkardt/cpp_src/ihs/ihs.html) and various interpolation methods including machine learning algorithms
- Using multidimensional interpolation approximation you can vary parameters by sliders and see immediately theoretical spectrum, which corresponds to this geometry. Fitting can be performed on the basis of visual comparison with experiment or using automatic procedure and quantitative criteria.
- Supports direct prediction of geometry parameters by machine learning
- Includes automatic and semi-automatic component analysis

## Installation
Download the repository and run a Jupyter notebook from example folder. You need Jupyter, Pandas, Scikit-learn and py-design to be installed. The best way is to use the [Anaconda](https://www.anaconda.com/download/) distribution

## Usage
See examples folder.

### This project is developing thanks to
- Grigory Smolentsev
- Sergey Guda
- Alexandr Soldatov
- Carlo Lamberti
- Alexander Guda
- Oleg Usoltsev
- Yury Rusalev
- Andrea Martini

### If you like the software acknowledge it using the references below:
- G. Smolentsev, A.V. Soldatov "Quantitative local structure refinement from XANES: multidimensional interpolation approach" Journal of Synchrotron Radiation 2006, 13, 19-29.
- G. Smolentsev, A.V. Soldatov "FitIt: new software to extract structural information on the basis of XANES fitting" Comp. Matter. Science V. 39, N 3, 2007, 569-574.

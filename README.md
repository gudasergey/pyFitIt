![Logo](http://www.nano.sfedu.ru/upload/medialibrary/37a/Logo_2.png)
# pyFitIt
Python implementation of FitIt software to fit X-ray absorption near edge structure (XANES). The python version is extended with additional features: machine learning, automatic component analysis, joint convolution fitting and others.

[PyFitIt website](http://hpc.nano.sfedu.ru/pyfitit/)

## Features
- Uses ipywidgets to construct the portable GUI
- Calculates XANES by [FDMNES](http://neel.cnrs.fr/spip.php?rubrique1007&lang=en) or [FEFF](http://monalisa.phys.washington.edu/)
- Interpolates XANES in order to speedup fitting. Support different types of interpolation point generation: grid, random, [IHS](http://people.sc.fsu.edu/~jburkardt/cpp_src/ihs/ihs.html) and various interpolation methods including machine learning algorithms
- Using multidimensional interpolation approximation you can vary parameters by sliders and see immediately theoretical spectrum, which corresponds to this geometry. Fitting can be performed on the basis of visual comparison with experiment or using automatic procedure and quantitative criteria.
- Supports direct prediction of geometry parameters by machine learning
- Includes automatic and semi-automatic component analysis

## Installation
pip install pyfitit

## Usage
See [examples](http://hpc.nano.sfedu.ru/pyfitit/download/examples.zip). Manual is included in the examples archive.

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
- Aram Bugaev
- Mikhail Soldatov

### If you like the software acknowledge it using the references below:
A. Martini, S. A. Guda, A. A. Guda, G. Smolentsev, A. Algasov, O. Usoltsev, M. A. Soldatov, A. Bugaev, Yu. Rusalev, C. Lamberti, A. V. Soldatov "PyFitit: the software for quantitative analysis of XANES spectra using machine-learning algorithms" Computer Physics Communications. 2019. DOI: 10.1016/j.cpc.2019.107064

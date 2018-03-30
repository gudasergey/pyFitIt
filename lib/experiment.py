# import os
# import importlib
import utils
from importlib.machinery import SourceFileLoader

def LoadExperiment(folder):
    # folder = os.path.abspath(folder)

    # ParseExperiment = importlib.import_module(folder.replace('/','.')+'.ParseExperiment')
    ParseExperiment = SourceFileLoader(utils.randomString(10), folder+'/ParseExperiment.py').load_module()
    return ParseExperiment.parse()

class Experiment:
    def __init__(self, name, xanes, fit_intervals, geometryParamRanges):
        self.name = name
        self.xanes = xanes
        self.fit_intervals = fit_intervals
        self.moleculaConstructor = None # функция конструирует класс Molecula по заданным геометрическим параметрам (задаются структурой определенной в minimize)
        self.defaultParams = None
        self.smoothDefaultParams = None # класс, описанный в smoothLib с операцией [] тип размазки -> параметры
        self.geometryParamRanges = geometryParamRanges

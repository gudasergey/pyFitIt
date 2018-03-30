import importlib

def LoadExperiment(folder):
    ParseExperiment = importlib.import_module(folder.replace('/','.')+'.ParseExperiment')
    return ParseExperiment.parse()

class Experiment:
    def __init__(self, name, xanes, fit_intervals):
        self.name = name
        self.xanes = xanes
        self.fit_intervals = fit_intervals
        self.moleculaConstructor = None # функция конструирует класс Molecula по заданным геометрическим параметрам (задаются структурой определенной в minimize)
        self.defaultParams = None
        self.smoothDefaultParams = None # класс, описанный в smoothLib с операцией [] тип размазки -> параметры

import types
from copy import copy

from ipywidgets import widgets


class ControlsBuilder:
    controls = {}
    buildResult = []
    controlsStack = []

    def addSlider(self, name, min, max, step, value, type='f'):
        if type == 'f':
            s = widgets.FloatSlider(description=name, min=min, max=max, step=step, value=value,
                                orientation='horizontal', continuous_update=False)
        elif type == 'i':
            s = widgets.IntSlider(description=name, min=min, max=max, step=step, value=value,
                              orientation='horizontal', continuous_update=False)
        else:
            raise Exception('Unknown type')
        self.pushControl(name, s)

    def addFloatText(self, name, value, disabled=False):
        self.pushControl(name,
            widgets.FloatText(
                value=value,
                description=name,
                disabled=disabled
            ))

    def addValidMark(self, name, value):
        self.pushControl(name,
            widgets.Valid(
                value=value,
                description=name,
            ))

    def beginBox(self):
        self.controlsStack.append([])

    def endBox(self, type):
        controls = self.controlsStack.pop()
        if type == 'h': self.pushControl('', widgets.HBox(controls))
        else: self.pushControl('', widgets.VBox(controls))

    def pushControl(self, name, control):
        self.controls[name] = control
        if len(self.controlsStack) == 0:
            self.buildResult.append(control)
        else:
            self.controlsStack[-1].append(control)


class ControlsManager:
    context = types.SimpleNamespace()

    onControlChangedDelegate = None
    updatingModel = False

    def setup(self, controls, ui):
        self.context.controls = controls
        self.bindControlsEvents()
        self.drawControls(ui)

    def updateLoop(self, name, oldValue, newValue):
        if self.updatingModel: return
        self.updatingModel = True
        self.onControlChanged(name, oldValue, newValue)  # checks changes before populating them to the model
        self.updatingModel = False

    def onControlChanged(self, name, oldValue, newValue):
        if self.onControlChangedDelegate is not None:
            self.onControlChangedDelegate(self.context, name, oldValue, newValue)

    def drawControls(self, controls):
        from IPython.core.display import display
        for c in controls:
            display(c)

    def bindControlsEvents(self):
        for name, c in self.context.controls.items():
            c.observe(lambda change, n=name: self.updateLoop(n, change['old'], change['new']), names='value')


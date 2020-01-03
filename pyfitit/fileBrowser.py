import os
from . import utils
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript

class FileBrowser(object):
    def __init__(self, funcName):
        self.path = os.getcwd()
        self._update_files()
        self._chosenFileName = None
        self.funcName = funcName

    @property
    def chosenFileName(self):
        assert self._chosenFileName is not None, "File was not chosen"
        return self._chosenFileName

    def _update_files(self):
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            content = os.listdir(self.path)
            content.sort()
            for f in content:
                ff = self.path + "/" + f
                if os.path.isdir(ff):
                    self.dirs.append(f)
                else:
                    self.files.append(f)

    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box

    def _update(self, box):
        clear_output()
        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = os.path.join(self.path, b.description)
            self._update_files()
            self._update(box)

        buttons = []
        if self.files or self.dirs:
            button = widgets.Button(description='..')
            button.add_class('folder')
            button.add_class('parentFolder')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.dirs:
            button = widgets.Button(description=f)
            button.add_class('folder')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.files:
            button = widgets.Button(description=f)
            button.add_class('file')
            button.on_click(on_click)
            buttons.append(button)
        if len(buttons) == 0:
            buttons.append(widgets.HTML("Replace "+self.funcName+"() by the following expression to save chosen path:<br>"+self.funcName+"('"+self.path+"',...)"))
        box.children = tuple([widgets.HTML("<h2>%s</h2>" % (self.path,))] + buttons)
        box.add_class('fileBrowser')
        display(box)
        if len(buttons) == 0: self._chosenFileName = self.path

def openFile(funcName, *p):
    if len(p)>0 :
        display(widgets.HTML("Delete path argument to choose file interactively: "+funcName+'()'))
        return type('obj', (object,), {'chosenFileName' : p[0]})
    assert utils.isJupyterNotebook()
    f = FileBrowser(funcName)
    f.widget()
    return f

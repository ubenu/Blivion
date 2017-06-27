# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:20:14 2016

@author: Maria Schilstra
Code extracted artly from Sando Tosi (2009): 
Matplotlib for Python Developers - Packt Publishing Ltd, Birminham

"""
import sys
from PyQt5 import QtGui as gui
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT)
from matplotlib.widgets import SpanSelector

from OctetData import OctetData

class MplCanvas(FigureCanvas):
    """ 
    Class representing the FigureCanvas widget to be embedded in the GUI
    """
    curve_colours = ['blue',
                     'green',
                     'red',
                     'yellow',
                     'cyan',
                     'magenta',
                     'purple',
                     'brown',
                     'white',
                     'black'
                    ] 

    def __init__(self, parent):
        self.fig = Figure()
        
        self.gs = gridspec.GridSpec(5, 2) 
        self.gs.update(left=0.05, right=0.95, wspace=0.3, hspace=0.45)
        self.data_plot = self.fig.add_subplot(self.gs[1:,0])
        self.data_res_plot = self.fig.add_subplot(self.gs[0,0], sharex=self.data_plot)
        self.load_plot = self.fig.add_subplot(self.gs[:2,1])
        self.fsat_plot = self.fig.add_subplot(self.gs[3:,1])
        self.fsat_res_plot = self.fig.add_subplot(self.gs[2,1], sharex=self.fsat_plot)
        self.gs.tight_layout(self.fig)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, 
                        widgets.QSizePolicy.Expanding, widgets.QSizePolicy.Expanding)
                        
        self.set_fig_annotations()
        FigureCanvas.updateGeometry(self)  

    def set_fig_annotations(self):
        self.data_plot.set_xlabel("Time (s)")
        self.data_plot.set_ylabel("Response (nm)")
        self.data_res_plot.set_ylabel("Residuals")
        self.load_plot.set_xlabel("Trace")
        self.load_plot.set_ylabel("Response")
        self.fsat_plot.set_xlabel("Sugar loading")
        self.fsat_plot.set_ylabel("Fractional saturation")
        self.fsat_res_plot.set_ylabel("Residuals")
        self.data_res_plot.locator_params(axis='y',nbins=4)        
        self.fsat_res_plot.locator_params(axis='y',nbins=4)        
    
        
    def drawOriginalData(self, x, y, boundaries = None):
        self.data_plot.cla()
        self.data_plot.set_xlabel("Time (s)")
        self.data_plot.set_ylabel("Value")
        self.data_res_plot.set_ylabel("Residuals")
        self.data_res_plot.cla()
        dp = self.data_plot.plot(x, y)
        for i in range(len(dp)):
            dp[i].set_color(self.curve_colours[i])
        self.set_fig_annotations()
        if not boundaries is None:
            self.boundary_markers = []
            for pos in boundaries:
                self.boundary_markers.append(self.createVLine(pos=pos))
        self.fig.canvas.draw()
        
    def drawFittedData(self, x, y):
        self.data_plot.plot(x, y, color='k',linestyle='--')
        self.set_fig_annotations()
        self.fig.canvas.draw()

    def drawResiduals(self, x, y):
        rp = self.data_res_plot.plot(x, y)
        for i in range(len(rp)):
            rp[i].set_color(self.curve_colours[i])
#        self.data_res_plot.locator_params(axis='y',nbins=4)
        self.set_fig_annotations()
        self.fig.canvas.draw()
        
    def drawFittedResults(self, x_obs, y_obs, x_fit, y_fit, params):
        self.fsat_plot.cla()
        y_max, x_half, h = params
        y_obs_norm, y_fit_norm = y_obs/y_max, y_fit/y_max
        self.fsat_plot.plot(x_fit, y_fit_norm, color = 'k')
        self.fsat_plot.plot(x_obs, y_obs_norm, 
                              linestyle='None', marker='o', 
                                markerfacecolor='r', markersize=12)
        txtparams = 'Max: {:.1f} \nHalf sat: {:.2f} \nHill coeff: {:.1f}'.format(y_max, x_half, h)
        self.fsat_plot.text(x_half, 0.1, txtparams, fontsize=12)        
        self.fig.canvas.draw()
        
        
    def createVLine(self, pos=0.0, color='k', width=1.0):
        vLine = self.data_plot.axvline(pos)
        return vLine
        
    def moveVLine(self, vLine, pos):
        if not vLine is None:
            vLine.set_xdata(pos)
            self.fig.canvas.draw()
        
    def removeVLine(self, vLine):
        if not vLine is None:
            vLine.remove()
            del vLine
                
    def clearFigure(self):
        self.data_plot.cla()
        self.data_res_plot.cla()
        self.fsat_plot.cla()
        self.set_fig_annotations()
        self.fig.canvas.draw()
       
class NavigationToolbar(NavigationToolbar2QT):
                        
    def __init__(self, canvas_, parent_):
        self.toolitems = (
            ('Home', 'Reset original view', 'mjs-home', 'home'),
            (None, None, None, None), # this is a separator
            ('Zoom', 'Zoom in to rectangle', 'mjs-zoom-to-rect', 'zoom'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'mjs-pan', 'pan'),
            (None, None, None, None),
            ('Save', 'Save plot (graphics format)', 'mjs-save', 'save_figure'),
            )
        NavigationToolbar2QT.__init__(self,canvas_,parent_)  
        
    def _init_toolbar(self):
        # Reimplementation to 1) get rid of edit_curves, and 
        # 2) get control over icons
    
        self.basedir = ".\\Resources\\"
                        
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self._icon(image_file + '.png'),
                                         text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
                    
        figureoptions = None # Added to get rid of curve editor
        if figureoptions is not None:
            a = self.addAction(self._icon("qt4_editor_options.png"),
                               'Customize', self.edit_parameters)
            a.setToolTip('Edit curves line and axes parameters')

        self.buttons = {}

        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = widgets.QLabel("", self)
            self.locLabel.setAlignment(
                    qt.Qt.AlignRight | qt.Qt.AlignTop)
            self.locLabel.setSizePolicy(
                widgets.QSizePolicy(widgets.QSizePolicy.Expanding,
                                  widgets.QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        # reference holder for subplots_adjust window
        self.adj_window = None
        
    def toggle_pan_zoom(self):
        if self._active == "PAN":
            self.pan()
        elif self._active == "ZOOM":
            self.zoom()

        
class ApplicationWindow(widgets.QMainWindow):
    """
    Class representing the main window of the application
    """
    def __init__(self):
        widgets.QMainWindow.__init__(self)
        self.setWindowTitle("Blimp")
        self.setWindowIcon(gui.QIcon(".\\Resources\\Blimp.png"))
        self.main_widget = widgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_widget.setFocus()
        self.canvas = MplCanvas(self.main_widget)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.main_widget)
        vb_layout = widgets.QVBoxLayout(self.main_widget)
        vb_layout.addWidget(self.canvas)
        vb_layout.addWidget(self.plot_toolbar)

        self.status = self.statusBar()
#        status.showMessage("Ready")

        self.span = SpanSelector(self.canvas.data_plot, self.onSelectSpan, 
        'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
        self.span.active = False

        
        # Create actions for opening and closing data files
        actionOpen = self._createAction("&Open data", self.on_open, icon="mjs-open-data", tip="Open data from cvs file")
        actionClose = self._createAction("&Close data", self.on_close, icon="mjs-close-data", tip="Close data")
        actionSave = self._createAction("&Save results", self.on_save, icon="mjs-save-data", tip="Save results to cvs file")
        actionQuit = self._createAction("&Quit", self.close, shortcut="Ctrl+Q", tip="Quit the application")
        actionAnalyze = self._createAction("&Analyze", self.startAnalysis, checkable=True, icon="computer-analysis", tip="Analyze the trace_ids")        
                
        self.analysisToolBar = self.addToolBar("Analysis")
        self.analysisToolBar.addAction(actionOpen)
        self.analysisToolBar.addAction(actionSave)
        self.analysisToolBar.addAction(actionClose)
        self.analysisToolBar.addSeparator()
        self.analysisToolBar.addAction(actionAnalyze)

        self.analysisToolBar.addSeparator()

        self.phase_combo = widgets.QComboBox()
        self.phase_combo.setSizeAdjustPolicy(widgets.QComboBox.AdjustToContents)
        self.phase_combo.currentIndexChanged.connect(self.onphasecombo)
#        self.connect(self.phase_combo, qt.SIGNAL('currentIndexChanged(int)'), self.onphasecombo)
        self.analysisToolBar.addWidget(self.phase_combo)

        self.phase_spin = widgets.QDoubleSpinBox()
        self.phase_spin.valueChanged.connect(self.onphasespin)
#        self.connect(self.phase_spin, qt.SIGNAL('valueChanged(double)'), self.onphasespin)
        self.analysisToolBar.addWidget(self.phase_spin)

        self.analysisToolBar.addSeparator()

        self.func_combo = widgets.QComboBox()
        self.func_combo.setSizeAdjustPolicy(widgets.QComboBox.AdjustToContents)
        self.func_combo.currentIndexChanged.connect(self.onfunccombo)
        self.analysisToolBar.addWidget(self.func_combo)
     
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(actionOpen)
        fileMenu.addAction(actionSave)
        fileMenu.addAction(actionClose)
        fileMenu.addSeparator()
        fileMenu.addAction(actionQuit)
                        
        self.octet_data = OctetData()
        
        # Status flags
        self._dataOpen = False
        self._analyzing = False
                
    def on_open(self):
        file_path = widgets.QFileDialog.getOpenFileName(self, "Open Data File", "", 
        "CSV data files (*.csv)")[0]
        if file_path:
            if self._dataOpen:
                self.on_close()
            self.octet_data.importData(file_path)
            self._dataOpen = True
            x = self.octet_data.getIndependent()
            y = self.octet_data.getDependent()
            bounds = self.octet_data.phase_lbounds
            self.canvas.drawOriginalData(x, y, bounds)
            self.phase_spin.setRange(x.min(), x.max())
            self.phase_combo.addItems(self.octet_data.phase_names) 
            self.func_combo.addItems(list(self.octet_data.fit_funcs.values()))
            self.func_combo.setCurrentIndex(2)
            self._analyzing = False
            
    def on_save(self):
        pass
                    
    def startAnalysis(self):
        if self._analyzing == False:
            self._analyzing = True
            self.span.active = True
            self.plot_toolbar.toggle_pan_zoom()
        else:
            self._analyzing = False
            self.span.active = False
                
        
    def on_close(self):
#        self.legendList.clear()
        self.canvas.clearFigure()
        self._dataOpen = False
        self._analyzing = False
        self.phase_combo.clear() 
        self.func_combo.clear()
               
    def onSelectSpan(self, xmin, xmax):
        if self._dataOpen and self._analyzing:
            self.octet_data.analyzeSegment(xmin,xmax)
            x = self.octet_data.getIndependent()
            y = self.octet_data.getDependent()
            bounds = self.octet_data.phase_lbounds
            self.canvas.drawOriginalData(x, y, bounds)
            for i in range(len(self.octet_data.fitted)):
                if not self.octet_data.fitted[i] is None:
                    x=self.octet_data.fitted[i]['Time']
                    y=self.octet_data.fitted[i][self.octet_data.trace_ids]
                    self.canvas.drawFittedData(x, y)
                if not self.octet_data.residuals[i] is None:
                    x=self.octet_data.residuals[i]['Time']
                    y=self.octet_data.residuals[i][self.octet_data.trace_ids]
                    self.canvas.drawResiduals(x, y)
            if self.octet_data.area_analyzed.sum() == 3:
                params = self.octet_data.analyze_results()
                obs = self.octet_data.results[['Sugar loading', 'Amplitude']]
                obs = obs.loc[self.octet_data.trace_ids]
                fit = self.octet_data.create_fitted_curve(self.octet_data.getFunc(4), 
                                                          params, 0.0, 0.6, 100)
                self.canvas.drawFittedResults(obs['Sugar loading'], obs['Amplitude'], 
                                              fit['x'], fit['y'], params)
                
                
    def onphasespin(self, value):
        phase = self.phase_combo.currentIndex()
        vline = self.canvas.boundary_markers[phase]
        self.canvas.moveVLine(vline, value)
        
    def onphasecombo(self, value):
        bound = self.octet_data.phase_lbounds[value]
        self.phase_spin.setValue(bound)
                    
    def onfunccombo(self, value):
        self.octet_data.fit_func = value
        self.octet_data.results = None

                    
    def _createAction(self, text, slot=None, shortcut=None, icon=None, tip=None, 
                     checkable=False):
        action = widgets.QAction(text, self)
        if icon is not None:
            action.setIcon(gui.QIcon(".\\Resources\\%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action
        
    
        
# Make sure that this is the __main__ (ie startup) script     
if __name__ == '__main__':
    # Create the Qt application
    app = widgets.QApplication(sys.argv)
    # Create and show the mpl Widget
    aw = ApplicationWindow()
    aw.show()
    # Execute the main event loop (app.exec_()) and exit (eventially) with the usual python exit code
    sys.exit(app.exec_())

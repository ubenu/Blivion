# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:11:32 2016

@author: Maria Schilstra
"""

#from PyQt5.uic import loadUiType

from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets

from matplotlib.widgets import SpanSelector
from blivion_mpl import MplCanvas, NavigationToolbar
from blivion_data import BlivionData

# Original:
#Ui_MainWindow, QMainWindow = loadUiType('blivion.ui')

# To avoid using .ui file (from QtDesigner) and loadUIType, 
# created a python-version of the .ui file using pyuic5 from command line
# Here: pyuic5 blivion.ui -o blivion_ui.py
# Then import .py package, as below.
# (QMainWindow is a QtWidget; UI_MainWindow is generated by the converted .ui)

import blivion_ui as ui

class Main(widgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)

        self.canvas = MplCanvas(self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.plot_toolbar = NavigationToolbar(self.canvas, self.mpl_window)
        self.mpl_layout.addWidget(self.canvas)
        self.mpl_layout.addWidget(self.plot_toolbar)
        
        self.span = SpanSelector(self.canvas.data_plot, self.on_select_span, 
        'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
        
        self.action_open.triggered.connect(self.on_open)
        self.action_save.triggered.connect(self.on_save)
        self.action_close.triggered.connect(self.on_close)
        self.action_quit.triggered.connect(self.close)
        
        self.action_get_base.triggered.connect(self.on_get_base)
        self.action_get_loads.triggered.connect(self.on_get_loads)
        self.action_get_association.triggered.connect(self.on_get_association)

#        self.action_association_model.triggered.connect(self.on_association_model)
#        self.action_saturation_model.triggered.connect(self.on_saturation_model)
#        self.action_phase_boundaries.triggered.connect(self.on_phase_boundaries)
#        self.action_reduce_n.triggered.connect(self.on_reduce_n)
        
        self.blivion_data = BlivionData()
        self.file_name = ""
        self.file_path = ""

        self.span.set_active(False)

        self._data_open = False
        self._getting_base = False
        self._getting_loads = False
        self._getting_association = False
        
        self.action_open.setEnabled(True)
        self.action_save.setEnabled(False)
        self.action_close.setEnabled(False)
        self.action_quit.setEnabled(True)
        self.action_get_base.setEnabled(False)
        self.action_get_loads.setEnabled(False)
        self.action_get_association.setEnabled(False)
                
    def on_open(self):
        file_path = widgets.QFileDialog.getOpenFileName(self, 
        "Open Data File", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if file_path:
            self.file_path = file_path
            info = qt.QFileInfo(file_path)
            self.blivion_data.file_name = info.fileName()
            if self._data_open:
                self.on_close()
            self.blivion_data.import_data(file_path)
            x = self.blivion_data.get_data_x()
            y = self.blivion_data.get_data_y()
            self.action_get_base.setEnabled(True)
            self.canvas.draw_data(x, y)
            self._data_open = True
            self._getting_base = False
            self._getting_loads = False
            self._getting_association = False
            
            self.action_open.setEnabled(True)
            self.action_save.setEnabled(False)
            self.action_close.setEnabled(True)
            self.action_quit.setEnabled(True)
            self.action_get_base.setEnabled(True)
            self.action_get_loads.setEnabled(False)
            self.action_get_association.setEnabled(False)
            
    def on_save(self):
        file_path = widgets.QFileDialog.getSaveFileName(self, 
        "Save Results File", "", "CSV data files (*.csv);;All files (*.*)")[0]
        if file_path:
            self.blivion_data.export_results(file_path)
        
    def on_close(self):
        self.blivion_data = BlivionData()
        self.canvas.clear_figure()
        self._data_open = False
        self._getting_base = False
        self._getting_loads = False
        self._getting_association = False
        
        self.action_open.setEnabled(True)
        self.action_save.setEnabled(False)
        self.action_close.setEnabled(False)
        self.action_quit.setEnabled(True)
        self.action_get_base.setEnabled(False)
        self.action_get_loads.setEnabled(False)
        self.action_get_association.setEnabled(False)
        
    def on_get_base(self):
        if self.action_get_base.isChecked():
            if self.action_get_loads.isEnabled() and self.action_get_loads.isChecked():
                self.action_get_loads.setChecked(False)
            if self.action_get_association.isEnabled() and self.action_get_association.isChecked():
                self.action_get_association.setChecked(False)            
            self.plot_toolbar.switch_off_pan_zoom()
            self._getting_base = True
            self._getting_loads = False
            self._getting_association = False
            self.span.set_active(True)   
        else:
            self._getting_base = False
            self._getting_loads = False
            self._getting_association = False
            self.span.set_active(False)   
            
        
    def on_get_loads(self):
        if self.action_get_loads.isChecked():
            if self.action_get_base.isEnabled() and self.action_get_base.isChecked():
                self.action_get_base.setChecked(False)
            if self.action_get_association.isEnabled() and self.action_get_association.isChecked():
                self.action_get_association.setChecked(False)            
            self.plot_toolbar.switch_off_pan_zoom()
            self._getting_base = False
            self._getting_loads = True
            self._getting_association = False
            self.span.set_active(True)
        else:
            self._getting_base = False
            self._getting_loads = False
            self._getting_association = False
            self.span.set_active(False)
            
    def on_get_association(self):
        if self.action_get_association.isChecked():
            if self.action_get_base.isEnabled() and self.action_get_base.isChecked():
                self.action_get_base.setChecked(False)
            if self.action_get_loads.isEnabled() and self.action_get_loads.isChecked():
                self.action_get_loads.setChecked(False)
            self.plot_toolbar.switch_off_pan_zoom()
            self._getting_base = False
            self._getting_loads = False
            self._getting_association = True
            self.span.set_active(True)    
        else:
            self._getting_base = False
            self._getting_loads = False
            self._getting_association = False
            self.span.set_active(False)    
        
    def on_association_model(self):
        t = "Apologies"
        m = "Association model not yet implemented"
        mb = widgets.QMessageBox()
        mb.setText(m)
        mb.setWindowTitle(t)
        mb.setIcon(widgets.QMessageBox.Information)
        mb.exec_()
        
    def on_saturation_model(self):
        t = "Apologies"
        m = "Saturation model not yet implemented"
        mb = widgets.QMessageBox()
        mb.setText(m)
        mb.setWindowTitle(t)
        mb.setIcon(widgets.QMessageBox.Information)
        mb.exec_()
        
    def on_phase_boundaries(self):
        t = "Apologies"
        m = "Phase boundary maniputation not yet implemented"
        mb = widgets.QMessageBox()
        mb.setText(m)
        mb.setWindowTitle(t)
        mb.setIcon(widgets.QMessageBox.Information)
        mb.exec_()
        
    def on_reduce_n(self):
        t = "Apologies"
        m = "Reduction of number of points not yet implemented"
        mb = widgets.QMessageBox()
        mb.setText(m)
        mb.setWindowTitle(t)
        mb.setIcon(widgets.QMessageBox.Information)
        mb.exec_()
        
    def on_select_span(self, xmin, xmax):
        self.span.set_active(False)
        if self._getting_base:
            self.blivion_data.set_baseline_measurements(xmin, xmax)
            self._draw_results()
            self._draw_analysis()
            self._write_results()
            self.action_get_loads.setEnabled(True)
            self.action_get_base.setChecked(False)
            self._getting_base = False
        if self._getting_loads:
            self.blivion_data.set_loads_measurements(xmin, xmax)
            self._draw_results()
            self._write_results()
            self._draw_analysis()
            self.action_get_association.setEnabled(True)
            self.action_get_loads.setChecked(False)
            self._getting_loads = False
        if self._getting_association:
            self.blivion_data.set_association_measurements(xmin, xmax)
            self._draw_results()
            self._write_results()
            self._draw_analysis()
            self.action_save.setEnabled(True)
            self.action_get_association.setChecked(False)
            self._getting_association = False
            
    def _draw_results(self):
        x = self.blivion_data.get_data_x()
        y = self.blivion_data.get_data_y()
        self.canvas.draw_data(x, y)
        for phase in self.blivion_data.fitted:
            if not self.blivion_data.fitted[phase] is None:
                x=self.blivion_data.fitted[phase]['time']
                y=self.blivion_data.fitted[phase][self.blivion_data.trace_ids]
                self.canvas.draw_fitted_data(x, y) 
        for phase in self.blivion_data.fitted:
            if not self.blivion_data.residuals[phase] is None:
                x=self.blivion_data.residuals[phase]['time']
                y=self.blivion_data.residuals[phase][self.blivion_data.trace_ids]
                self.canvas.draw_residuals(x, y) 
                                
    def _draw_analysis(self):
        if self.blivion_data.results_acquired['baseline']:
            if self.blivion_data.results_acquired['loaded']:
                load = self.blivion_data.results['Sugar loading']
                self.canvas.draw_sugar_loading(load)
                if self.blivion_data.results_acquired['association']:
                    self.blivion_data.set_fractional_saturation_results()
                    if self.blivion_data.results_acquired['fractional saturation']:
                        params = self.blivion_data.fractional_saturation_params
                        mask = self.blivion_data.results['success'] == 1.0
                        obs = self.blivion_data.results[['Sugar loading', 'Amplitude (obs)',
                                                         'Amplitude (calc)']][mask]
                        fit = self.blivion_data.get_fractional_saturation_curve()
                        res = obs['Amplitude (obs)'] - obs['Amplitude (calc)']
                        self.canvas.draw_fractional_saturation(obs['Sugar loading'], obs['Amplitude (obs)'],
                                                               res, fit['x'], fit['y'], params)

    def _write_results(self):
        r = self.blivion_data.results
        tbr = self.tblResults
        tbr.setColumnCount(len(r.columns))
        tbr.setRowCount(len(r.index))
        tbr.setVerticalHeaderLabels(r.index)
        tbr.setHorizontalHeaderLabels(r.columns)
        for i in range(len(r.index)):
            for j in range(len(r.columns)):
                tbr.setItem(i,j,widgets.QTableWidgetItem(str(r.iat[i, j])))

        if self.blivion_data.results_acquired['fractional saturation']:
            p = self.blivion_data.get_fractional_saturation_params_dataframe()
            tbp = self.tblFitParams
            tbp.setColumnCount(len(p.columns))
            tbp.setRowCount(len(p.index))
            tbp.setVerticalHeaderLabels(p.index)
            tbp.setHorizontalHeaderLabels(p.columns)
            for i in range(len(p.index)):
                for j in range(len(p.columns)):
                    tbp.setItem(i,j,widgets.QTableWidgetItem(str(p.iat[i, j])))
            
            f = self.blivion_data.get_fractional_saturation_curve()
            tbf = self.tblFittedCurve
            tbf.setColumnCount(len(f.columns))
            tbf.setRowCount(len(f.index))
            tbf.setHorizontalHeaderLabels(f.columns)
            for i in range(len(f.index)):
                for j in range(len(f.columns)):
                    tbf.setItem(i,j,widgets.QTableWidgetItem(str(f.iat[i, j])))
            
        
        
# Standard main loop code
if __name__ == '__main__':
    import sys
#    sys.tracbacklimit = 10
    app = widgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
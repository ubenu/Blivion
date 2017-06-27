# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:20:14 2016

@author: Maria Schilstra

"""
from PyQt4 import QtGui as gui
from PyQt4 import QtCore as qt

from PyQt4.uic import loadUiType
 
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)

Ui_MainWindow, QMainWindow = loadUiType('octanavir.ui')    

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        
        self.connect(self.actionOpen, qt.SIGNAL("triggered()"), self.openDataFile)
        
    def addPlot(self, fig):
        self.dataCanvas = FigureCanvas(fig)
        self.layoutPlots.addWidget(self.dataCanvas)
        self.dataCanvas.draw()
        self.toolbar = NavigationToolbar(self.dataCanvas, 
                self.pltData, coordinates=True)
        self.addToolBar(self.toolbar)
        
    def openDataFile(self):
        return gui.QFileDialog.getOpenFileName(self, "Open Data File", "", 
        "CSV data files (*.csv)")

        
    
        
        
if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui as gui
    from PyQt4 import QtCore as qt
    import numpy as np
 
    fig1 = Figure()
    ax1f1 = fig1.add_subplot(111)
    ax1f1.plot(np.random.rand(20))
# 
#    fig2 = Figure()
#    ax1f2 = fig2.add_subplot(121)
#    ax1f2.plot(np.random.rand(5))
#    ax2f2 = fig2.add_subplot(122)
#    ax2f2.plot(np.random.rand(10))
# 
#    fig3 = Figure()
#    ax1f3 = fig3.add_subplot(111)
#    ax1f3.pcolormesh(np.random.rand(20,20))
 
    app = gui.QApplication(sys.argv)
    main = Main()
    
    main.addPlot(fig1)
#    fn = main.openDataFile()
#    print(fn)
#    main.addfig('One plot', fig1)
#    main.addfig('Two plots', fig2)
#    main.addfig('Pcolormesh', fig3)

    main.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


def fn_exp(x, params):
    n = params.shape[0]
    if n == 3:
        return params[0] + params[1]*np.exp(-x*params[2])
    elif n == 5:
        return params[0] + params[1]*np.exp(-x*params[2]) + params[3]*np.exp(-x*params[4])
    elif n == 7:
        return params[0] + params[1]*np.exp(-x*params[2]) + params[3]*np.exp(-x*params[4]) + params[5]*np.exp(-x*params[6])
    else:
        return x
    

def fn_2exp(x,a0,a1,a2,k1,k2):
    return a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2)
    
def fn_1exp(x,a0,a1,k1):
    return a0 + a1*np.exp(-x*k1)
    
def fn_hill(x,ymax,xh,h):
    return ymax/(np.power(xh/x,h)+1.0)
    

root = tk.Tk()
file_path = filedialog.askopenfilename()
root.withdraw()
    
# Prepare figure area
plt.figure()
pt1 = plt.subplot(2, 2, 1)
pt2 = plt.subplot(2, 2, 2)
pt3 = plt.subplot(2, 2, 3)
pt4 = plt.subplot(2, 2, 4)

# Prepare data
d_all = pd.read_csv(file_path) #('OctetTestData.csv')
cols = [0,1,3,5,7,9,11,13,15]
d_all = d_all.iloc[:,cols]
d_all = d_all[0:-1:5]
d_all.columns = ['time','A1','B1','C1','D1','E1','F1','G1','H1']

d_all.plot(x='time',ax=pt1, legend=None)

data = np.zeros((d_all.columns.size-1,5))
dout = pd.DataFrame(data)
dout.index = d_all.columns[1:]
dout.columns = ['zero','load','amp0', 'Sugar loading', 'Amplitude']
dset = np.zeros((3,))

def onselect1(xmin, xmax):
    indmin, indmax = np.searchsorted(d_all['time'],(xmin, xmax))
    d=d_all[indmin:indmax]
    t0 = d.iloc[0,0]
    x = (d['time'] - t0)
    pt2.cla()
    d.plot(x='time',ax=pt2, linestyle='--', linewidth=3.0, legend=None)
    dfit = x+t0
    ddiff = x+t0
    if t0 < 750:
        means = d.iloc[:,1:].mean()
        col = 'load'
        dset[1] = 1
        if t0 < 250:
            col = 'zero'
            dset[0] = 1
        dout[col] = means
    
    else:
        for curve in d.iloc[:,1:]:
            dset[2] = 1
            y = d[curve] 
            pest = (y.min(), (y.max()-y.min())/2, 1.0, 2.0/x.max(), 1.0)
            dout['amp0'][curve] = 0.0
            try:
                errmsg, params = "", []
                params, covar, infodict, errmsg, ier = curve_fit(fn_2exp, x, y, pest, 
                                    full_output=1, ftol=1.5e-8, xtol=1.5e-8)
                a0,a1,a2,k1,k2 = params
                yfit = fn_2exp(x,a0,a1,a2,k1,k2)
                ydiff = y - yfit
                dfit = pd.concat([dfit, yfit], axis=1)
                headings = dfit.columns.values
                headings[-1] = curve + "fit"
                ddiff = pd.concat([ddiff, ydiff], axis=1)
                headings = ddiff.columns.values
                headings[-1] = curve + "dif"
                ddiff.columns = headings
                dout['amp0'][curve] = a0
                            
            except ValueError:
                print("Value Error")
            except RuntimeError as e:  
                print("Runtime Error:" + str(e))
            except:
                print("Other error")
        
        pt1.cla()
        pt3.cla()
        d_all.plot(x='time',ax=pt1, legend=None)
        dfit.plot(x='time', ax=pt1, color='black', legend=None)
        dfit.plot(x='time', ax=pt2, color='black', legend=None)
        ddiff.plot(x='time', ax=pt3, legend=None)
        
    if dset.sum() == 3:
        dout['Sugar loading'] = dout['load'] - dout['zero']
        dout['Amplitude'] = dout['load'] - dout['amp0']
        doutsorted = dout.sort_values('Sugar loading')
        print(doutsorted)
        pt4.cla()
        xrange = (0.0,0.6)
        try:
                        
            xdata=doutsorted['Sugar loading']
            ydata=doutsorted['Amplitude']
            pest=[ydata.max(),xdata.max()/2.0,1.0]
            params, covar, infodict, errmsg, ier = curve_fit(fn_hill, xdata, ydata, pest, full_output=1)
            ymax,xhalf,h = params
    
            xplotfit = pd.Series(np.linspace(xrange[0],xrange[1],100))
            yplotfit = fn_hill(xplotfit,ymax,xhalf,h)/ymax
            ydatanorm = doutsorted['Amplitude']/ymax
            ddatanorm = doutsorted.copy()
            ddatanorm['Amplitude'] = ydatanorm
            dplotfit = pd.concat([xplotfit, yplotfit], axis=1)
            dplotfit.columns=['Sugar loading', 'Fit']
            txtparams = 'Max: {:.1f} \nHalf sat: {:.2f} \nHill coeff: {:.1f}'.format(ymax,xhalf,h)
            pt4.cla()
            ddatanorm.plot(x='Sugar loading', y='Amplitude', ax=pt4, style=['ro'], markersize=12, 
                            xlim=xrange, legend=None)            
            dplotfit.plot(x='Sugar loading',y='Fit', color='black', zorder=2, ax=pt4, legend=None)
            pt4.text(xhalf+(xrange[1]-xrange[0])/5,0.5,txtparams,fontsize=16)
    
        except ValueError:
            print("Value Error")
        except RuntimeError as e:  
            print("Runtime Error:" + str(e))
        except:
            doutsorted.plot(x='Sugar loading', y='Amplitude', ax=pt4, 
                            style=['ro'], markersize=12, 
                            xlim=xrange, legend=None)
            print("Other error")
       
            
span1 = SpanSelector(pt1, onselect1, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))


plt.show()


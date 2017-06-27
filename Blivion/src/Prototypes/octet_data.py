# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 10:31:43 2016

@author: Maria
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

pd.options.mode.chained_assignment = None  
# suppresses unnecessary warning when creating self.working_data
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
# suppresses unnecessary runtime warnings when fitting data

class OctetData():
    
    def __init__(self):
        # constants
        (self.START, self.BASELINE, self.LOADING, self.LOADED, 
         self.ASSOCIATION, self.DISSOCIATION, self.FINISHED) = range(7)
        (self.FN_LINE, self.FN_1EXP, self.FN_2EXP, self.FN_3EXP, 
         self.FN_HILL) = range(5)  
        self.phases = (self.START, self.BASELINE, self.LOADING, self.LOADED, 
                       self.ASSOCIATION, self.DISSOCIATION, self.FINISHED)
        self.phase_names = ['start', 'baseline', 'loading', 'loaded', 
        'association', 'dissociation', 'end']
        
        self.fit_funcs = (self.FN_LINE, self.FN_1EXP, self.FN_2EXP, 
                          self.FN_3EXP, self.FN_HILL)
        self.fit_func_names = ['Straight line', 'Single exponential', 
        'Double exponential', 'Triple exponential', 'Hill function']

        # templates for results
        self.phase_lbounds = np.zeros((len(self.phases)), dtype=float)
        self.residuals, self.fitted = [], []
        for i in self.phases:
            self.residuals.append(None)
            self.fitted.append(None)
        self.results = None
        self.association_params_opt = None

        # attributes
        self.raw_data = None
        self.working_data = None
        self.trace_ids = None
        self.area_measured = {'baseline': False, 'loaded': False, 'association': False }
        
        # default settings
        self.data_reduction_factor = 5
        self.fit_func = self.FN_2EXP
     
    def import_data(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self._create_working_data()
        tEnd = self.working_data['time'].iloc[-1]
        self.phase_lbounds[:] = 0.0, 10.0, 240.0, 540.0, 780.0, tEnd, tEnd
        
    def _create_working_data(self):
        selected = self.raw_data.columns[1::2]
        self.trace_ids = self.raw_data.columns[0::2]         
        self.working_data = self.raw_data[selected]
        self.working_data.columns = self.trace_ids
        self.working_data['time'] = self.raw_data.iloc[:,0]
        self.working_data = self.working_data[0:-1:self.data_reduction_factor]

    def set_baseline_measurements(self, start, stop):
        if self.results is None:
            self.results = self.get_results_template(self.trace_ids)            
        indmin, indmax = self.get_span_indices(start, stop)
        selection = self.working_data[indmin:indmax]
        sel_zeros, sel_ones = selection.copy(), selection.copy() 
        sel_zeros[self.trace_ids] = 0.0
        sel_ones[self.trace_ids] = 1.0
        t = selection['time']
        means = selection[self.trace_ids].mean()
        self.results['baseline'] = means
        self.area_measured['baseline'] = True
        self.set_measurements()
        self.residuals[self.BASELINE] = selection[self.trace_ids] - means
        self.residuals[self.BASELINE]['time'] = t
        self.fitted[self.BASELINE] = sel_ones.copy() * means 
        self.fitted[self.BASELINE]['time'] = t                
            
    def set_loads_measurements(self, start, stop):
        if self.results is None:
            self.results = self.get_results_template(self.trace_ids)            
        indmin, indmax = self.get_span_indices(start, stop)
        selection = self.working_data[indmin:indmax]
        sel_zeros, sel_ones = selection.copy(), selection.copy() 
        sel_zeros[self.trace_ids] = 0.0
        sel_ones[self.trace_ids] = 1.0
        t = selection['time']
        means = selection[self.trace_ids].mean()
        self.results['loaded'] = means
        self.area_measured['loaded'] = True
        self.set_measurements()
        self.residuals[self.LOADED] = selection[self.trace_ids] - means
        self.residuals[self.LOADED]['time'] = t
        self.fitted[self.LOADED] = sel_ones.copy() * means 
        self.fitted[self.LOADED]['time'] = t                

    def set_association_measurements(self, start, stop):
        pnames = self.get_parameter_names(self.fit_func)
        indmin, indmax = self.get_span_indices(start, stop)
        selection = self.working_data[indmin:indmax]
        self.results['association'] = 0.0
        self.association_params_opt = pd.DataFrame(np.zeros((len(self.trace_ids), 
                                                             len(pnames)), dtype=float))
        self.association_params_opt.columns = pnames
        self.association_params_opt.index = self.trace_ids
        sel_zeros = selection.copy() 
        sel_zeros[self.trace_ids] = 0.0
        self.fitted[self.ASSOCIATION] = sel_zeros.copy()
        self.residuals[self.ASSOCIATION] = sel_zeros.copy()
        func = self.get_func(self.fit_func)
        t = selection['time']
        x = t - t.iloc[0]  
        for trace in self.trace_ids:
            y = selection[trace]
            p_est = self.get_parameter_estimates(self.fit_func, x, y)
            errmsg, params = "", []
            try:
                params, covar, infodict, errmsg, ier = curve_fit(func, 
                    x, y, p_est, full_output=1)
                self.results['association'][trace] = params[0]
                for i in range(len(pnames)):
                    self.association_params_opt[pnames[i]][trace] = params[i]
                p = tuple(params)
                y_fit = func(x, *p)
                y_res = y - y_fit
                self.fitted[self.phases[self.ASSOCIATION]][trace] = y_fit
                self.residuals[self.phases[self.ASSOCIATION]][trace] = y_res
            except ValueError as e:
                print("Value Error:" + str(e))
            except RuntimeError as e:  
                print("Runtime Error:" + str(e))
            except:
                print("Other error") 
                
        self.area_measured['association'] = True
        self.set_measurements()
        print(self.results)
        
    def set_measurements(self):
        sl = (self.results['loaded'] - self.results['baseline'])
        self.results['Sugar loading'][self.trace_ids] = sl
        amp = (self.results['loaded'] - self.results['association'])
        self.results['Amplitude'][self.trace_ids] = amp
        
        
    def get_fractional_saturation_params(self):
        data = pd.DataFrame()
        data['Sugar loading'] = self.results['Sugar loading']
        data['Amplitude'] = self.results['Amplitude']
        data = data.sort_values('Sugar loading')
        params = None
        try:
            x = data['Sugar loading']
            y = data['Amplitude']
            p_est = self.get_parameter_estimates(self.FN_HILL, x, y)
            func = self.get_func(self.FN_HILL)
            params, covar, infodict, errmsg, ier = curve_fit(func, x, y, 
                                                             p_est, full_output=1)
            ymax, xhalf, h = params
        except ValueError as e:
            print("Value Error:" + str(e))
        except RuntimeError as e:  
            print("Runtime Error:" + str(e))
        except:
            print("Other error")
        return params    
            
    def get_fractional_saturation_curve(self, params, x_start, x_stop, x_num):
        return self.create_fitted_curve(self.FN_HILL, params, x_start, x_stop, x_num)
            
    def get_results_template(self, trace_ids):
        header = ['baseline', 'loaded', 'association', 'Sugar loading', 'Amplitude']
        data = np.zeros((len(trace_ids), len(header)), dtype=float)
        template = pd.DataFrame(data)
        template.index = trace_ids
        template.columns = header
        return template
        
    def get_span_indices(self, start, stop):
        return np.searchsorted(self.working_data['time'],(start, stop))
        
    def create_fitted_curve(self, func_id, params, x_start, x_stop, x_num):
        data = pd.DataFrame()
        func = self.get_func(func_id) 
        p = tuple(params)
        data['x'] = np.linspace(x_start, x_stop, x_num) 
        y = func(data['x'], *p)
        data['y'] = y
        return data
        
    def setPhaseStart(self, time, phase):
        if not self.working_data is None:
            t = max(time, 0.0)
            t = min(t, self.working_data['time'].iloc[-1])
            self.phase_lbounds[phase] = t
        
    def get_parameter_estimates(self, func_id, x, y):
        if func_id in [self.FN_LINE, self.FN_1EXP, self.FN_2EXP, self.FN_3EXP]:
            n = {self.FN_LINE:2, self.FN_1EXP:3, self.FN_2EXP:5, self.FN_3EXP:7}
            a0 = y.min()
            a1 = (y.min() - y.max()) / 2.0
            k1 = 4.0 / x.max()
            a2 = a1 / 2.0
            k2 = 2.0 / x.max()
            a3 = a2 / 2.0
            k3 = 1.0 / x.max()
            estimates = (a0, a1, k1, a2, k2, a3, k3)
            return estimates[0 : n[self.fit_func]]
        elif func_id == self.FN_HILL:
            ymax, xhalf, h = y.max(), x.max(), 1.0
            return(ymax, xhalf, h)
        
    def get_parameter_names(self, func_id):
        if func_id == self.FN_LINE:
            return['a', 'b']
        elif func_id == self.FN_1EXP:
            return ['a0', 'a1', 'k1']
        elif func_id == self.FN_2EXP:
            return ['a0', 'a1', 'k1', 'a2', 'k2']
        elif func_id == self.FN_3EXP:
            return ['a0', 'a1', 'k1', 'a2', 'k2', 'a3', 'k3']
        elif func_id == self.FN_HILL:
            return ['ymax', 'xhalf', 'h']
        else:
            return None        
        
    def get_func(self, func_id):
        if func_id == self.FN_LINE:
            return self.fn_straight_line
        if func_id == self.FN_1EXP:
            return self.fn_1exp
        elif func_id == self.FN_2EXP:
            return self.fn_2exp
        elif func_id == self.FN_3EXP:
            return self.fn_3exp
        elif func_id == self.FN_HILL:
            return self.fn_hill
        else:
            return None
        
    def get_data_x(self):
        try:
            return self.working_data['time']
        except:
            print("No independent")
            
    def get_data_y(self, trace_ids=[]):
        if trace_ids == []:
            trace_ids = self.trace_ids
        try:
            return self.working_data[trace_ids]  
        except:
            print("No dependent")
            
    def fn_straight_line(self, x, *p):
        a, b = p
        return (a + b*x)
    
    def fn_1exp(self, x, *p):
        a0, a1, k1 = p
        return (a0 + a1*np.exp(-x*k1))
        
    def fn_2exp(self, x, *p):
        a0, a1, k1, a2, k2 = p
        return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2))
        
    def fn_3exp(self, x, *p):
        a0, a1, k1, a2, k2, a3, k3 = p
        return (a0 + a1*np.exp(-x*k1) + a2*np.exp(-x*k2) + a3*np.exp(-x*k3))
        
    def fn_hill(self, x, *p):
        ymax, xhalf, h = p
        return ymax / (np.power(xhalf/x, h) + 1.0)
        
 

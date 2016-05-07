import numpy as np
import pylab as pl
import math 
from sympy import *

class NN:
    def __init__(self):
        pass
    
    def set_input_layer(self, x):
        self.input_layer = np.zeros(x)
        return self.input_layer
    
    def set_output_layer(self, x):
        self.output_layer = np.zeros(x)
        self.before_output_layer = np.zeros(x)
        self.supervised_data = np.zeros(x)
        return self.output_layer, self.before_output_layer, self.supervised_data
    
    def set_hidden_layer(self, x):
        self.hidden_layer = np.zeros(x)
        self.before_hidden_layer = np.zeros(x)
        return self.hidden_layer, self.before_hidden_layer
    
    def setup(self):
        w_k = np.zeros(len(self.output_layer))
        self.w_kj = np.array([w_k for i in range(len(self.hidden_layer))])
        w_j = np.zeros(len(self.hidden_layer))
        self.w_ji = np.array([w_j for i in range(len(self.input_layer))])
        return self.w_kj, self.w_ji
        
    def initialize(self, hidden=None):
        for i in range(len(self.hidden_layer)):
            for j in range(len(self.output_layer)):
                self.w_kj[i][j] = np.random.uniform(-1.0/math.sqrt(1.0/len(self.hidden_layer)), 1.0/math.sqrt(1.0/len(self.hidden_layer)))
            
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                self.w_ji[i][j] = np.random.uniform(-1.0/math.sqrt(1.0/len(self.input_layer)), 1.0/math.sqrt(1.0/len(self.input_layer)))
     
        if hidden is None:
            u = Symbol('u')
            self.hfunction = 1/(1+exp(-u))
            self.diff_hf = diff(self.hfunction)
        else:
            self.hfunction = hidden
            self.diff_hf = diff(self.hfunction)
    
    def supervised_function(self, f, idata):
        x_1 = Symbol('x_1')
        x_2 = Symbol('x_2') 

        for i in range(len(idata)):
            self.input_layer[i] = idata[i]
    
        for i in range(len(self.supervised_data)):
            self.supervised_data[i] = f.subs([(x_1, idata[0]), (x_2, idata[1])]) 
            
    def set_hidden_error(self, j):
        u = Symbol("u")
        diff_hf = self.diff_hf 
        hidden_error = 0
        for k in range(len(self.output_layer)):
            delta_z = diff_hf.subs([(u, self.before_output_layer[k])]) 
            hidden_error += self.w_kj[j][k]*(self.supervised_data[k] - self.output_layer[k])*delta_z
        return hidden_error
        
    def calculation(self):
        u = Symbol("u")
        hfunction = self.hfunction
        diff_hf = self.diff_hf 
        
        for i in range(len(self.input_layer)):
            self.before_hidden_layer = np.matrix(self.w_ji).T*np.matrix(self.input_layer).T
            
        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i] = hfunction.subs([(u, self.before_hidden_layer[i])])
            
        for i in range(len(self.before_output_layer)):
            self.before_output_layer = np.matrix(self.w_kj).T*np.matrix(self.hidden_layer).T
                                                   
        for i in range(len(self.output_layer)):
            self.output_layer[i] = hfunction.subs([(u, self.before_output_layer[i])]) 
                
    def output_ad(self):
        u = Symbol("u")
        hfunction = self.hfunction
        diff_hf = self.diff_hf 
        
        eta = self.eta
        for j in range(len(self.hidden_layer)):
            for k in range(len(self.output_layer)):
                delta_J = self.supervised_data[k] - self.output_layer[k]
                delta_z = self.output_layer[k]*(1-self.output_layer[k])
                delta_v = self.hidden_layer[j]
                self.w_kj[j][k] += eta*delta_J*delta_z*delta_v
    
    def input_ad(self):  
        u = Symbol("u")
        hfunction = self.hfunction
        diff_hf = self.diff_hf 
        
        eta = self.eta
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                hidden_error = self.set_hidden_error(j)
                delta_y = self.hidden_layer[j]*(1-self.hidden_layer[j])
                delta_u = self.input_layer[i]
                self.w_ji[i][j] += eta*hidden_error*delta_y*delta_u
                
    def simulate(self, N, eta):
        self.eta = eta
        for i in range(N):
            self.calculation()
            self.output_ad()
            self.calculation()
            self.input_ad()
        return self.output_layer
    
    def main(self, N, f, idata, eta, i=2, h=2, o=1):
        self.set_input_layer(i)
        self.set_hidden_layer(h)
        self.set_output_layer(o)
        self.setup()
        self.initialize()
        self.supervised_function(f, idata)
        self.simulate(N, eta)
        return self.output_layer[0]
    
    def set_network(self, i=2, h=2, o=1):
        self.set_input_layer(i)
        self.set_hidden_layer(h)
        self.set_output_layer(o)
        self.setup()
        self.initialize()
    
    def main2(self, N, f, idata, eta):
        self.supervised_function(f, idata)
        self.simulate(N, eta)
        return self.output_layer[0]
    
    def realize(self, f, idata):
        self.supervised_function(f, idata)
        self.calculation()
        return self.output_layer[0]
                
class RNN:
    def __init__(self):
        pass
    
    def set_input_layer(self, x):
        self.input_layer = np.zeros(x)
        return self.input_layer
    
    def set_output_layer(self, x):
        self.output_layer = np.zeros(x)
        self.before_output_layer = np.zeros(x)
        self.supervised_data = np.zeros(x)
        return self.output_layer, self.before_output_layer, self.supervised_data
    
    def set_hidden_layer(self, x):
        self.hidden_layer = np.zeros(x)
        self.before_hidden_layer = np.zeros(x)
        return self.hidden_layer, self.before_hidden_layer
    
    def setup(self):
        w_k = np.zeros(len(self.output_layer))
        self.w_kj = np.array([w_k for i in range(len(self.hidden_layer))])
        w_j = np.zeros(len(self.hidden_layer))
        self.w_ji = np.array([w_j for i in range(len(self.input_layer))])
        return self.w_kj, self.w_ji
        
    def initialize(self, hidden=None):
        for i in range(len(self.hidden_layer)):
            for j in range(len(self.output_layer)):
                self.w_kj[i][j] = np.random.uniform(-1.0/math.sqrt(1.0/len(self.hidden_layer)), 1.0/math.sqrt(1.0/len(self.hidden_layer)))
            
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                self.w_ji[i][j] = np.random.uniform(-1.0/math.sqrt(1.0/len(self.input_layer)), 1.0/math.sqrt(1.0/len(self.input_layer)))
     
        if hidden is None:
            u = Symbol('u')
            self.hfunction = 1/(1+exp(-u))
            self.diff_hf = diff(self.hfunction)
        else:
            self.hfunction = hidden
            self.diff_hf = diff(self.hfunction)
    
    def supervised_function(self, sdata):
        for i in range(len(self.supervised_data)):
            self.supervised_data[i] = sdata[i]
            
    def set_hidden_error(self, j):
        u = Symbol("u")
        diff_hf = self.diff_hf 
        hidden_error = 0
        for k in range(len(self.output_layer)):
            delta_z = diff_hf.subs([(u, self.before_output_layer[k])]) 
            hidden_error += self.w_kj[j][k]*(self.supervised_data[k] - self.output_layer[k])*delta_z
        return hidden_error
        
    def calculation(self):
        u = Symbol("u")
        hfunction = self.hfunction
        diff_hf = self.diff_hf
        
        for i in range(len(self.input_layer)):
            self.before_hidden_layer = np.matrix(self.w_ji).T*np.matrix(self.input_layer).T
            
        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i] = hfunction.subs([(u, self.before_hidden_layer[i])])
            
        for i in range(len(self.before_output_layer)):
            self.before_output_layer = np.matrix(self.w_kj).T*np.matrix(self.hidden_layer).T
                                                   
        for i in range(len(self.output_layer)):
            self.output_layer[i] = hfunction.subs([(u, self.before_output_layer[i])]) 
                
    def output_ad(self):
        u = Symbol("u")
        hfunction = self.hfunction
        diff_hf = self.diff_hf 
        
        eta = self.eta
        for j in range(len(self.hidden_layer)):
            for k in range(len(self.output_layer)):
                delta_J = self.supervised_data[k] - self.output_layer[k]
                delta_z = self.output_layer[k]*(1-self.output_layer[k])
                delta_v = self.hidden_layer[j]
                self.w_kj[j][k] += eta*delta_J*delta_z*delta_v
    
    def input_ad(self):  
        u = Symbol("u")
        hfunction = self.hfunction
        diff_hf = self.diff_hf 
        
        eta = self.eta
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                hidden_error = self.set_hidden_error(j)
                delta_y = self.hidden_layer[j]*(1-self.hidden_layer[j])
                delta_u = self.input_layer[i]
                self.w_ji[i][j] += eta*hidden_error*delta_y*delta_u
                
    def simulate(self, idata, sdata, eta):
        self.eta = eta
        self.thidden = np.array([])
        self.toutput = np.array([])
        for i in range(len(idata)):
            self.supervised_function(sdata[i])
            for j in range(len(idata[i])):
                self.input_layer[j] = idata[-i-1][j]
            for j in range(len(self.hidden_layer)):
                self.input_layer[len(idata[-i-1])+j-1] = self.hidden_layer[j]
            self.calculation()
            self.output_ad()
            self.calculation()
            self.input_ad()
        return self.output_layer
    
    def main(self, idata, sdata, eta, N, i=2, h=2, o=1):
        self.set_input_layer(i+h)
        self.set_hidden_layer(h)
        self.set_output_layer(o)
        self.setup()
        self.initialize()
        return self.output_layer[0]
    
    def set_network(self, i=2, h=5, o=1):
        self.set_input_layer(i+h)
        self.set_hidden_layer(h)
        self.set_output_layer(o)
        self.setup()
        self.initialize()
    
    def main2(self, idata, sdata, eta):
        self.simulate(idata, sdata, eta)
        return self.output_layer[0]
    
    def realize(self, idata):
        for i in range(len(idata)):
            for j in range(len(idata[-i])):
                self.input_layer[j] = idata[i][j]
            for j in range(len(self.hidden_layer)):
                self.input_layer[len(idata[i])+j-1] = self.hidden_layer[j]
            self.calculation()
        return self.output_layer[0]
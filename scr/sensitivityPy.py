# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:43:57 2021

@author: tefav
"""

class sensitivityPy:

    def __init__(self, dss, node=None, load_mult=1, load=None):
        self.dss = dss
        self.load_mult = load_mult
        self.node = node
        self.load = load

    def voltageProfile(self, loadkw=None, gen_kw=None, gen_kv=None):
        
        if self.node != None: 
            # create generator
            self.__new_1ph_gen(gen_kv, gen_kw)
            
        if self.load != None:
            self.__modifyLoad(kw=loadkw, kvar=loadkw, load=self.load)
        
        # define the time of the day:
        self.dss.text(f"set loadmult={self.load_mult}")
        
        self.dss.text("solve")
        
        # get the sensitivity in volts 
        voltages = self.dss.circuit_all_bus_vmag()
        
        return voltages
    
    def __modifyLoad(self, kw, kvar, load):
        self.dss.text(f"edit load.{load} "
                      f"kw={kw} "
                      f"kvar={kvar}")
        
    def __new_1ph_gen(self, gen_kv, kw):
        self.dss.text(f"new generator.gen "
                      f"phases=1 "
                      f"kv={gen_kv} "
                      f"bus1={self.node} "
                      f"kw={kw} "
                      f"kva={kw} "
                      f"pf=1")


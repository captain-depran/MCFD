# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:51:02 2025

@author: Luke
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class param:
    def __init__(self,name,low,high,start,step_size):
        self.name=name
        self.low=low
        self.high=high
        self.current=start
        self.chain=[]
        self.step_lim=step_size
        self.fail_count=0
    def uniform_prob(self,x):
        n=self.high-self.low
        if x > self.high or x <= self.low:
            return 0
        elif x > self.low and x <= self.high:
            return 1/n
        
    def walk(self):
        self.proposed=self.current+np.random.normal(0,self.step_lim**2)
        
    def prior(self):
        return(self.uniform_prob(self.proposed))
    
class explore:
    
    def __init__(self,xs,ys,model_func,theta):
        self.xs=xs
        self.ys=ys
        self.model_func=model_func
        self.pass_count=0
        self.step_count=0
        self.post_last=0
        self.pass_rate=[]
        self.theta=theta
        self.post_list=[]
    
    def burn(self,tests):
        print("--- BURN IN PHASE ---")
        for n in tqdm(range(0,tests)):
            self.step()
        
        #self.pass_count=0
        #self.step_count=0
        #self.pass_rate=[]
        self.post_pass=[]
        for param in self.theta:
            param.chain=[]
            param.fail_count=0
        
        
        
    def run(self,tests,print_out):
        print("--- RUN PHASE ---")
        for n in tqdm(range (0,tests)):
            self.step()
        print("--- Done! ---")
        if print_out==True:
            for param in self.theta:
                print(param.name+": ",np.mean(param.chain))
        
    
    def l_hood(self,sigma,model_dat,true_dat):
        term_1=-1*(len(true_dat)/2)*np.log(2*np.pi*(sigma.proposed**2))
        term_2=(1/(2*(sigma.proposed**2)))*np.sum((true_dat-model_dat)**2)
        return term_1-term_2
    
    def prior(self,theta):
        value=0
        for param in theta:
            prior_val=param.prior()
            if prior_val>0:
                value+=np.log(prior_val)
            else:
                value=np.inf
                param.fail_count+=1
        return value
    
    def test(self,prop,curr):
        log_r=prop-curr
        if log_r <-100:
            r=0
        elif log_r > 0:
            r=1
        else:
            r=np.exp(log_r)
        if r >=1:
            return True
        elif r!=0 and r>=np.random.random():
            return True
        else:   
            return False
    
    def step(self):
        theta_values=[]
        for param in self.theta:
            param.walk()
            theta_values.append(param.proposed)
        model_dat=self.model_func(self.xs,*theta_values[:-1])
        prior=self.prior(self.theta)
        if prior==np.inf:
            test_pass=False
            self.post_list.append(0)
        else:
            post=self.l_hood(self.theta[-1],model_dat,self.ys)+self.prior(self.theta)
            test_pass=self.test(post,self.post_last)
            self.post_list.append(post)
        self.step_count+=1
        
        if test_pass==True:
            self.pass_count+=1
            self.post_last=post
            for param in self.theta:
                param.current=param.proposed
                param.chain.append(param.current)
        self.pass_rate.append((100)*(self.pass_count/self.step_count))
                
    
    def first_step(self):
        theta_values=[]
        for param in self.theta:
            param.walk()
            theta_values.append(param.current)
            param.chain.append(param.current)
        model_dat=self.model_func(self.xs,*theta_values[:-1])
        post=self.l_hood(self.theta[-1],model_dat,self.ys)+self.prior(self.theta)
        self.post_last=post
        
class multirun:
    def __init__(self,walkers,func,xs,ys,theta):
        self.chains=walkers
        self.explorers=[]
        self.theta=theta
        for n in range(0,walkers):
            self.explorers.append(explore(xs,ys,func,theta))
            if n!=0:
                for param in self.explorers[-1].theta:
                    scope=param.high-param.low
                    frac=scope/walkers
                    param.start=param.low+((n)*frac)
                    self.explorers[-1].first_step()
                

        
    def run(self,tests,burn_frac,trace):
        run_frac=1-burn_frac
        all_params=[]
        for explorer,n in zip(self.explorers,range(0,len(self.explorers))):
            print("Run #",n+1)
            explorer.burn(int(tests*burn_frac))
            explorer.run(int(tests*run_frac),False)
            for param in explorer.theta:
                all_params.append(param.chain)
        return(all_params)
        
    def merge(self,all_chains):
        merged_chains=[]
        for n in range(0,len(self.theta)):
            merged_chains.append(np.concatenate((all_chains[n::len(self.theta)])))
        return np.array(merged_chains)
    
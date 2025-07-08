# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:05:32 2025

@author: Luke

SIMPLE TEST OF MCFD FOR A LINE WITH SCATTERED POINTS
"""

import MCFD
import numpy as np
import matplotlib.pyplot as plt
import corner

def line(x,m,c):
    return ((m*x)+c)

def synth_data():
    """
    Synthetic noisey data
    """
    m=3
    c=1
    theta=[m,c]
    
    xs=np.linspace(0,15,100)
    ys=line(xs,*theta)
    for n in range(0,len(ys)):
        ys[n]=ys[n]+np.random.normal(0.0,1.2**2)
    return xs,ys

xs,ys=synth_data()

#params
m=MCFD.param("Grad",-5,5,0,0.2)
c=MCFD.param("Y-intercept",-3,3,0,0.2)
sigma=MCFD.param("Vari Sigma",0,1,0.1,0.1)

theta=[m,c,sigma]
explore=MCFD.explore(xs,ys,line,theta)

n=1000000


explore.first_step()

explore.burn(int(n*0.5)) #Half the runs are for burning in
explore.run(int(n*0.5),True)

mean=[]
param_values=[]
param_names=[]

plt.plot(theta[0].chain,marker=".")
plt.ylim([theta[0].low,theta[0].high])
plt.show()


plt.plot(explore.pass_rate)
plt.title("Pass Rate")
plt.show()


"""
Use the mean value for the chain to calculate the fitted value
"""
for param in explore.theta:
    param_values.append(param.chain)
    mean.append(np.mean(param.chain))
    param_names.append(param.name)
param_values=np.array(param_values).T 
    

plt.scatter(xs,ys,marker=".")
xs=np.linspace(0,15,1000)
plt.plot(xs,line(xs,*mean[:-1]),color="r")
plt.show()

figure=corner.corner(param_values,labels=param_names)

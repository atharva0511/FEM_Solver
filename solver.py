
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from tkinter import ttk
import time 

xis = np.linspace(0,5,4)#np.array([0,2,3,5])
def E(x):
    return 8*10**9
def A(x):
    return 2+2*x
def b(x):
    return 8

uBars = {0:0}

tBars = {5:50}
loads = {1:-2000,4:3000}

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
root.mainloop()

def productFunc(func1,func2):
    def func(x):
        return func1(x)*func2(x)
    return func

def gaussianIntegral(func,x1,x2):
    parameters = [-0.57735,0.57735]
    sum = 0
    for s in parameters:
        x = (1-s)*0.5*x1+(1+s)*0.5*x2
        sum+=func(x)
    return sum*0.5*(x2-x1)

####################### Start of solution ##############################
startTime = time.time()
K = np.zeros(shape = (len(xis),len(xis)))
F = np.zeros(shape=(len(xis),1))

I = np.identity(3)
for i in range(0,len(xis)-1):
    # B transpose * B
    BTB = np.array([[1,-1],[-1,1]])/((xis[i+1]-xis[i])**2)
    def N1(x):
        return np.array([[(x - xis[i+1])/(xis[i]-xis[i+1])],[(x-xis[i])/(xis[i+1]-xis[i])]])
    # elemental stiffness matrix
    Ke = BTB*gaussianIntegral(productFunc(E,A),xis[i],xis[i+1])
    # elemental force matrix
    Fe = gaussianIntegral(productFunc(N1,b),xis[i],xis[i+1])
    
    #aaply non essential conditions
    for pos,traction in tBars.items():
        if pos>=xis[i] and pos<xis[i+1]:
            Fe = Fe + N1(pos)*A(pos)*traction

    # apply loads to Fe matrix
    for pos,load in loads.items():
        if pos>=xis[i] and pos<xis[i+1]:
            Fe = Fe+N1(pos)*load

    F[i:i+2] = F[i:i+2] + Fe
    K[i:i+2,i:i+2] = K[i:i+2,i:i+2]+Ke
    # print(K)

# Solving for displacements
fixedIndices = [k for k in uBars]
fixedIndices.sort()
K_ = np.delete(K,np.s_[fixedIndices],0)
bias = np.zeros(shape=(len(xis)-len(fixedIndices),1))
for i in fixedIndices:
    bias+=(np.array([K_[:,i]]).T)*uBars[i]
F_ = np.delete(F,np.s_[fixedIndices],0)
F_ -=bias
K_ = np.delete(K_,np.s_[fixedIndices],1)
u = np.linalg.inv(K_)@F_

U = np.zeros(shape=(len(xis),1))

# Arrange the u matrix
ind = 0
for i in range(0,len(xis)):
    if i in fixedIndices:
        U[i,0] = uBars[i]
    else:
        U[i,0] = u[ind,0]
        ind+=1

############# !!!!!!!!!!!!!!!!!!!!!!  DONE !!!!!!!!!!!!!!!!!!!!! #####################
print(time.time()-startTime)

if len(xis)<6:
    print("displacements: \n",U)
    print("stiffness matrix:\n ",K)
    print("force matrix:\n ",F)

#plots
plt.plot(xis,U)
plt.show()


    
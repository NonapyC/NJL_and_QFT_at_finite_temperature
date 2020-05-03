#-----------------------------------------------------------
#Caluculating the quark mass by solving the gap equation at finite chemical potential
#and temperature.
#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy import integrate
from math import exp, sqrt, log, pi, tanh
plt.style.use('ggplot')
font = {'family' : 'meiryo'}

L = 631.0       #CUTOFF_CONST (MeV)
G = 0.000005073 #COUPLING_CONST (MeV^(-2))
m = 5.5         #CURRENT_MASS (MeV)

Nmax = 200   #T upper limit(MeV)
Nmin = 32   #T lower limit(MeV)
N = 1       #inverse of step size
Mmax = 400   #mu upper limit (MeV)
Mmin = 1   #mu lower limit(MeV)
M = 1       #inverse of setp size

#--------------------------------------------------------------
#Solve the nonliner eqs for m* and mu.
#We define the temperature at which m* has reached half as T_c.
#-------------------------------------------------------------
f = open("text.csv", "w")
for t in [n/N for n in range(Nmin*N,Nmax*N)]:
    for mu in [n/M for n in range(Mmin*M,Mmax*M)]:
        intpart1 = lambda p,x0,x1: p**2/(sqrt(p**2 + x0**2))*(tanh(1/(2*t)*(sqrt(p**2 + x0**2)-x1)) + tanh(1/(2*t)*(sqrt(p**2 + x0**2)+x1)))
        intpart2 = lambda p,x0,x1: p**2 * (tanh(1/(2*t)*(sqrt(p**2 + x0**2)+x1)) - tanh(1/(2*t)*(sqrt(p**2 + x0**2)-x1))) #muの積分part
        def fun(x):
            return [m + x[0]*G*13/(2*pi**2)*integrate.quad(intpart1, 0, L, args = (x[0],x[1]))[0]-x[0], mu-G/(pi**2)*integrate.quad(intpart2, 0, L, args = (x[0],x[1]))[0]-x[1]]
        sol = optimize.root(fun, [500,300] ,method = 'hybr')
        if sol.x[0] < 336.042296117/2: 
            print(sol.x[0], ",", t, ",",mu)
            f.writelines([str(t),",",str(mu),",",str(sol.x[0]),",",str(),"\n"])
            break

f.writelines([str(),",",str(356.6),",",str(),",",str(1),"\n"])
f.writelines([str(),",",str(356.6),",",str(),",",str(3),"\n"])
f.writelines([str(),",",str(356.4),",",str(),",",str(5),"\n"])
f.writelines([str(),",",str(356.0),",",str(),",",str(7),"\n"])
f.writelines([str(),",",str(355.6),",",str(),",",str(10),"\n"])
f.writelines([str(),",",str(354.7),",",str(),",",str(13),"\n"])
f.writelines([str(),",",str(354.1),",",str(),",",str(15),"\n"])
f.writelines([str(),",",str(353.4),",",str(),",",str(17),"\n"])
f.writelines([str(),",",str(352.2),",",str(),",",str(20),"\n"])
f.writelines([str(),",",str(351),",",str(),",",str(23),"\n"])
f.writelines([str(),",",str(350),",",str(),",",str(25),"\n"])
f.writelines([str(),",",str(347.5),",",str(),",",str(30),"\n"])

f.close()

#---------------------------------------------------
# Draw the Phase-Diagram.
#---------------------------------------------------
data=pd.read_csv("text.csv")
data.plot(x=data.columns[1],y=[data.columns[0],data.columns[3]],title='QCD Phase-Diagram',color=["r","r"],style=["--","-"])
plt.legend(['Crossover line',"1st-order transition line"])
plt.ylim([0,250])
plt.xlim([0,400])
plt.ylabel("Temperature (MeV)")
plt.xlabel("Chemical Potential (MeV)")
plt.text(245,190,"QGP-Phase",fontsize=13)
plt.text(50,75,"Chiral Symmetry Broken",fontsize=13)
plt.text(260,177,r"$\langle\bar{\psi}\psi\rangle=0$",fontsize=10)
plt.text(107,62,r"$\langle\bar{\psi}\psi\rangle\neq0$",fontsize=10)
plt.minorticks_on()
plt.show()

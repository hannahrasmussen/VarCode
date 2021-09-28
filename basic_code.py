#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time

from DES import Differential_Equation_Solver_new as DES
from MD import Muon_Decay as MD
from MD import Meta_Muon_Decay as MMD
from nu_nu_coll import nu_nu_collisions as coll
from CollisionApprox import new_Collision_approx as ca
from Interpolate import interp

# Physical Constants:

# In[2]:


a_value = 1/137    #The fine structure constant; unitless
dtda_part2 = 2*np.pi/3 #constant used in driver
Gf = 1.166*10**-11 #This is the fermi constant in units of MeV^-2
me = .511          #Mass of an electron in MeV
mpi_neutral = 135  #Mass of a neutral pion in MeV
mpi_charged = 139.569  #Mass of a charged pion in MeV
mPL = 1.124*10**22 #Planck mass in MeV
mu = 105.661       #Mass of a muon in MeV
f_pi = 131         #MeV, not really sure what this constant means 
x0 = me/mu


# In[3]:


n = 10                        #number of steps for gauss laguerre quadrature

#boxsize = 1
T_initial = 25
a_initial = 1/T_initial

f_TINY = 1e-20
f_MINI = 1e-25
f_SMALL = 1e-30
f_BUFFER = 1e-40
MIN_eps_BUFFER = 12
a_MAXMULT = 2

#constants relating to the sterile neutrino:
D = 1./1.79**3          #a parameter that acts as a fraction of the number density of fermions?


# In[4]:


@nb.jit(nopython=True)
def I1(eps,x): #Energy Density
    numerator = (np.e**eps)*(eps**2)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return numerator/denominator

@nb.jit(nopython=True)
def I2(eps,x): #Pressure
    numerator = (np.e**eps)*(eps**4)
    denominator = ((eps**2+x**2)**.5)*(np.e**((eps**2+x**2)**.5)+1)
    return numerator/denominator

@nb.jit(nopython=True)
def dI1(eps,x): #Derivative of Energy Density
    numerator = (np.e**eps)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return (-x)*numerator/denominator

@nb.jit(nopython=True)
def dI2(eps,x): #Derivative of Pressure
    numerator = (np.e**eps)*3*(eps**2)
    denominator = ((eps**2+x**2)**.5)*(np.e**((eps**2+x**2)**.5)+1)
    return (-x)*numerator/denominator

eps_values, w_values = np.polynomial.laguerre.laggauss(10)

@nb.jit(nopython=True)
def calc_I1(x):
    return np.sum(w_values*I1(eps_values,x)) 

@nb.jit(nopython=True)
def calc_I2(x):
    return np.sum(w_values*I2(eps_values,x))

@nb.jit(nopython=True)
def calc_dI1(x):
    return np.sum(w_values*dI1(eps_values,x)) 

@nb.jit(nopython=True)
def calc_dI2(x):
    return np.sum(w_values*dI2(eps_values,x)) 

def calculate_integral(n,I,x): #n is number of steps to take, I is the function to integrate over, x is me/temp 
    return np.sum(w_values*I(eps_values,x))  


# In[5]:
#@nb.jit(nopython=True)
#def diracdelta(difference,i,E_array): #NOTE: this code is written to calculate the energy range of particle B in the decay
#    #defining boxsizeL and boxsizeR:
#    if i==0:
#        boxsizeR = E_array[1] - E_array[0]
#        boxsizeL = boxsizeR
#    elif len(E_array)-i==1:
#        boxsizeL = E_array[i] - E_array[i-1]
#        boxsizeR = boxsizeL
#    else: 
#        boxsizeL = E_array[i] - E_array[i-1]
#        boxsizeR = E_array[i+1] - E_array[i]
#    if -(1/2)*boxsizeL<= -difference <(1/2)*boxsizeR:
#        return 1/(boxsizeL/2 + boxsizeR/2)
#    else:
#        return 0

@nb.jit(nopython=True)
def diracdelta(Energy,E0,i,E_array):
    if i==0:
        boxsizeR = E_array[1] - E_array[0]
        boxsizeL = boxsizeR
    elif len(E_array)-i==1:
        boxsizeL = E_array[i] - E_array[i-1]
        boxsizeR = boxsizeL
    else: 
        boxsizeL = E_array[i] - E_array[i-1]
        boxsizeR = E_array[i+1] - E_array[i]
    
    x = E0 - Energy
    if E0 - 0.6 * boxsizeR <= Energy <= E0 - 0.4 * boxsizeR:
        x = E0 - (Energy + 0.5 * boxsizeR)
        A = 0.1 * boxsizeR
        return 2/(boxsizeR + boxsizeL) * (0.5 + 0.75 / A**3 * (x**3 / 3 - A**2 * x))
    elif E0 - 0.4 * boxsizeR <= Energy <= E0 + 0.4 * boxsizeL:
        return 2 / (boxsizeL + boxsizeR)
    elif E0 + 0.4 * boxsizeL <= Energy <= E0 + 0.6 * boxsizeL:
        x = E0 - (Energy - 0.5 * boxsizeL)
        A = 0.1 * boxsizeL
        return 2/(boxsizeR + boxsizeL) * (0.5 - 0.75 / A**3 * (x**3 / 3 - A**2 * x))
    else:
        return 0
    

@nb.jit(nopython=True)
def diracdelta2(Energy,Emin,Emax,E_B,gammaL,v,i,E_array): #E_array is the energy array, i is the index of the box we're at now
    #defining boxsizeL and boxsizeR:
    if i==0:
        boxsizeR = E_array[1] - E_array[0]
        boxsizeL = boxsizeR
    elif len(E_array)-i==1:
        boxsizeL = E_array[i] - E_array[i-1]
        boxsizeR = boxsizeL
    else: 
        boxsizeL = E_array[i] - E_array[i-1]
        boxsizeR = E_array[i+1] - E_array[i]
        
    r = 1/(2 * gammaL * v * E_B)
    if Emin - 0.5*boxsizeR <= Energy <= Emin:
        return r * (Energy + boxsizeR - Emin - 0.5 * boxsizeR) * 2 / (boxsizeR + boxsizeL)
    elif Emin <= Energy <= Emin + 0.5*boxsizeL:
        return r * (Energy + boxsizeR - Emin - 0.5 * boxsizeR) * 2 / (boxsizeR + boxsizeL)
    elif Emin + 0.5*boxsizeL <= Energy <= Emax - 0.5 * boxsizeR:
        return r
    elif Emax - 0.5* boxsizeR <= Energy <= Emax:
        return r * (Emax - (Energy-boxsizeL) - 0.5 * boxsizeL) * 2 / (boxsizeR + boxsizeL)
    elif Emax <= Energy <= Emax + 0.5*boxsizeL:
        return r * (Emax - (Energy - boxsizeL) - 0.5 * boxsizeL) * 2 / (boxsizeR + boxsizeL)
    else:
        return 0
        

#@nb.jit(nopython=True)
#def diracdelta(difference,de): #NOTE: this code is written to calculate the energy range of particle B in the decay
#    if -(1/2)*de<= difference <(1/2)*de: 
#        return 1/de
#    else:
#        return 0
#
#@nb.jit(nopython=True)
#def diracdelta2(Energy,Emin,Emax,E_B,gammaL,v,boxsize): 
#    if (Emin-boxsize)<=Energy<=Emin:
#        return (Energy+boxsize-Emin)/(2*gammaL*v*E_B*boxsize)
#    elif (Emax-boxsize)<=Energy<=Emax:
#        return (Emax-Energy)/(2*gammaL*v*E_B*boxsize)
#    elif Emin<= Energy <=Emax: 
#        return 1/(2*gammaL*v*E_B) #where E_B is the energy of the neutrinos in the other frame (pion at rest)
#    else:
#        return 0

@nb.jit(nopython=True)
def pB_other(mA,mB,mC):
    part1 = (mB**2-mA**2-mC**2)/(-2*mA)
    return (part1**2-mC**2)**.5

@nb.jit(nopython=True)
def energyandmomentum(mA,mB,mC,EA,theta): #theta is the angle at which the sterile neutrino decays
    pA = (EA**2-mA**2)**(1/2)
    pBo = pB_other(mA,mB,mC)
    pCo = pBo
    pxB = pBo*np.sin(theta)
    pxC = pCo*np.sin(theta+np.pi)
    pzB = .5*pA + (EA*pBo*np.cos(theta)/mA) + (pA*(mB**2 - mC**2)/(2*mA**2))
    pzC = .5*pA + (EA*pCo*np.cos(theta+np.pi)/mA) + (pA*(mC**2 - mB**2)/(2*mA**2))
    EB = (.5*EA) + (EA*(mB**2 - mC**2)/(2*mA**2)) + (pA*pBo*np.cos(theta)/mA)
    EC = (.5*EA) + (EA*(mC**2 - mB**2)/(2*mA**2)) + (pA*pCo*np.cos(theta+np.pi)/mA)
    return pxB,pxC,pzB,pzC,EB,EC

@nb.jit(nopython=True)
def energyB(mA,mB,mC,EA,theta): #NOTE: I keep putting 0 in for theta because it shouldn't matter yet, but it will soon
    pA = (EA**2-mA**2)**(1/2)
    pBo = pB_other(mA,mB,mC)
    EB = (.5*EA) + (EA*(mB**2 - mC**2)/(2*mA**2)) + (pA*pBo*np.cos(theta)/mA)
    return EB


@nb.jit(nopython=True)
def plasmaenergy(mA,mB,mC,EA,theta): #again, this function assumes that the neutrino is particle B
    pA = (EA**2-mA**2)**(1/2)
    pBo = pB_other(mA,mB,mC)
    pCo = pBo
    EB = (.5*EA) + (EA*(mB**2 - mC**2)/(2*mA**2)) + (pA*pBo*np.cos(theta)/mA)
    EC = (.5*EA) + (EA*(mC**2 - mB**2)/(2*mA**2)) + (pA*pCo*np.cos(theta+np.pi)/mA)
    return EC/(EB+EC) #Returns FRACTION of energy given to the plasma in the decay process

#@nb.jit(nopython=True)
#def trapezoid(array,dx): #dx will just be boxsize for our cases currently
#    total = np.sum(dx*(array[1:]+array[:-1])/2)
#    return total

@nb.jit(nopython=True)
def trapezoid(y_array,x_array):
    total = np.sum((x_array[1:]-x_array[:-1])*(y_array[1:]+y_array[:-1])/2)
    return total

@nb.jit(nopython=True)
def decay2(ms,angle):  #angle is the mixing angle of vs with active neutrinos
    numerator = 9*(Gf**2)*a_value*(ms**5)*((np.sin(angle))**2)
    denominator = 512*np.pi**4
    gamma = numerator/denominator
    return gamma

@nb.jit(nopython=True)
def decay5(ms,angle): #angle is the mixing angle of the sterile neutrino with the active neutrinos
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    part2 = ms*((ms**2)-(mpi_neutral**2))*(np.sin(angle))**2
    gamma = part1*part2
    return gamma

@nb.jit(nopython=True)
def decay6(ms,angle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+me)**2)*((ms**2) - (mpi_charged-me)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(angle))**2
    gamma = part1*part2
    return 2*gamma #because vs can decay into either pi+ and e- OR pi- and e+

@nb.jit(nopython=True)
def decay7(ms,angle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(angle))**2
    gamma = part1*part2
    return 2*gamma #because vs can decay into either pi+ and u- OR pi- and u+

@nb.jit(nopython=True)
def tH(ms,angle):
    return 1/(decay2(ms,angle)+decay5(ms,angle)+decay6(ms,angle)+decay7(ms,angle))

@nb.jit(nopython=True)
def nH(Tcm,t,ms,angle): #number density of decaying particles
    part1 = D*3*1.20206/(2*np.pi**2)
    part2 = Tcm**3*np.e**(-t/tH(ms,angle))
    return part1*part2

#constants relating to decay 6 and 7 that could be calculated outside of driver:
E_B6 = energyB(mpi_charged,0,mu,mpi_charged,0) #describes the energy of the muon neutrino from the pion decay from decay 6 in the rest frame of the pion 
p_B6 = E_B6 #because the neutrino is massless
E_B7 = energyB(mpi_charged,0,mu,mpi_charged,0) #describes the energy of the muon neutrino from the pion decay from decay 7 in the rest frame of the pion
p_B7 = E_B7 #because the neutrino is massless
E_mu6 = mpi_charged - E_B6 #describes the energy of the muon from the pion decay from decay 6 in the rest frame of the pion
p_mu6 = (E_mu6**2 - mu**2)**(1/2) #momentum of the muon in the rest frame of the pion
E_mu7 = mpi_charged - E_B7 #describes the energy of the muon from the pion decay from decay 7 in the other frame
p_mu7 = (E_mu7**2 - mu**2)**(1/2) #momentum of the muon in the other frame


# In[6]:


@nb.jit(nopython=True)
def dP(ms,angle,mB,mC,EA,decay):
    gammaU = decay(ms,angle) 
    gammaL = EA/mA
    pA = (EA**2-mA**2)**(1/2)
    v = pA/EA
    pxB,pxC,pzB,pzC,EB,EC = energyandmomentum(mA,mB,mC,EA,0) #just putting 0 in for theta because I think it's arbitrary right now
    prob = (gammaU/(2*gammaL**2*v*pB_other(mA,mB,mC)))*diracdelta(mA,mB,mC,EA,EB)
    return prob


# In[7]:


@nb.jit(nopython=True)
def poly(x,yp,xp,o):
    y=0
    for i in range(o):
        su=1
        sd=1
        for j in range(o):
            if j != i:
                su=su*(x-xp[j])
                sd=sd*(xp[i]-xp[j])
        y=y+((su/sd)*np.log(np.abs(yp[i])))
    if yp[0]>=0:
        c=np.exp(y)
    else:
        c=-np.exp(y)
    return c

@nb.jit(nopython=True)
# Note: j is an index, not a p-value
def C_round(j,f1,p):
    c,c_frs = coll.cI(j,f1,p)
    if abs(c/c_frs) < 3e-15:
        return 0
    else:
        return c

#This only includes nu-nu, not nu-e;  nu-e added in model in driver()
@nb.jit(nopython=True,parallel=True)
def C_short(p,f1,T,k):
    c = np.zeros(len(p))
    
    for i in nb.prange(1,len(p)-1):
        if i < k[0]:
            c[i] = C_round(i,f1[:k[1]],p[:k[1]])
#            c[i] += ve.cI(i, f1[:k[1]],p[:k[1]],T)
        elif i < k[1]:
            c[i] = C_round(i,f1[:k[2]],p[:k[2]])
#            c[i] += ve.cI(i, f1[:k[2]],p[:k[2]],T)
        else:
            c[i] = C_round(i,f1,p)
#            c[i] += ve.cI(i,f1,p,T)
    return c

'''
def find_breaks(f):
    k_0 = np.where(f < f_TINY)[0][0]
    k_1 = np.where(f < f_MINI)[0][0]
    k_2 = np.where(f < f_SMALL)[0][0]
    for i in range(k_0, len(f)):
        if f[i] > f_TINY:
            k_0 = i+1
    for i in range(k_1,len(f)):
        if f[i] > f_MINI:
            k_1 = i+1
    for i in range(k_2,len(f)):
        if f[i] > f_SMALL:
            k_2 = i+1
    if k_0 >= len(f):
        k_0 = len(f) - 1
    if k_1 >= len(f):
        k_1 = len(f) - 1
    if k_2 >= len(f):
        k_2 = len(f) - 1
    return [k_0, k_1, k_2]
'''

def find_breaks(f, E5_index=0, E2_index=0):
    if (len(np.where(f < f_TINY)[0]) > 0):
        k_0 = np.where(f < f_TINY)[0][0]
    else: 
        k_0 = len(f) - 1
    if (len(np.where(f < f_MINI)[0]) > 0):
        k_1 = np.where(f < f_MINI)[0][0]
    else:
        k_1 = len(f) - 1
    if (len(np.where(f < f_SMALL)[0]) > 0):
        k_2 = np.where(f < f_SMALL)[0][0]
    else:
        k_2 = len(f) - 1
    
    for i in range(k_0, len(f)):
        if f[i] > f_TINY:
            k_0 = i+1
    for i in range(k_1,len(f)):
        if f[i] > f_MINI:
            k_1 = i+1
    for i in range(k_2,len(f)):
        if f[i] > f_SMALL:
            k_2 = i+1
            
    Echeck = [E5_index, E2_index]
    k_return = [k_0, k_1, k_2]
    for j in range(3):
        for i in range(2):
            if Echeck[i] - MIN_eps_BUFFER < k_return[j] <= Echeck[i]:
                k_return[j] += 2 * MIN_eps_BUFFER
            if Echeck[i] <= k_return[j] < Echeck[i] + MIN_eps_BUFFER:
                k_return[j] += MIN_eps_BUFFER
        for jj in range(j+1,3):
            if k_return[jj] < k_return[j] + MIN_eps_BUFFER:
                k_return[jj] = k_return[j] + MIN_eps_BUFFER
        if k_return[j] > len(f):
            k_return[j] = len(f) - 1
#    if k_0 >= len(f):
#        k_0 = len(f) - 1
#    if k_1 >= len(f):
#        k_1 = len(f) - 1
#    if k_2 >= len(f):
#        k_2 = len(f) - 1
    return k_return

# In[44]:


# This function assumes T_initial * a_initial = 1
# here, e_array is a required input.  Boxsize is calculated from that.
def driver(ms,mixangle,a_init,y_init, e_array, eps_small, eps_buffer, dx, N_steps = 10, dN_steps = 10, pl_last = False, first = False, temp_fin=0):
    
    A_model, n_model = ca.model_An(a_init, y_init[-2], 0.9*y_init[-2] * a_init)
    @nb.jit(nopython=True)
    def C_ve(p_array, Tcm, T, f):
        C_array = p_array**n_model * (f - ca.f_eq(p_array, T, 0))
        return - A_model * ca.n_e(T) * Gf**2 * T**(2-n_model) * C_array
    
    #constants so we don't have to calculate these a bunch of times in derivatives function:
    d2constant = decay2(ms,mixangle) 
    d5constant = decay5(ms,mixangle) 
    d6constant = decay6(ms,mixangle) 
    d7constant = decay7(ms,mixangle)
    MMDinit = MMD.gammanu((mu/2)*(1-x0**2))
    MDinit = MD.gammanu((mu/2)*(1-x0**2))


    num = len(y_init)
#    e_array = np.linspace(0,(num-4)*boxsize,num-3)#epsilon array
#    boxsize = e_array[1] - e_array[0]
    E_B2 = energyB(ms,0,0,ms,0)
    E_B5 = energyB(ms,0,135,ms,0)
    
    #constants referrring to decay 6; the initial decay and the decay of the pion into the muon
    E_pi6 = energyB(ms,mpi_charged,me,ms,0) #energy of the charged pion from decay 6, theta is 0 because this is our chosen direction
    p_pi6 = (E_pi6**2 - mpi_charged**2)**(1/2) #momentum of charged pion from decay 6
    #theta = 2*np.pi*np.random.rand()
    gammapi6 = E_pi6/mpi_charged
    v6 = p_pi6/E_pi6
    E_B6max = gammapi6*(E_B6 + (v6*E_B6))
    E_B6min = gammapi6*(E_B6 - (v6*E_B6))
    
    #constants referring to decay 7; the initial decay, the decay of the pion into the muon, and the decay of the FIRST muon 
    E_pi7 = energyB(ms,mpi_charged,mu,ms,0) #energy of the charged pion from decay 7, theta is 0 because this is our chosen direction
    Eu = ms-E_pi7 #Energy of the FIRST muon from decay 7, contains the rest of the energy that didn't go into the pion
    p_pi7 = (E_pi7**2 - mpi_charged**2)**(1/2) #momentum of charged pion from decay 7
    #theta = 2*np.pi*np.random.rand()
    gammapi7 = E_pi7/mpi_charged
    v7 = p_pi7/E_pi7
    E_B7max = gammapi7*(E_B7 + (v7*E_B7))
    E_B7min = gammapi7*(E_B7 - (v7*E_B7))
    
    #constants referring to the muon decay in decay 6:
    #theta = 2*np.pi*np.random.rand()
    E_mumin6 = gammapi6*(E_mu6 - (v6*p_mu6))
    E_mumax6 = gammapi6*(E_mu6 + (v6*p_mu6))
    
    #constants referring to the SECOND muon decay in decay 7:
    #theta = 2*np.pi*np.random.rand()
    E_mumin7 = gammapi7*(E_mu7 - (v7*p_mu7))
    E_mumax7 = gammapi7*(E_mu7 + (v7*p_mu7))
    
    def make_der(kk):
        @nb.jit(nopython=True)
        def derivatives(a,y): #y is a vector with length 102 for now, y[-2] is temp and y[-1] is time, the rest are prob functions for now
            d_array = np.zeros(len(y))
            Tcm = 1/a #We always need T_initial * a_initial = 1

            dtda_part1 = mPL/(2*a)
            dtda_part3 = (y[-2]**4*np.pi**2)/15
            dtda_part4 = 2*y[-2]**4*calc_I1(me/y[-2])/np.pi**2
            dtda_part6 = ms*nH(Tcm,y[-1],ms,mixangle)
#            dtda_part7 = (Tcm**4/(2*np.pi**2))*trapezoid(y[:int(num-3)]*e_array[:int(num-3)]**3,boxsize)
            dtda_part7 = (Tcm**4/(2*np.pi**2))*trapezoid(y[:int(num-3)]*e_array[:int(num-3)]**3,e_array[:int(num-3)])
            dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part6+dtda_part7))**.5
            d_array[-1] = dtda

            #df/da for the neutrinos and antineutrinos at epsilon = 0:
            d6b_e0 = 2*(1-x0**2)*d6constant*gammapi6*(mu**2)*(Gf**2)*E_mu6*nH(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*MMDinit)
            d7b_e0 = 2*(1-x0**2)*d7constant*(Eu/mu)*(mu**2)*(Gf**2)*nH(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*MDinit)
            d7c_e0 = 2*(1-x0**2)*d7constant*gammapi7*(mu**2)*(Gf**2)*E_mu7*nH(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*MMDinit)
            d_array[0] = d6b_e0+d7b_e0+d7c_e0

            if kk[0] == 0:
                c = coll.C(e_array*Tcm,y[:num-3]) 
                c += C_ve(e_array*Tcm, Tcm, y[-2], y[:num-3])
                c *= dtda
            else:
                c = C_short(e_array*Tcm,y[:num-3],y[-2],kk) 
                c += C_ve(e_array*Tcm, Tcm, y[-2], y[:num-3])
                c *= dtda
            for i in range (1,num-3): #because all the derivatives are dF/da except Temp and Time
                eps = e_array[i]
                coefficient = (2*np.pi**2)/(eps**2*Tcm**2*a**3)
#                d2 = (decay2(ms,mixangle)*diracdelta((i*boxsize*Tcm)-E_B2,boxsize*Tcm)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
#                d5 = (decay5(ms,mixangle)*diracdelta((i*boxsize*Tcm)-E_B5,boxsize*Tcm)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
#                d6 = .5*(decay6(ms,mixangle)*diracdelta2((i*boxsize*Tcm),E_B6min,E_B6max,E_B6,gammapi6,v6,boxsize)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
                d2 = (d2constant*diracdelta((eps*Tcm),E_B2,i,e_array*Tcm)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
                d5 = (d5constant*diracdelta((eps*Tcm),E_B5,i,e_array*Tcm)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
                d6 = .5*(d6constant*diracdelta2((eps*Tcm),E_B6min,E_B6max,E_B6,gammapi6,v6,i,e_array*Tcm)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
#                d6b = decay6(ms,mixangle)*(1/(2*gammapi6*v6*p_mu6))*MMD.u_integral(E_mumin6,E_mumax6,i*boxsize*Tcm,ms,mixangle)*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda
                d6b = d6constant*(1/(2*gammapi6*v6*p_mu6))*MMD.u_integral(E_mumin6,E_mumax6,eps*Tcm,ms,mixangle)*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda
#                d7a = .5*(decay7(ms,mixangle)*diracdelta2((i*boxsize*Tcm),E_B7min,E_B7max,E_B7,gammapi7,v7,boxsize)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
                d7a = .5*(d7constant*diracdelta2((eps*Tcm),E_B7min,E_B7max,E_B7,gammapi7,v7,i,e_array*Tcm)*nH(Tcm,y[-1],ms,mixangle)*a**3)*dtda
#                d7b = MD.v(Eu,i*boxsize*Tcm,ms,mixangle)*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda #times two because there are 2 neutrinos coming out with the same energy distribution
                d7b = MD.v(Eu,eps*Tcm,ms,mixangle)*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda #times two because there are 2 neutrinos coming out with the same energy distribution
#                d7c = decay7(ms,mixangle)*(1/(2*gammapi7*v7*p_mu7))*MMD.u_integral(E_mumin7,E_mumax7,i*boxsize*Tcm,ms,mixangle)*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda
                d7c = d7constant*(1/(2*gammapi7*v7*p_mu7))*MMD.u_integral(E_mumin7,E_mumax7,eps*Tcm,ms,mixangle)*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda
                d_array[i] = coefficient*(d2+d5+d6+d6b+d7a+d7b+d7c) + c[i]#neutrinos only, antineutrinos not included

            df_array = d_array[:-3]*e_array**3/(2*np.pi**2) 
            dQda_part1 = ms*nH(Tcm,y[-1],ms,mixangle)*a**3*dtda/tH(ms,mixangle)
#            dQda_part2 = Tcm**4*a**3*trapezoid(df_array,boxsize)
            dQda_part2 = Tcm**4*a**3*trapezoid(df_array,e_array)
            dQda = dQda_part1-dQda_part2
            d_array[-3] = dQda

            dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calc_I1(me/y[-2]) + (1/3)*(calc_I2(me/y[-2])))
            dTda_constant2 = 2*me*y[-2]*a**3/(np.pi**2)
            dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
            dTda_numerator2 = dQda/y[-2]
            dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) - (dTda_constant2*(calc_dI1(me/y[-2]))) - ((1/3)*dTda_constant2*(calc_dI2(me/y[-2])))
            dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
            d_array[-2] = dTda

            return d_array
        return derivatives
    
#    a_array, y_matrix = DES.destination_x(derivatives, y_init, steps, d_steps, a_init, a_final)
    a0 = a_init
    y0 = np.zeros(num)
    
    a_max = max(e_array[-MIN_eps_BUFFER-1] / E_B2, a0 * a_MAXMULT)
    
    a_results = []
    y_results = []
    

    for i in range(len(y0)):
        y0[i] = y_init[i]
    a_results.append(a0)
    y_results.append(y0)
    for i in range(N_steps):
        if first:
            k = [0,0,0]
        else:
            k = find_breaks(y0[:num-3],E5_index = np.where(e_array < E_B5 * a0)[0][-1], E2_index = np.where(e_array < E_B2 * a0)[0][-1])
        print(e_array[k[0]], e_array[k[1]], e_array[k[2]], e_array[-1] - e_array[-2])
        a_array, y_matrix, dx_new = DES.destination_x_dx(make_der(np.array(k)), y0,1, dN_steps, a0, a_max, dx)
 
        if a_array[-1] >= a_max:       
#        if np.where(e_array < E_B2 * a_array[-1])[0][-1] > len(e_array) - MIN_eps_BUFFER:
            break
        
        a0 = a_array[-1]
        y0 = np.zeros(num)
        for j in range(len(y0)):
            y0[j] = y_matrix[j][-1]
        dx = dx_new
        print(i+1, a0, dx)
        
        if a0 == a_results[-1]:
            break
            
        a_results.append(a0)
        y_results.append(y0)
        
        if a0 >= a_init * a_MAXMULT:
            break
        eps = np.where(y0[:-3] > f_SMALL)[0][-1]             ## Changed this line! (CK-7/16/20)
        if eps - eps_small > (eps_buffer - eps_small) / 2:
            break
        
        if y0[-2] < temp_fin:
            break
                
#    a_array, y_matrix = DES.destination_x(derivatives, y_init, N_steps, dN_steps, a_init, a_final)
#    pl_vals = [0,200,400,600,800,1000]
#    pl_vals = [0,2,4,6,8]
    if pl_last:
        f_v = np.zeros(len(e_array))
        for j in range(len(e_array)):
            f_v[j] = y_matrix[j][-1]
        print(1/a_array[-1])
        plt.figure()
        plt.semilogy(e_array,f_v)
        plt.semilogy(e_array,y_init[:num-3],linestyle='--')

        d = derivatives(a_array[-1],np.transpose(y_matrix)[:][-1])
        c = coll.C(e_array*1/a_array[-1],f_v) * d[-1]
        plt.figure()
        plt.loglog(e_array,c,linestyle='--',color='k')
        plt.loglog(e_array,-c,linestyle='--',color='r')
        plt.loglog(e_array,d[:num-3],color='k')
        plt.show()

    return a_results, y_results, dx


# In[45]:

def forward(mH, y_v, e_array, a):
    eps_small_new = np.where(y_v[:-3] > f_SMALL)[0][-1]
    
    e_up = np.where(e_array <  0.5 * mH * a )[0][-1]
    if e_up == len(e_array) - 1:
        e_up = int( 0.5 * mH * a / (e_array[-1] - e_array[-2]))
        e_up += MIN_eps_BUFFER
#    eps_small_new = int(max(eps_small_new, e_up))
        
    xp = np.zeros(2)
    yp = np.zeros(2)
    kk = 0
    while (yp[1] >= yp[0]):
        if eps_small_new + kk >= len(e_array):
            break
        else:
            for i in range(2):
                yp[i] = y_v[eps_small_new + kk + i - 1]
                xp[i] = e_array[eps_small_new + kk + i - 1]
            kk += 1
    
    if eps_small_new + kk == len(e_array):
        print("f is increasing at last box?")
        return y_v, e_array
    
    bxsz = abs(xp[1] - xp[0])
    new_len = len(y_v)
    if y_v[eps_small_new] < f_BUFFER:
        new_len =  eps_small_new
    else:
        for i in range(20*eps_small_new):
            if interp.log_linear_extrap(xp[0] + i * bxsz, xp, yp) > f_BUFFER:
                new_len = i + eps_small_new
            else:
                break
        new_len = max(eps_small_new + MIN_eps_BUFFER, new_len + 1) + 1
    new_len = max(new_len, e_up)
    y = np.zeros(new_len+3)
    y[-1] = y_v[-1]
    y[-2] = y_v[-2]
    y[-3] = y_v[-3]
    
    eps_array = np.zeros(new_len)
    for i in range(eps_small_new):
        eps_array[i] = e_array[i]
        y[i] = y_v[i]
    if len(eps_array) > eps_small_new:
        for i in range(eps_small_new, len(eps_array)):
            eps_array[i] = e_array[eps_small_new-1] + (i+1 - eps_small_new) * bxsz
            y[i] = interp.log_linear_extrap(eps_array[i], xp, yp)

    return y, eps_array    


def next_step(mH,mixangle,a,y,dx,e_array,eps_small,eps_buffer,Ns,dNs,fn,nr,plot=True,first=False,temp_fin=0):
    st = time.time()
    a_v, y_v, dx = driver(mH,mixangle,a,y,e_array,eps_small, eps_buffer, dx, N_steps = Ns, dN_steps = dNs,first=first,temp_fin=temp_fin)
    print("Run time: {:.3} hrs".format((time.time()-st)/3600) )
    y_save_form = []
    for i in range(len(y)):
        y_save_form.append(np.zeros(len(a_v)))
    for i in range(len(a_v)):
        for j in range(len(y)):
            y_save_form[j][i] = y_v[i][j]

    temp_fn = '{}-{}'.format(fn,nr)
    np.save(temp_fn + '-a',a_v)
    np.save(temp_fn + '-y',y_save_form)
    np.save(temp_fn + '-e',e_array)
###############
##  These lines have been added (CK-7/20/20)    
#    eps_small_new = np.where(y_v[-1][:-3] > f_SMALL)[0][-1]
#        
#    xp = np.zeros(2)
#    yp = np.zeros(2)
#    for i in range(2):
#        yp[i] = y_v[-1][eps_small_new + i -1]
#        xp[i] = e_array[eps_small_new + i -1]
#        
#    bxsz = abs(xp[1] - xp[0])
#    new_len = len(y_v[-1])
#    for i in range(20*eps_small_new):
#        if interp.log_linear_extrap(xp[0] + i * bxsz, xp, yp) > f_BUFFER:
#            new_len = i + eps_small_new
#        else:
#            break
#    
#    new_len = max(eps_small_new + MIN_eps_BUFFER, new_len + 1) + 1
#    
#    y = np.zeros(new_len+3)
#    y[-1] = y_v[-1][-1]
#    y[-2] = y_v[-1][-2]
#    y[-3] = y_v[-1][-3]
#    
#    eps_array = np.zeros(new_len)
#    for i in range(eps_small_new):
#        eps_array[i] = e_array[i]
#        y[i] = y_v[-1][i]
#    for i in range(eps_small_new, len(eps_array)):
#        eps_array[i] = e_array[eps_small_new-1] + (i+1 - eps_small_new) * bxsz
#        y[i] = interp.log_linear_extrap(eps_array[i], xp, yp)
        
    a = a_v[-1]
    y, eps_array = forward(mH, y_v[-1], e_array, a)

    eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
    eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
#######################

#    eps_small_new = np.where(y_v[-1] < f_SMALL)[0][0]
#
#    xp = np.zeros(2)
#    yp = np.zeros(2)
#
#    for i in range(2):
#        yp[i] = y_v[-1][eps_small_new + 1 - i]
#        xp[i] = eps_small_new + 1 - i
#    ov = 2
#    print(xp)
#    new_len = len(y_v[-1])
#    for i in range(eps_small_new,20*eps_small_new):
#        if poly(i,yp,xp,ov) > f_BUFFER:
#            new_len = i
#        else:
#            break
#
#    new_len = max(eps_small_new + MIN_eps_BUFFER, new_len + 1) + 1
#    a = a_v[-1]
#    y = np.zeros(new_len+3)
#    y[-1] = y_v[-1][-1]
#    y[-2] = y_v[-1][-2]
#    y[-3] = y_v[-1][-3]
#    eps_array = np.zeros(new_len)
#    for i in range(eps_small_new):
#        y[i] = y_v[-1][i]
#        eps_array[i] = i * (e_array[1]-e_array[0])
#    for i in range(eps_small_new,new_len):
#        y[i] = poly(i,yp,xp,ov)
#        eps_array[i] = i * (e_array[1]-e_array[0])
#    eps_small = np.where(y < f_SMALL)[0][0]
#    eps_buffer = np.where(y < f_BUFFER)[0][0]
    
    if plot:
        plt.figure()
        plt.semilogy(y[:-3])
        plt.semilogy(y_v[-1][:-3],linestyle='--')
        plt.figure()
        plt.loglog(e_array, e_array**2 * y_v[-1][:-3],linestyle='--')
        plt.loglog(e_array, e_array**2 / ( np.exp(e_array / y[-2] / a) + 1 ),color='0.50',linestyle='-.')
        plt.ylim(1e-10,10)
        plt.show()
    return a, y, dx, eps_array, eps_small, eps_buffer


# In[46]:


def spread_eps(y,eps_array):    
    y_new = np.zeros(int((len(y)-3)/2)+3)
    eps_arr = np.zeros(len(y_new)-3)
    new_boxsize = (eps_array[1]-eps_array[0])*2
    y_new[-1] = y[-1]
    y_new[-2] = y[-2]
    y_new[-3] = y[-3]
    y_new[0] = y[0]
    dn = y[:-3] * eps_array**2
    for i in range(1,len(eps_arr)):
        eps_arr[i] = i * new_boxsize
        y_new[i] = (- y_new[i-1] * eps_arr[i-1]**2 + 1/3 * (dn[2*i] + 4 * dn[2*i-1] + dn[2*i-2]))/eps_arr[i]**2
    return y_new, eps_arr
def simple_spread_eps(y,eps_array):    
    y_new = np.zeros(int((len(y)-3)/2)+3)
    eps_arr = np.zeros(len(y_new)-3)
    new_boxsize = (eps_array[1]-eps_array[0])*2
    y_new[-1] = y[-1]
    y_new[-2] = y[-2]
    y_new[-3] = y[-3]
    y_new[0] = y[0]
    for i in range(1,len(eps_arr)):
        eps_arr[i] = i * new_boxsize
        y_new[i] = y[2*i]
    return y_new, eps_arr

def babysteps(mH,mixangle,N_runs,Ns,dNs,filename,prev_run=False,prev_run_file="",prev_i=-1,temp_fin=0.001):
    if prev_run:
        a_g = np.load(prev_run_file + '-a.npy')
        y_g = np.load(prev_run_file + '-y.npy')
        a = a_g[prev_i]
        y = np.zeros(len(y_g))   #must match num above
        for i in range(len(y)):
            y[i] = y_g[i][prev_i]
        try:
            eps_arr = np.load(prev_run_file + '-e.npy')
        except:
            print("No file {} found, using boxsize = 1".format(prev_run_file + '-e.npy'))
            eps_arr = np.linspace(0,(len(y)-4)*1,len(y)-3)
        dx = 1e-7
    else:
        # if it's the First run, we assume boxsize = 1
        initial_boxsize = 0.5
        a = 1/15
        eps_small = - int(np.log(f_SMALL)/initial_boxsize)
        eps_buffer = max(eps_small + MIN_eps_BUFFER, - int(np.log(f_BUFFER)/initial_boxsize))
        y = np.zeros(eps_buffer + 3)
        eps_arr = np.zeros(eps_buffer)
        for i in range(len(eps_arr)):
            eps_arr[i] = i * initial_boxsize
            y[i] = 1 / (np.exp(eps_arr[i])+1)
        y[-2] = 15
        
        print("Initial Run. Tcm = {:.3f} MeV, eps_s = {}, eps_b = {}".format(1/a,eps_small,eps_buffer))
        a, y, dx, eps_arr, eps_small, eps_buffer = next_step(mH,mixangle,a,y,1e-7,eps_arr,eps_small,eps_buffer,1,100,filename,0,plot=True,first=True)
        
    eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
    try:
        eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
    except:
        eps_buffer = len(y) - 4

    if eps_small > 500 or len(y) > 1000:
        y, eps_arr = simple_spread_eps(y, eps_arr)
    eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
    try:
        eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
    except:
        eps_buffer = len(y) - 4

    tau = tH(mH,mixangle)

    for nr in range(N_runs):
        print("Run {} of {}. Tcm = {:.3f} MeV, T= {:.3f} MeV, t/tau = {:.3f}, eps_s = {}, eps_b = {}".format(nr+1,N_runs,1/a,y[-2],y[-1]/tau,eps_arr[eps_small],eps_arr[eps_buffer]))
        a, y, dx, eps_arr, eps_small, eps_buffer = next_step(mH,mixangle,a,y,dx,eps_arr,eps_small,eps_buffer,Ns,dNs,filename,nr,plot=True,temp_fin=temp_fin)
        if eps_small > 500 or len(y) > 1000:
            y, eps_arr = simple_spread_eps(y, eps_arr)
        eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
        try:
            eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
        except:
            eps_buffer = len(y) - 4

        if y[-2] < temp_fin:
            break


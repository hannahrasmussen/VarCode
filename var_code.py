#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time

from DES import Differential_Equation_Solver as DES
from nu_nu_coll import nu_nu_collisions as coll
from CollisionApprox import Collision_approx as ca
from Interpolate import interp


# In[2]:


alpha = 1/137
D = 1./1.79**3 
dtda_part2 = 2*np.pi/3
f_pi = 131 
Gf = 1.166*10**-11 
me = .511        
mpi_neutral = 135  
mpi_charged = 139.569  
mPL = 1.124*10**22 
mu = 105.661  
eps_e = me/mu
Enumax = (mu/2)*(1-(eps_e)**2)

# In[3]:


n = 10   
f_TINY = 1e-20
f_MINI = 1e-25
f_SMALL = 1e-30
f_BUFFER = 1e-40
MIN_eps_BUFFER = 12
a_MAXMULT = 2
x_values, w_values = np.polynomial.laguerre.laggauss(n)  
x_valuese, w_valuese = np.polynomial.legendre.leggauss(n)
step_counter_log = 0
small_boxsize = 0.5
eps_small_box = 16
num_small_boxes = int(eps_small_box/small_boxsize) + 1
initial_boxsize = 0.5


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

@nb.jit(nopython=True)
def calculate_integral(I,x): #I is the function to integrate over, x is me/temp 
    return np.sum(w_values*I(x_values,x))  

@nb.jit(nopython=True)
def trapezoid(y_array,x_array):
    total = np.sum((x_array[1:]-x_array[:-1])*(y_array[1:]+y_array[:-1])/2)
    return total


# In[5]:


@nb.jit(nopython=True)
def rate1(ms,mixangle): 
    numerator = 9*(Gf**2)*alpha*(ms**5)*((np.sin(mixangle))**2)
    denominator = 512*np.pi**4
    Gamma = numerator/denominator
    return Gamma

@nb.jit(nopython=True)
def rate2(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    part2 = ms*((ms**2)-(mpi_neutral**2))*(np.sin(mixangle))**2
    Gamma = part1*part2
    return Gamma

@nb.jit(nopython=True)
def rate3(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+me)**2)*((ms**2) - (mpi_charged-me)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma

@nb.jit(nopython=True)
def rate4(ms,mixangle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(mixangle))**2
    Gamma = part1*part2
    return 2*Gamma 

@nb.jit(nopython=True)
def ts(ms,mixangle):
    return 1/(rate1(ms,mixangle)+rate2(ms,mixangle)+rate3(ms,mixangle)+rate4(ms,mixangle))

@nb.jit(nopython=True)
def ns(Tcm,t,ms,mixangle):
    part1 = D*3*1.20206/(2*np.pi**2)
    part2 = Tcm**3*np.e**(-t/ts(ms,mixangle))
    n_s = part1*part2
    return n_s


# In[6]:


@nb.jit(nopython=True)
def diracdelta(E,EI,i,E_arr):
    if i==0:
        bxR = E_arr[1] - E_arr[0]
        bxL = bxR
    elif len(E_arr)-i==1:
        bxL = E_arr[i] - E_arr[i-1]
        bxR = bxL
    else: 
        bxL = E_arr[i] - E_arr[i-1]
        bxR = E_arr[i+1] - E_arr[i]
    
    if EI - 0.6 * bxR <= E <= EI - 0.4 * bxR:
        x = EI - (E + 0.5 * bxR)
        A = 0.1 * bxR
        return 2/(bxL + bxR) * (0.5 + 0.75 / A**3 * (x**3 / 3 - A**2 * x))
    elif EI - 0.4 * bxR <= E <= EI + 0.4 * bxL:
        return 2 / (bxL + bxR)
    elif EI + 0.4 * bxL <= E <= EI + 0.6 * bxL:
        x = EI - (E - 0.5 * bxL)
        A = 0.1 * bxL
        return 2/(bxL + bxR) * (0.5 - 0.75 / A**3 * (x**3 / 3 - A**2 * x))
    else:
        return 0
    

@nb.jit(nopython=True)
def diracdelta2(E,EBmin,EBmax,E_B,gamma,v,i,E_arr): 
    if i==0:
        bxR = E_arr[1] - E_arr[0]
        bxL = bxR
    elif len(E_arr)-i==1:
        bxL = E_arr[i] - E_arr[i-1]
        bxR = bxL
    else: 
        bxL = E_arr[i] - E_arr[i-1]
        bxR = E_arr[i+1] - E_arr[i]
        
    r = 1/(2 * gamma * v * E_B)
    if EBmin - 0.5*bxR <= E <= EBmin:
        return r * (E - EBmin + 0.5 * bxR) * 2 / (bxR + bxL)
    elif EBmin <= E <= EBmin + 0.5*bxL:
        return r * (E - EBmin + 0.5 * bxR) * 2 / (bxR + bxL)
    elif EBmin + 0.5*bxL <= E <= EBmax - 0.5 * bxR:
        return r
    elif EBmax - 0.5*bxR <= E <= EBmax:
        return r * (EBmax - E + 0.5 * bxL) * 2 / (bxR + bxL)
    elif EBmax <= E <= EBmax + 0.5*bxL:
        return r * (EBmax - E + 0.5 * bxL) * 2 / (bxR + bxL)
    else:
        return 0

@nb.jit(nopython=True)
def EB(mA,mB,mC): 
    E_B = (mA**2 + mB**2 - mC**2)/(2*mA)
    return E_B

@nb.jit(nopython=True)
def Gammamua(a,b): #for both electron neutrinos and muon neutrinos for decay types III and IV
    if a>Enumax:
        return 0
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_b1 = (-1/4)*(me**4)*mu*np.log(abs(2*b-mu))
    part_b2 = (-1/6)*b
    part_b3 = 3*(me**4)+6*(me**2)*mu*b
    part_b4 = (mu**2)*b*(4*b-3*mu)
    part_b = (part_b1+part_b2*(part_b3+part_b4))/mu**3
    part_a1 = (-1/4)*(me**4)*mu*np.log(abs(2*a-mu))
    part_a2 = (-1/6)*a
    part_a3 = 3*(me**4)+6*(me**2)*mu*a
    part_a4 = (mu**2)*a*(4*a-3*mu)
    part_a = (part_a1+part_a2*(part_a3+part_a4))/mu**3
    integral = part_b-part_a
    Gam_mua = constant*integral
    return Gam_mua

@nb.jit(nopython=True)
def Gammamub(): #for both electron neutrinos and muon neutrinos for decay types III and IV 
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_a1 = 3*(me**4)*(mu**2)*np.log(abs(2*Enumax-mu))
    part_a2 = 6*(me**4)*Enumax*(mu+Enumax)
    part_a3 = 16*(me**2)*mu*(Enumax**3)
    part_a4 = 4*(mu**2)*(Enumax**3)*(3*Enumax - 2*mu)
    part_a5 = 24*mu**3
    part_b1 = 3*(me**4)*(mu**2)*np.log(abs(-mu))/part_a5
    integral = ((part_a1+part_a2+part_a3+part_a4)/part_a5)-part_b1
    Gam_mub = -1*constant*integral
    return Gam_mub

Gam_mub = Gammamub()

@nb.jit(nopython=True)
def u_integral(E_mumin,E_mumax,Eactive):
    Eu_array = ((E_mumax-E_mumin)/2)*x_valuese + ((E_mumax+E_mumin)/2)
    integral = 0
    for i in range(n):
        gammau = Eu_array[i]/mu
        pu = (Eu_array[i]**2-mu**2)**(1/2)
        vu = pu/Eu_array[i]
        Gam_mua = Gammamua(Eactive/(gammau*(1+vu)),min(Enumax,Eactive/(gammau*(1-vu))))
        integral = integral + (w_valuese[i]*((E_mumax-E_mumin)/2)*(1/(2*gammau*vu))*Gam_mua)
    return integral


# In[7]:


@nb.jit(nopython=True)
def C_round(j,f,p):
    c,c_frs = coll.cI(j,f,p)
    if abs(c/c_frs) < 3e-15:
        return 0
    else:
        return c
    
@nb.jit(nopython=True)
def smallgrid(p,f,k0,k1):
    boxsize = p[-1] - p[-2]
    p_small_boxsize = p[1] - p[0]
    N = max(int(np.round(p[int(k1)-1]/p_small_boxsize,0))+1, int(1.25*k0))
    p_small = np.zeros(N)
    f_small = np.zeros(N)
    
    x_int = np.zeros(6)
    y_int = np.zeros(6)
    
    for i in range(num_small_boxes):
        p_small[i] = p[i]
        f_small[i] = f[i]
    for i in range(num_small_boxes,N):
        p_small[i] = i * p_small_boxsize
        k = int(np.round((p_small[i]-p[num_small_boxes-1])/boxsize,0)) + num_small_boxes - 1
        if np.round((p_small[i] - p[num_small_boxes-1]) / boxsize, 5) % 1 == 0:
            f_small[i] = f[k]
        else:
            if k+3 < len(p):
                for j in range(6):
                    x_int[j] = p[k + j - 2]
                    y_int[j] = f[k + j - 2]
                    if y_int[j] < 0:
                        print ("smallgrid",x_int[j], y_int[j])
                        print (k+j-2, k0, k1)
                        print ("f = ", f)
            else:
                x_int[:] = p[-6:]
                y_int[:] = f[-6:]
            f_small[i] = np.exp(interp.lin_int(p_small[i],x_int,np.log(y_int)))
    return p_small, f_small

@nb.jit(nopython=True)
def biggrid(p,f,k1):
    boxsize = p[-1] - p[-2]
    p_small_boxsize = p[1] - p[0]
    new_small_boxes = int(np.round(p[num_small_boxes-1]/boxsize,0)) + 1
    N = new_small_boxes + int(k1 - num_small_boxes)
    
    p_big = np.zeros(N)
    f_big = np.zeros(N)
    
    mult = int(round(boxsize/p_small_boxsize,0))
    for i in range(new_small_boxes):
        p_big[i] = p[mult * i]
        f_big[i] = f[mult * i]
    for i in range(new_small_boxes, N):
        p_big[i] = p[num_small_boxes + (i-new_small_boxes)]
        f_big[i] = f[num_small_boxes + (i-new_small_boxes)]
        
    return p_big, f_big

@nb.jit(nopython=True,parallel=True)
def C_short(p,f,T,k):
    c = np.zeros(len(p))
    boxsize = p[-1] - p[-2]

    if k[0] == 0:
        p_smallgrid, f_smallgrid = smallgrid(p,f,num_small_boxes,len(p))
        p_wholegrid, f_wholegrid = biggrid(p,f,len(p))
        for i in nb.prange(1,len(p)-1):
            if i < num_small_boxes:
                c[i] = C_round(i, f_smallgrid, p_smallgrid)
            else:
                c[i] = C_round((i-num_small_boxes)*2 + num_small_boxes , f_smallgrid, p_smallgrid)
 #               c[i] = C_round(i-num_small_boxes+1+int(eps_small_box/boxsize),f_wholegrid,p_wholegrid)
    else:
        k0 = num_small_boxes
        p_smallgrid, f_smallgrid = smallgrid(p,f,k0,k[1])
        p_biggrid, f_biggrid = biggrid(p,f,k[2])
        p_wholegrid, f_wholegrid = biggrid(p,f,len(p))

        for i in nb.prange(1,len(p)-1):
            if i < k0:
                c[i] = C_round(i, f_smallgrid, p_smallgrid)
            elif i < k[1]:
                c[i] = C_round(i-num_small_boxes+1+int(np.round(p[num_small_boxes-1]/boxsize,0)), f_biggrid, p_biggrid)
            else:
                c[i] = C_round(i-num_small_boxes+1+int(np.round(p[num_small_boxes-1]/boxsize,0)),f_wholegrid,p_wholegrid)
    return c 

def find_breaks(f, E2_index=0, E1_index=0):
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
            
    Echeck = [E2_index, E1_index]
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
        if k_return[j] >= len(f):
            k_return[j] = len(f) - 1

    return k_return


# In[8]:


def driver(ms, mixangle, a_init, y_init, e_array, eps_small, eps_buffer, dx, N_steps = 10, dN_steps = 10, pl_last = False, first = False, temp_fin=0):
    
    A_model, n_model = ca.model_An(a_init, y_init[-2], 1/(0.9*y_init[-2]*a_init))
    
    num = len(y_init)
    d1r = rate1(ms,mixangle) 
    d2r = rate2(ms,mixangle) 
    d3r = rate3(ms,mixangle) 
    d4r = rate4(ms,mixangle)

    E_B1 = ms/2
    E_B2 = (ms**2 - mpi_neutral**2)/(2*ms)
    
    #constants referring to initial part of decay 3:
    E_pi3 = EB(ms,mpi_charged,me) 
    p_pi3 = (E_pi3**2 - mpi_charged**2)**(1/2) 
    v3 = p_pi3/E_pi3
    gammapi3 = E_pi3/mpi_charged
    
    #constants referring to decay 3a:
    E_B3 = EB(mpi_charged,0,mu)
    E_B3max = gammapi3*E_B3*(1+v3)
    E_B3min = gammapi3*E_B3*(1-v3)
                        
    #additional constants referring to decay 3b:
    E_mu3 = EB(mpi_charged,mu,0) 
    p_mu3 = (E_mu3**2 - mu**2)**(1/2) 
    E_mumax3 = gammapi3*(E_mu3 + (v3*p_mu3))
    E_mumin3 = gammapi3*(E_mu3 - (v3*p_mu3))
    
    #constants referring the initial decay of decay 4:
    E_pi4 = EB(ms,mpi_charged,mu) 
    p_pi4 = (E_pi4**2 - mpi_charged**2)**(1/2)
    v4 = p_pi4/E_pi4
    gammapi4 = E_pi4/mpi_charged
    Eu = ms-E_pi4 
    
    #constants referring to decay 4b:
    E_B4 = EB(mpi_charged,0,mu)
    E_B4max = gammapi4*E_B4*(1 + v4)
    E_B4min = gammapi4*E_B4*(1 - v4)
    
    #constants referring to decay 4c:
    E_mu4 = EB(mpi_charged,mu,0)
    p_mu4 = (E_mu4**2 - mu**2)**(1/2) 
    E_mumax4 = gammapi4*(E_mu4 + (v4*p_mu4))
    E_mumin4 = gammapi4*(E_mu4 - (v4*p_mu4))
    
    #@nb.jit(nopython=True)
    def C_ve(p_array, Tcm, T, f):
        C_array = p_array**n_model * (f - ca.f_eq(p_array, T, 0))
        return - A_model * ca.n_e(T) * Gf**2 * T**(2-n_model) * C_array
    
    def make_der(kk):
        @nb.jit(nopython=True)
        def derivatives(a,y): 
            d_array = np.zeros(len(y))
            Tcm = 1/a 

            dtda_part1 = mPL/(2*a)
            dtda_part3 = (y[-2]**4*np.pi**2)/15
            dtda_part4 = 2*y[-2]**4*calculate_integral(I1,me/y[-2])/np.pi**2
            dtda_part6 = ms*ns(Tcm,y[-1],ms,mixangle)
            dtda_part7 = (Tcm**4/(2*np.pi**2))*trapezoid(y[:int(num-3)]*e_array[:int(num-3)]**3,e_array[:int(num-3)])
            dtda = dtda_part1/(dtda_part2*(dtda_part3+dtda_part4+dtda_part6+dtda_part7))**.5
            d_array[-1] = dtda

            #df/da for the neutrinos and antineutrinos at epsilon = 0:
            d3b_e0 = 2*(1-eps_e**2)*d3r*gammapi3*(mu**2)*(Gf**2)*E_mu3*ns(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*Gam_mub)
            d4b_e0 = 2*(1-eps_e**2)*d4r*(Eu/mu)*(mu**2)*(Gf**2)*ns(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*Gam_mub)
            d4c_e0 = 2*(1-eps_e**2)*d4r*gammapi4*(mu**2)*(Gf**2)*E_mu4*ns(Tcm,y[-1],ms,mixangle)*dtda/(np.pi*Gam_mub)
            d_array[0] = d3b_e0+d4b_e0+d4c_e0

            if kk[0] == 0:
                c = coll.C(e_array*Tcm,y[:num-3]) 
            else:
                c = C_short(e_array*Tcm,y[:num-3],y[-2],kk) 
            c += C_ve(e_array*Tcm, Tcm, y[-2], y[:num-3])
            c *= dtda
            
            for i in range (1,num-3): 
                eps = e_array[i]
                coefficient = (2*np.pi**2)/(eps**2*Tcm**2*a**3)
                d1 = d1r*diracdelta((eps*Tcm),E_B1,i,e_array*Tcm)
                d2 = d2r*diracdelta((eps*Tcm),E_B2,i,e_array*Tcm)
                d3a = .5*d3r*diracdelta2(eps*Tcm,E_B3min,E_B3max,E_B3,gammapi3,v3,i,e_array*Tcm)
                d3b = (d3r/(2*gammapi3*v3*p_mu3*Gam_mub))*u_integral(E_mumin4,E_mumax4,eps*Tcm)
                Gam_mua = Gammamua((eps*Tcm)/(gammapi4*(1+v4)),min(Enumax,(eps*Tcm)/(gammapi4*(1-v4))))
                d4a = (d4r/(2*gammapi4*v4))*(Gam_mua/Gam_mub)
                d4b = .5*d4r*diracdelta2((eps*Tcm),E_B4min,E_B4max,E_B4,gammapi4,v4,i,e_array*Tcm)
                d4c = (d4r/(2*gammapi4*v4*p_mu4*Gam_mub))*u_integral(E_mumin4,E_mumax4,eps*Tcm)
                
                d_array[i] = coefficient*(d1+d2+d3a+d3b+d4a+d4b+d4c)*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda + c[i] #neutrinos only, antineutrinos not included

            df_array = d_array[:-3]*e_array**3/(2*np.pi**2) 
            dQda_part1 = ms*ns(Tcm,y[-1],ms,mixangle)*a**3*dtda/ts(ms,mixangle)
            dQda_part2 = Tcm**4*a**3*trapezoid(df_array,e_array)
            dQda = dQda_part1-dQda_part2
            d_array[-3] = dQda

            dTda_constant1 = (4*np.pi**2/45)+(2/np.pi**2)*(calculate_integral(I1,me/y[-2]) + (1/3)*calculate_integral(I2,me/y[-2]))
            dTda_constant2 = 2*me*y[-2]*a**3/(np.pi**2)
            dTda_numerator1 = -3*a**2*y[-2]**3*dTda_constant1
            dTda_numerator2 = dQda/y[-2]
            dTda_denominator = (3*y[-2]**2*a**3*dTda_constant1) - (dTda_constant2*(calculate_integral(dI1,me/y[-2]) - (1/3)*calculate_integral(dI2,me/y[-2])))
            dTda = (dTda_numerator1 + dTda_numerator2)/dTda_denominator
            d_array[-2] = dTda

            return d_array
        return derivatives
    
    a0 = a_init
    y0 = y_init
    
    f_FULL = 2 * np.pi**2 * ns(1/a0, y_init[-1], ms, mixangle) / ( 1/a0 * E_B2**2 * (e_array[-1]-e_array[-2]) )
    decay_on = True
    if f_FULL < f_BUFFER:
        decay_on = False
    a_max = a0 * a_MAXMULT
    if decay_on:
        a_max = min(e_array[-2*MIN_eps_BUFFER-1] / E_B1, a_max)
    print("f_FULL = ", f_FULL, "a0 = ", a0, "a_max = ", a_max)
    
    a_results = []
    y_results = []
    a_results.append(a0)
    y_results.append(y0)
    
    for i in range(N_steps):
        if first:
            k = [0,0,0]
        else:
            E2_index = np.where(e_array < E_B2 * a0)[0][-1]
            E1_index = np.where(e_array < E_B1 * a0)[0][-1]
            k = find_breaks(y0[:num-3], E2_index, E1_index)
        
        print(e_array[k[0]], e_array[k[1]], e_array[k[2]], e_array[-1] - e_array[-2])
        
        #try:
        #    func = make_der(np.array(k))
        #    dyda = func(a0,y0)
        #    y_test_step = 0.9 * y0 + dyda * dx
        #    if np.min(y_test_step[:-3]) < 0:
        #        too_small_ind = np.argmin(y_test_step[:-3])
        #        dx_pre = dx
        #        dx = - 0.9 * y0[too_small_ind] / dyda[too_small_ind]
        #        print("Step too big.")
        #        fn_step = "{}-{}-stepdata-{}".format(ms,mixangle,step_counter_log)
        #        np.savez(fn_step,ms=ms,mixangle=mixangle,y0=y0,e_array=e_array, eps_small=eps_small, eps_buffer=eps_buffer, dx_pre=dx_pre, dyda=dyda, new_step=dx)
        #        step_counter_log += 1
        #    a_array, y_matrix, dx_new = DES.destination_x_dx(func, y0,1, dN_steps, a0, a_max, dx)
        #except:
        #    np.savez('CRASHfile',ms=ms,mixangle=mixangle,a_init=a_init,y_init=y_init, e_array=e_array, eps_small=eps_small, eps_buffer=eps_buffer, dx_init=dx_init,i=i,k=np.array(k),y0=y0,a0=a0,a_max=a_max,dx=dx)
        #    raise
            
        a_array, y_matrix, dx_new = DES.destination_x_dx(make_der(np.array(k)), y0, N_steps, dN_steps, a0, a_max, dx)
        
        a0 = a_array[-1]
        y0 = np.zeros(len(y_matrix))
        for j in range(len(y0)):
            y0[j] = y_matrix[j][-1]
        dx = dx_new
        print(i+1, a0, dx)
        
        if a0 == a_results[-1]:
            break
            
        a_results.append(a0)
        y_results.append(y0)
        
        if a0 >= a_max:
            print(a_results[0],a_array[-1], a_max)
            break
        
        eps = np.where(y0[:-3] > f_SMALL)[0][-1] 
        if eps - eps_small > (eps_buffer - eps_small) / 2:
            break
        
        if y0[-2] < temp_fin:
            break
        
        #if intsave:
        #    ysave = np.array(y_results)
        #    np.save(int_save_file+'-a', a_results)
        #    np.save(int_save_file+'-y', np.transpose(ysave))
            
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
    return a_results, y_results, dx, decay_on


# In[9]:


def forward(ms, y_v, e_array, a, decay_on):
    eps_small_new = np.where(y_v[:-3] > f_SMALL)[0][-1]
            
    xp = np.zeros(2)
    yp = np.zeros(2)
    j = 0
    while (yp[1] >= yp[0]):
        if eps_small_new + j >= len(e_array):
            break
        else:
            for i in range(2):
                yp[i] = y_v[eps_small_new + j + i - 1]
                xp[i] = e_array[eps_small_new + j + i - 1]
            j += 1
    
    if eps_small_new + j == len(e_array):
        print("f is increasing at last box?")
        return y_v, e_array
    
    #eps_small_new = eps_small_new + j
    bxsz = abs(xp[1] - xp[0])
    new_len = len(y_v)
    
    
    
    
    for i in range(20*eps_small_new):
        if interp.log_linear_extrap(xp[0] + i * bxsz, xp, yp) > f_BUFFER:
            new_len = i + eps_small_new
        else:
            break
    new_len = max(eps_small_new + MIN_eps_BUFFER, new_len + 1) + 1
    
    if decay_on:
        e_up = np.where(e_array <  0.5 * ms * a )[0][-1]
        if e_up == len(e_array) - 1:
            e_up = int( 0.5 * ms * a / (e_array[-1] - e_array[-2]))
            e_up += MIN_eps_BUFFER * 2
        new_len = max(new_len, e_up)    
        e_test = e_array[eps_small_new-1] + (new_len - (MIN_eps_BUFFER + 1) - eps_small_new) * bxsz
        while e_test / a <= 0.5 * ms:
            new_len += MIN_eps_BUFFER
            e_test = e_array[eps_small_new-1] + (new_len - (MIN_eps_BUFFER + 1) - eps_small_new) * bxsz
    
    #if y_v[eps_small_new] < f_BUFFER:
    #    new_len = eps_small_new
    #else:
    #    if decay_on:
    #        e_up = np.where(e_array < 0.5 * mH * a)[0][-1]
    #        if e_up > len(e_array) - 2 * MIN_eps_BUFFER:
    #            e_up += MIN_eps_BUFFER * 3
    #        e_test = e_array[eps_small_new-1] + (new_len - (MIN_eps_BUFFER + 1) - eps_small_new) * bxsz
    #        while (e_test - 3*MIN_eps_BUFFER * bxsz)/a <= 0.5 * mH:
    #            e_up += MIN_eps_BUFFER
    #            e_test = e_array[eps_small_new-1] + (e_up - (MIN_eps_BUFFER + 1) - eps_small_new) * bxsz
    #        
    #        e_temp = max(eps_small_new, e_up)
    #    else:
    #        e_temp = eps_small_new
    #        
    #    if e_temp > len(y_v) - 5:
    #        new_len = e_temp
    #        
    #        eps_small_new = len(e_array)
    #        yp[0] = y_v[-5]
    #        yp[1] = y_v[-4]
    #        xp[0] = e_array[-2]
    #        xp[1] = e_array[-1]
    #    else:
    #        if y_v[e_temp] < f_BUFFER:
    #            new_len = e_temp
    #            eps_small_new = e_temp
    #        else:
    #            new_len = e_temp
    #            
    #            yp[0] = y_v[e_temp]
    #            yp[1] = y_v[e_temp+1]
    #            xp[0] = e_array[e_temp]
    #            xp[1] = e_array[e_temp+1]
    #            e_extrap = e_temp
    #
    #            while e_temp < len(y_v) - 5:
    #                if y_v[e_temp] < f_BUFFER:
    #                    break
    #                if y_v[e_temp+1] < y_v[e_temp] and yp[1]/yp[0] > y_v[e_temp+1]/y_v[e_temp]:
    #                    yp[0] = y_v[e_temp]
    #                    yp[1] = y_v[e_temp+1]
    #                    xp[0] = e_array[e_temp]
    #                    xp[1] = e_array[e_temp+1]
    #                    e_extrap = e_temp
    #                e_temp += 2
    #            
    #            for i in range(20*eps_small_new):
    #                if interp.log_linear_extrap(xp[0] + i * bxsz, xp, yp) > f_BUFFER:
    #                    new_len = i + e_extrap
    #                else:
    #                    break
    #                
    #            eps_small_new = e_extrap
    
    
    
    
    y = np.zeros(new_len+3)
    y[-1] = y_v[-1]
    y[-2] = y_v[-2]
    y[-3] = y_v[-3]
    
    eps_array = np.zeros(new_len)
    for i in range(eps_small_new):
        eps_array[i] = e_array[i]
        y[i] = y_v[i]
    for i in range(eps_small_new, len(eps_array)):
        eps_array[i] = e_array[eps_small_new-1] + (i+1 - eps_small_new) * bxsz
        y[i] = interp.log_linear_extrap(eps_array[i], xp, yp)
    #if len(eps_array) > eps_small_new:
    #    for i in range(eps_small_new, len(eps_array)):
    #        eps_array[i] = e_array[eps_small_new-1] + (i+1 - eps_small_new) * bxsz
    #        y[i] = interp.log_linear_extrap(eps_array[i], xp, yp)
    
    return y, eps_array   
    
def next_step(ms,mixangle,a,y,dx,e_array,eps_small,eps_buffer,Ns,dNs,fn,nr,plot=True,first=False,temp_fin=0):
    st = time.time()
    
    a_v, y_v, dx, decay_on = driver(ms, mixangle, a, y, e_array, eps_small, eps_buffer, dx, N_steps = Ns, dN_steps = dNs, first=first, temp_fin=temp_fin)
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
    #np.savez(temp_fn+'-inputs',mH=mH,mixangle=mixangle,a=a,y=y,dx=dx, e_array=e_array, eps_small=eps_small, eps_buffer=eps_buffer)
    
    a = a_v[-1]
    y, eps_array = forward(ms, y_v[-1], e_array, a, decay_on)

    eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
    eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
    
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


# In[10]:


def det_new_ics(a_i, T_i, f_SMALL, f_BUFFER, MIN_eps_BUFFER):
    a = a_i
    eps_small = - int(np.log(f_SMALL)/initial_boxsize)
    eps_buffer = max(eps_small + MIN_eps_BUFFER, - int(np.log(f_BUFFER)/initial_boxsize))

    initial_size = int((eps_buffer * initial_boxsize - eps_small_box)/initial_boxsize) + num_small_boxes
    y = np.zeros(initial_size + 3)
    eps_arr = np.zeros(initial_size)
    for i in range(num_small_boxes):
        eps_arr[i] = i * small_boxsize
        y[i] = 1 / (np.exp(eps_arr[i])+1)
    for i in range(num_small_boxes, len(eps_arr)):
        eps_arr[i] = eps_arr[num_small_boxes-1] + (i+1-num_small_boxes) * initial_boxsize
        y[i] = 1 / (np.exp(eps_arr[i])+1)
    y[-2] = T_i

    return a, y, eps_arr, eps_small, eps_buffer

def simple_spread_eps(y,eps_array):
    old_boxsize = eps_array[num_small_boxes+1] - eps_array[num_small_boxes]
    new_boxsize = 2 * old_boxsize
    new_length = int((len(eps_array) - num_small_boxes)/2) + num_small_boxes
    y_new = np.zeros(new_length+3)
    eps_arr = np.zeros(new_length)
    y_new[-1] = y[-1]
    y_new[-2] = y[-2]
    y_new[-3] = y[-3]
    y_new[0] = y[0]
    for i in range(1, num_small_boxes):
        eps_arr[i] = eps_array[i]
        y_new[i] = y[i]
    for i in range(num_small_boxes, new_length):
        eps_arr[i] = eps_arr[num_small_boxes-1] + (i+1 - num_small_boxes) * new_boxsize
        y_new[i] = max(y[num_small_boxes + 2*(i-num_small_boxes) + 1],y[num_small_boxes + 2*(i-num_small_boxes)])
    return y_new, eps_arr

def babysteps(ms,mixangle,N_runs,Ns,dNs,filename,prev_run=False,prev_run_file="",prev_i=-1,temp_fin=0.001):
    if prev_run:
        a_g = np.load(prev_run_file + '-a.npy')
        y_g = np.load(prev_run_file + '-y.npy')
        a = a_g[prev_i]
        y = np.zeros(len(y_g)) 
        for i in range(len(y)):
            y[i] = y_g[i][prev_i]
        try:
            eps_arr = np.load(prev_run_file + '-e.npy')
        except:
            print("No file {} found, using boxsize = 1".format(prev_run_file + '-e.npy'))
            eps_arr = np.linspace(0,(len(y)-4)*1,len(y)-3)
        dx = 1e-7
    else:
        a, y, eps_arr, eps_small, eps_buffer = det_new_ics(1.0/15, 15.0, f_SMALL, f_BUFFER, MIN_eps_BUFFER)
        
        print("Initial Run. Tcm = {:.3f} MeV, eps_s = {}, eps_b = {}".format(1/a,eps_small,eps_buffer))
        a, y, dx, eps_arr, eps_small, eps_buffer = next_step(ms,mixangle,a,y,1e-7,eps_arr,eps_small,eps_buffer,1,100,filename,0,plot=True,first=True)
        
    eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
    #try:
    #    eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
    #except:
    #    eps_buffer = len(y) - 4
    
    if eps_small > 500 or len(y) > 1000:
        y, eps_arr = simple_spread_eps(y, eps_arr)
    
    eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
    try:
        eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
    except:
        eps_buffer = len(y) - 4

    tau = ts(ms,mixangle)

    for nr in range(1,N_runs):
        if y[-2] > 0.01:
            print("Run {} of {}. Tcm = {:.3f} MeV, T= {:.3f} MeV, t/tau = {:.3f}, eps_s = {}, eps_b = {}".format(nr+1,N_runs,1/a,y[-2],y[-1]/tau,eps_arr[eps_small],eps_arr[eps_buffer]))
        else:
            print("Run {} of {}. Tcm = {:.3f} keV, T= {:.3f} keV, t/tau = {:.3f}, eps_s = {}, eps_b = {}".format(nr+1,N_runs,1/a*1000,y[-2]*1000,y[-1]/tau,eps_arr[eps_small],eps_arr[eps_buffer]))
        a, y, dx, eps_arr, eps_small, eps_buffer = next_step(ms,mixangle,a,y,dx,eps_arr,eps_small,eps_buffer,Ns,dNs,filename,nr,plot=True,temp_fin=temp_fin)
        if eps_small > 500 or len(y) > 1000:
            y, eps_arr = simple_spread_eps(y, eps_arr)
        eps_small = np.where(y[:-3] > f_SMALL)[0][-1]
        try:
            eps_buffer = np.where(y[:-3] > f_BUFFER)[0][-1]
        except:
            eps_buffer = len(y) - 4

        if y[-2] < temp_fin:
            break


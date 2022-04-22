#!/bin/pyhton
import numpy as np
import math
import random
from numpy.fft import *
#initial parameters
Nx,Ny=64,64
dx,dy=1.0,1.0
nstep=20000
dt=0.01
c0=0.40
coefA=1.0
mobility=1.0
grad_coef=0.5
#initial con
def con_initial(Nx,Ny,c0):
    con=np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            conij=c0+random.uniform(-0.01,0.01)
            con[i][j]=conij
    return con
#prepare fft
def kspace(Nx,Ny,dx,dy):
    k2,k4=np.zeros((Nx,Ny)),np.zeros((Nx,Ny))
    for i in range(1,Nx+1):
        dkx=(2*math.pi)/(Nx*dx)
        if i<=Nx/2:
            value_kx=i*dkx
        else:
            value_kx=-(Nx-i)*dkx
        for j in range(Ny):
            dky=(2*math.pi)/(Ny*dy)
            if j<=Ny/2:
                value_ky=j*dky
            else:
                value_ky=-(Ny-j)*dky
            value_k2=value_kx**2+value_ky**2
            value_k4=value_k2**2
            k2[i-1][j-1]=value_k2
            k4[i-1][j-1]=value_k4
    return k2,k4
#part free energy to fft
def free_energy_fft(con):
    #2Ac*(1-c)*(1-2c)
    con_f=np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            con_f[i][j]=2*coefA*con[i][j]*(1-con[i][j])*(1-2*con[i][j])
    fcon_f=fft2(con_f)
    return fcon_f
#run
con=con_initial(Nx,Ny,c0)
k2,k4=kspace(Nx,Ny,dx,dy)
for i in range(nstep):
    fcon=fft2(con)
    fcon_f=free_energy_fft(con)
    for j in range(Nx):
        for k in range(Ny):
            numer=dt*mobility*k2[j][k]*fcon_f[j][k]
            denom=1.0+dt*mobility*grad_coef*k4[j][k]
            fcon[j][k]=(fcon[j][k]-numer)/denom
    con=ifft2(fcon).real
    for j in range(Nx):
        for k in range(Ny):
            if con[j][k]>=0.9999:
                con[j][k]=0.9999
            elif con[j][k]<0.00001:
                con[j][k]=0.00001
o=open('phase.dat','w')
for i in range(Nx):
    for j in range(Ny):
        o.write('%i  %i  %8.6f\n' % (i,j,con[i][j]))
    o.write('\n')
o.close()

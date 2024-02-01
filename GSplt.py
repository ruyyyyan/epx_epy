import math
import numpy as np
from scipy.sparse.linalg import inv
#from numpy.linalg import inv
import scipy.sparse as sps
import scipy.sparse.linalg
import sys
import matplotlib.pyplot as plt
sys.path.append('../../src/')
from pylab import *

ed = 0
tpd = 1.5
tpp = 0.55

ep = 3.5
pds = 1.5
pdp = 0.7
#pds = 0.00001
#pdp = 0.00001
pps = 0.9
ppp = 0.2

A = 6.0
B = 0.15
C = 0.58

difs = np.arange(0.00, 0.051, 1) 
Upps = np.arange(4.0, 4.01, 0.2)                        #4.0     3.0     2.0(8.06)
Vpps = np.arange(0.0, 0.01, 0.2)                        #0-2.0   0-1.4   0-1.0 
Udifs = np.arange(0.2, 0.21, 1)
# Upps = [0.0]
# Vpps = [0.0]
Norb = 7
Mc = 10
eta = 0.2

for Upp in Upps:
    Upp = round(Upp,1)
    fname = 'CP_Norb'+str(Norb)+'_Upp'+str(Upp)+'.txt'
    for Vpp in Vpps:
        Vpp = round(Vpp,1)
        for diff in difs:
            for Udif in Udifs:
                clf()
                print('############################################\ndiff=',diff)
                titlename = 'tpp'+str(tpp)+'_B'+str(B)+'_C'+str(C)+'_Upp'+str(Upp)+'_Mc'+str(Mc)   

                if Norb==7:
                    ff = 'GS_weights_Norb7_tpp'+str(tpp)+'_diff'+str(diff)+'_Upp'+str(Upp) \
                         +'_Vpp'+str(Vpp)+'_Mc'+str(Mc)+'_eta'+str(eta)+'.txt'
                if Norb==9 or Norb==11:
                    ff = 'GS_weights_'+'Norb'+str(Norb)+'_pps'+str(pps)+'_ppp'+str(ppp)+'_diff'+str(diff)+\
                         '_Upp'+str(Upp)+'_Vpp'+str(Vpp)+'_Mc'+str(Mc)+'_eta'+str(eta)+'.txt'

                a = loadtxt('./data_GS/'+ff,skiprows=0)
                if Norb==9 or Norb==11:
                    a = np.delete(a, 3, axis=1)
                for i in range(0,len(a)):
                    if abs(a[i,7]-a[i-1,7])>0.3:
                        cp = round((a[i-1,1]+a[i-1,2])*0.5+0.05,2)
                        idx = i-1
                aa = [cp, cp]
                bb = [0, 0.85]        

                #plot 1A1 symmetry       
                #d8_orb
                plot((a[0:idx,1]+a[0:idx,2])*0.5, a[0:idx,4], '-bo',markersize='5',label="$a_{1}a_{1}$")
                plot((a[0:idx,1]+a[0:idx,2])*0.5, a[0:idx,5], '-rs',markersize='5',label="$b_{1}b_{1}$")

                #d9_orb
                plot((a[0:idx,1]+a[0:idx,2])*0.5, a[0:idx,11], '-c>',markersize='5',label="$b_{1}L_x$")
                plot((a[0:idx,1]+a[0:idx,2])*0.5, a[0:idx,17], '-g^',markersize='5',label="$b_{1}L_y$")

                #d10_orb
                #plot((a[0:idx,1]+a[0:idx,2])*0.5, (a[0:idx,21]), '--yo',markersize='5',label="$d^{10}L_xL_x$")
                plot((a[0:idx,1]+a[0:idx,2])*0.5, (a[0:idx,22]), 'grey',markersize='5',marker='v',label="$d^{10}L_xL_y$")
                plot((a[0:idx,1]+a[0:idx,2])*0.5, (a[0:idx,23]), 'lawngreen',markersize='5',marker='s',label="$d^{10}L_yL_y$")

                #plot 3B1 symmetry       
                #d8_orb 
                plot((a[idx+1:,1]+a[idx+1:,2])*0.5, a[idx+1:,7], '-kp',markersize='5',label="$a_{1}b_{1}$")

                #d9_orb
                plot((a[idx+1:,1]+a[idx+1:,2])*0.5, a[idx+1:,9], 'orange',markersize='5',marker='o',label="$a_{1}L_x$")
                plot((a[idx+1:,1]+a[idx+1:,2])*0.5, a[idx+1:,15], '-mv',markersize='5',label="$a_{1}L_y$")
                plot((a[idx+1:,1]+a[idx+1:,2])*0.5, a[idx+1:,11], '-c>',markersize='5')
                plot((a[idx+1:,1]+a[idx+1:,2])*0.5, a[idx+1:,17], '-g^',markersize='5')

                #d10_orb
                plot((a[idx+1:,1]+a[idx+1:,2])*0.5, (a[idx+1:,23]), 'lawngreen',markersize='5',marker='s')
                for i in range(0,len(a)):
                    print ('Delta=', round((a[i,1]+a[i,2])*0.5,2),', total weight = ', sum(a[i,4:23]))



                plot(aa, bb, '-')


                if Norb==7:
                    title('Norb7'+'_tpd'+str(tpd)+'_tpp'+str(tpp)+'_diff'+str(diff)+'_Upp'+str(Upp)+'_Vpp'+str(Vpp)+\
                          '_B'+str(B)+'_C'+str(C)+'_Mc'+str(Mc)+'_CP'+str(cp), fontsize=8)
                elif Norb==9 or Norb==11:
                    title('Norb'+str(Norb)+'_pds1.5_pdp0.7_ep3.5_pps'+str(pps)+'_ppp'+str(ppp)+'_diff'+str(diff)+\
                          '_Upp'+str(Upp)+'_Vpp'+str(Vpp)+'_B'+str(B)+'_C'+str(C)+'_Mc'+str(Mc)+'_CP'+str(cp), fontsize=8)
                xlabel('$\Delta$',fontsize=15)
                ylabel('weight',fontsize=15)
                text(9, 0.8, '$^3\!B_1$',fontsize=15)
                text(2, 0.8, '$^1\!A_1$',fontsize=15)
                #xlim([0,3])
                #ylim([-6,10])
                #grid('on',linestyle="--", linewidth=0.5, color='black', alpha=0.5)
                legend(loc='best', fontsize=8.5, framealpha=1.0, edgecolor='black')
                if Norb==7:
                    savefig("GS_components_Norb7_tpd"+str(tpd)+'_A'+str(A)+'_diff'+str(diff)+'_Upp'+str(Upp)+'_Vpp'+str(Vpp)+".pdf")
                elif Norb==9 or Norb==11:
                    savefig("GS_components_Norb"+str(Norb)+'_diff'+str(diff)+"_pps"+str(pps)+'_ppp'+str(ppp)+'_ep'+str(ep)+'_Upp'+str(Upp)+\
                            '_Vpp'+str(Vpp)+".pdf")
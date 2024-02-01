import math
import numpy as np
from scipy.sparse.linalg import inv
#from numpy.linalg import inv
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy import integrate
import sys
import matplotlib.pyplot as plt
sys.path.append('../../src/')
from pylab import *

import parameters as pam
import lattice as lat
import variational_space as vs
import hamiltonian as ham
import basis_change as basis
import get_state as getstate
import utility as util
import plotfig as fig
import lanczos
import time
start_time = time.time()
M_PI = math.pi

Ms = ['-b','-r','-g','-m','-c','-k','-y','--b','--r','--g','--m','--c','--k','--y',\
      '-.b','-.r','-.g','-.m','-.c','-.k','-.y',':b',':r',':g',':m',':c',':k',':y']

def set_Phi(VS, ind):
    dim = VS.dim
    Phi = np.zeros(dim, dtype = complex)
    Phi[ind] = 1.0
    return Phi


def compute_Aw(H, VS, w_vals, state_index, label_index, fig_name, fname):
    '''
    plot state one by one with each state has different label
    '''
    print ('compute ', fig_name)
    clf()
    
    Aw = np.zeros(len(w_vals))
    Nstate = len(state_index)
    for j in range(0,Nstate):
        index = state_index[j]
        
        Aw_tmp = np.zeros(len(w_vals))     
        Aw_tmp, w_peak, weight = getAw(H,index,VS,w_vals)
                
        plt.plot(w_vals, Aw_tmp, Ms[j], linewidth=1, label=label_index[j])  

    # write data into file for reusage
    if pam.if_write_Aw==1:
        util.write_Aw(fig_name+fname+'.txt', Aw, w_vals)

    if fig_name=="Aw_d9_":
        xlim([-8,8])
        
    #xlim([0,15])
    #ylim([0,maxval])
    #ylim([0,0.5])
    #text(0.45, 0.1, '(a)', fontsize=16)
    grid('on',linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    legend(loc='best', fontsize=10, framealpha=1.0, edgecolor='black')
    title(fname, fontsize=8)
    xlabel('$\omega$',fontsize=15)
    ylabel('$A(\omega)$',fontsize=15)

    savefig(fig_name+fname+".pdf")
    print ("====================================")
    
def compute_Aw1(H, VS, w_vals, state_index, label_index, fig_name, fname):
    '''
    Added spectra for a set of states which share the same label
    '''
    print ('compute ', fig_name)
    clf()
    
    Aw = np.zeros(len(w_vals))
    Nstate = len(state_index)
    for j in range(0,Nstate):
        index = state_index[j]
        
        Aw_tmp = np.zeros(len(w_vals))     
        Aw_tmp, w_peak, weight = getAw(H,index,VS,w_vals)
        Aw += Aw_tmp
                
    plt.plot(w_vals, Aw, Ms[0], linewidth=1, label=label_index[0])  

    # write data into file for reusage
    if pam.if_write_Aw==1:
        util.write_Aw(fig_name+fname+'.txt', Aw, w_vals)

    if fig_name=="Aw_d9_":
        xlim([-8,8])
        
    #xlim([0,15])
    #ylim([0,maxval])
    #ylim([0,0.5])
    #text(0.45, 0.1, '(a)', fontsize=16)
    grid('on',linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    legend(loc='best', fontsize=10, framealpha=1.0, edgecolor='black')
    title(fname, fontsize=8)
    xlabel('$\omega$',fontsize=15)
    ylabel('$A(\omega)$',fontsize=15)

    savefig(fig_name+fname+".pdf")
    print ("====================================")
    
####################################################################################
def compute_Aw_d8_sym(H, VS, d_double, S_val, Sz_val, AorB_sym, A, w_vals, fig_name, fname):
    '''
    Compute A(w) for d8 states
    '''
    Aw_dd_total = np.zeros(len(w_vals))
    symmetries = pam.symmetries    
    Nsym = len(symmetries)
    for i in range(0,Nsym):
        sym = symmetries[i]
        print ("====================================")
        print ("start computing A_dd(w) for sym", sym)

        d8_state_indices = getstate.get_d8_state_indices_sym(VS,sym,d_double,S_val, Sz_val, AorB_sym, A)

        Aw_dd  = np.zeros(len(w_vals))
        Aw_dd1 = np.zeros(len(w_vals))
        
        for index in d8_state_indices:   
            '''method 1'''
            # use standard Lanczos to diagonalize the tridiagonal matrix
            Phi0 = set_Phi(VS, index)
            Aw, w_peak, weight = getAw(H, Phi0, VS, w_vals)
            Aw_dd += Aw  #*coef_frac_parentage[spinorb]
            #Aw_dGS += wgh_d*Aw
            
            '''method 2'''
#             # use Mona's generalized method designed for off-diagonal <w0|G|v0> 
#             # and set w0=v0 as a special case 
#             # use continued fraction with an, bn instead of diagonalizing the tridiagonal matrix
#             if pam.if_compute_Aw_with_Ffg==1:
#                 print('\n======================================================')
#                 print('Compute Aw via An and fn following Mona notes\n')

#                 F_gg = compute_F_fgs(H, VS, w_vals, Phi0, Phi0)
#                 Aw1 = -1.0/M_PI * np.imag(F_gg)    
#                 Aw_dd1 += Aw1
                
#                 Phi1 = set_Phi(VS, 20)
#                 assert(index!=0)
#                 F_fgs = compute_F_fgs(H, VS, w_vals, Phi0, Phi1)
#                 print('max F_fgs',max(F_fgs))
#                 F_gsf = compute_F_fgs(H, VS, w_vals, Phi1, Phi0)
#                 print('max F_gsf',max(F_gsf))
                
# #                 plt.plot(w_vals, F_fgs, Ms[i], linewidth=1, label=sym)
# #                 plt.plot(w_vals, F_gsf, Ms[i+1], '--',linewidth=1, label=sym)


            # write lowest peak data into file
            if pam.if_find_lowpeak==1 and pam.if_write_lowpeak_ep_tpd==1:
                if Norb==7:
                    write_lowpeak(flowpeak+'_'+sym+'.txt',A,ep,tpd,w_peak, weight)
                elif Norb==9:
                    write_lowpeak2(flowpeak+'_'+sym+'.txt',A,ep,pds,pdp,w_peak, weight)

        # write data into file for reusage
        if pam.if_write_Aw==1:
            #util.write_Aw(fname+'_'+sym+'.txt', Aw_dd, vals[0]-w_vals)
            util.write_Aw(fname+'_'+sym+'.txt', Aw_dd, w_vals)

        # accumulate Aw for each sym into total Aw_dd
        Aw_dd_total += Aw_dd

        subplot(Nsym,1,i+1)
        plt.plot(w_vals, Aw_dd, Ms[i], linewidth=1, label=sym)
        #plt.plot(w_vals, Aw_dGS, Ms[i], linewidth=1, label=sym)

        #if pam.if_compute_Aw_with_Ffg==1:
            #plt.plot(w_vals, Aw_dd1, '--b', linewidth=1, label='Mona new Lanczos')
        
        # plot atomic multiplet peaks
        #plot_atomic_multiplet_peaks(Aw_dd)

        if i==0:
            title(fname, fontsize=8)
        if i==Nsym-1:
            xlabel('$\omega$',fontsize=15)

        maxval = max(Aw_dd)
        #xlim([-5,20])
        ylim([0,maxval])
        ylim([0,0.1])
        #ylabel('$A(\omega)$',fontsize=17)
        #text(0.45, 0.1, '(a)', fontsize=16)
        #grid('on',linestyle="--", linewidth=0.5, color='black', alpha=0.5)
        legend(loc='best', fontsize=9.5, framealpha=1.0, edgecolor='black')
        #yticks(fontsize=12) 

    if Nsym>0 and pam.if_savefig_Aw==1:    
        savefig("Aw_d8_"+fname+"_sym.pdf")

    ############################################################
    # plot total A(w) for d8
    if pam.if_compute_Aw_dd_total == 1:
        clf()
        plt.plot(w_vals, Aw_dd_total,'-b', linewidth=1)
        title(fname, fontsize=8)
        maxval = max(Aw_dd_total)
        xlim([-5,0])
        ylim([0,maxval])
        xlabel('$\omega$',fontsize=17)
        ylabel('$A(\omega)$',fontsize=17)
        #text(0.45, 0.1, '(a)', fontsize=16)
        #grid('on',linestyle="--", linewidth=0.5, color='black', alpha=0.5)
        legend(loc='best', fontsize=9.5, framealpha=1.0, edgecolor='black')

        # plot atomic multiplet peaks
        plot_atomic_multiplet_peaks(Aw_dd_total)

        savefig("Aw_dd_"+fname+"_total.pdf")
            
#################################################################################
def getAw(matrix, Phi0, VS, w_vals):  
    # set up Lanczos solver
    dim = VS.dim
    scratch = np.empty(dim, dtype = complex)
    solver = lanczos.LanczosSolver(maxiter = pam.Lanczos_maxiter, 
                                   precision = 1e-12, 
                                   cond = 'UPTOMAX', 
                                   eps = 1e-8)
    
    # need copy Phi0 to x0, otherwise x0=Phi0 is for pointer so that Phi0 will be modified
    x0 = np.copy(Phi0)
    
    solver.first_pass(x0 = x0, scratch = scratch, H = matrix)
    V, D = solver.lanczos_diag_T()

    # D[0,:] is the eigenvector for lowest eigenvalue
    tab = np.abs(D[0,:])**2

    Aw = np.zeros(len(w_vals))
    weight = 0
    for n in range(len(V)):
        Aw += tab[n] * pam.eta / M_PI * ( (w_vals - V[n])**2 + pam.eta**2)**(-1)
        
    if pam.if_find_lowpeak==1:
        if pam.peak_mode=='highest_peak':
            w_peak = getAw_peak_highest(Aw, w_vals, D, tab)
        elif pam.peak_mode=='lowest_peak':
            w_peak = getAw_peak_lowest(Aw, w_vals, D, tab)
        elif pam.peak_mode=='lowest_peak_intensity':
            w_peak = getAw_peak_lowest_intensity(Aw, w_vals, D, tab)
    else:
        w_peak = 0; weight = 0
        
    return Aw, w_peak, weight


#################################################################################
def compute_F_fgs(matrix, VS, w_vals, Phi0, Phi1):
    '''
    <f|G|gs>, where gs denotes GS of N space
    See Koch's notes on computing off-diagonal G
    '''
    # set up Lanczos solver
    dim = VS.dim
    scratch = np.empty(dim, dtype = complex)
    solver = lanczos.LanczosSolver(maxiter = pam.Lanczos_maxiter, 
                                   precision = 1e-12, 
                                   cond = 'UPTOMAX', 
                                   eps = 1e-8)
    
    # need copy Phi0 to x0, otherwise x0=Phi0 is for pointer so that Phi0 will be modified
    x0 = np.copy(Phi0)
    w0 = np.copy(Phi1)
    
    # run L Lanczos iterations to get a_n and b_n
    # for computing off-diagonal G, also need c_n = <w0|vn>. see Mona's notes
    an, bn, cn = solver.compute_anbncn(mode = 'FIRST', x0 = x0, w0 = w0, scratch = scratch, H = matrix)
    
    # use continued fraction for recursively calculating <w0|G|v0> = F0
    zvals = w_vals + pam.eta*1j
    A0, F0 = solver.getAnfn(zvals, 0, an, bn, cn)
    
    return F0


def getAw_peak_highest(Aw, w_vals, D, tab):  
    '''
    find the position and weight of highest peak of Aw, which might be lowest
    '''    
    w_idx = np.argmax(Aw)
    print ('highest peak index',w_idx)
    w_peak = w_vals[w_idx]
    print ('highest peak at w = ', w_peak)
    
    '''
    # find the area below the whole peak, namely the peak weight
    # ==========================================================
    # 1. first find the peak's w-range: [w_min, w_max]
    wid = w_idx
    while Aw[wid]>1.e-3:
        #print w_vals[wid], Aw[wid]
        if Aw[wid-1]>Aw[wid]:
            break
        wid -= 1
    w_min = wid
    
    wid = w_idx
    while Aw[wid]>1.e-3:
        #print w_vals[wid], Aw[wid]
        if Aw[wid+1]>Aw[wid]:
            break
        wid += 1
    w_max = wid
    
    print ('highest peak w-range = [', w_vals[w_min], w_vals[w_max], ']')
    
    # 2. Simpson's rule
    weight = integrate.simps(Aw[w_min:w_max], w_vals[w_min:w_max])
    print ('highest peak, weight = ', w_peak, '  ', weight)

    # find the eigenvalue D[n] nearest to w_peak so that its index n
    # leads to weight = tab[n]; Note that this weight is actually for 
    # the single peak instead of the area below the whole peak
    tmp = []
    for n in range(len(D)):
        tmp.append(abs(D[n]-w_peak))
        
    idx = tmp.index(min(tmp))
    weight = tab[idx]
    assert(weight>=0.0 and weight<=1.0)
    '''
    return w_peak #, weight

def getAw_peak_lowest(Aw, w_vals, D, tab):  
    '''
    find the position and weight of lowest peak of Aw, which might be highest
    '''    
    w_idx = 0
    # go through the regime with Aw=0 (numerically ~1.e-6)
    while Aw[w_idx]<1.e-3:
        w_idx += 1
    #print 'Aw < 1.e-3 until ', w_vals[w_idx]

    # go up until the peak:
    while Aw[w_idx+1]>Aw[w_idx]:
        w_idx += 1
    w_peak = w_vals[w_idx]
    print ('lowest peak at w = ', w_peak)
    

    # find the area below the whole peak, namely the peak weight
    # ==========================================================
    # 1. first find the peak's w-range: [w_min, w_max]
    wid = w_idx
    while Aw[wid]>1.e-3:
        #print w_vals[wid], Aw[wid]
        if Aw[wid-1]>Aw[wid]:
            break
        wid -= 1
    w_min = wid
    
    wid = w_idx
    while Aw[wid]>1.e-3:
        #print w_vals[wid], Aw[wid]
        if Aw[wid+1]>Aw[wid]:
            break
        wid += 1
    w_max = wid
    
    #print 'lowest peak w-range = [', w_vals[w_min], w_vals[w_max], ']'
    
    # 2. Simpson's rule
    #weight = integrate.simps(Aw[w_min:w_max], w_vals[w_min:w_max])
    #print 'lowest peak, weight = ', w_peak, '  ', weight

    return w_peak #, weight

def getAw_peak_lowest_intensity(Aw, w_vals, D, tab):  
    w_idx = 0
    # go through the regime with Aw=0 (numerically ~1.e-6)
    while Aw[w_idx]<1.e-3:
        w_idx += 1
    print ('Aw < 1.e-3 until ', w_vals[w_idx])

    # go up until the peak:
    while Aw[w_idx+1]>Aw[w_idx]:
        w_idx += 1
    print ('lowest peak intensity = ', Aw[w_idx])
    
    return Aw[w_idx] 


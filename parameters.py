import math
import numpy as np
M_PI = math.pi

Mc = 10

# See George's email on Nov.4, 2020
# Need to tune ed to get single hole spectra peak
# to agree with experimental RIXS values
# These ed values are obtained in onehole_impurity_model
# following used to tune peaks consistent with RIXS results
ed = {'d3z2r2': 2.1,\
      'dx2y2' : 0,\
      'dxy'   : 0.6,\
      'dxz'   : 1.1,\
      'dyz'   : 1.1}
ed = {'d3z2r2': 0,\
      'dx2y2' : 0,\
      'dxy'   : 0,\
      'dxz'   : 0,\
      'dyz'   : 0}
ep_avgs = np.arange(3.5, 7.01, 3.5) 
ep_difs = np.arange(0.0, 0.101, 0.01) 
#for computing critical points and plotting precise figures
# ep_avgs1 = np.arange(0.0, 6.49, 0.5)
# ep_avgs2 = np.arange(6.5, 8.99, 0.1)
# ep_avgs3 = np.arange(9.00,12.01 , 0.5)
# ep_avgs=np.hstack((np.hstack((ep_avgs1, ep_avgs2)),ep_avgs3))

# ep_avgs=np.arange(7.9,8.21,0.02)
# ep_difs = np.arange(0.0, 0.051, 10) 

As = np.arange(6.0, 6.01, 1.0)
B = 0.15
C = 0.58
#As = np.arange(100, 100.1, 1.0)
#As = np.arange(0, 0.1, 1.0)
#B = 0
#C = 0

Upps = np.arange(0, 4.01, 10)
Vpps = np.arange(0.0, 0.01, 0.1)
Udifs = np.arange(0,1.01,10)

Upds = np.arange(0,0.51,10)
Updfs = np.arange(0,0.101,10)

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3z^2-r^2 
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 7
if Norb==3 or Norb==7:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    tpds = np.linspace(1.5, 1.5, num=1, endpoint=True) #[0.25]
    #tpds = [0.0001]
    tpddiffs = np.arange(-0.2, 0.2001, 0.04)
    tpps = [0.55]
elif Norb==9 or Norb==10 or Norb==11:    
    # pdp = sqrt(3)/4*pds so that tpd(b2)=tpd(b1)/2: see Eskes's thesis and 1990 paper
    # the values of pds and pdp between papers have factor of 2 difference
    # here use Eskes's thesis Page 4
    # also note that tpd ~ pds*sqrt(3)/2
    vals = np.linspace(1.3, 1.3, num=1, endpoint=True)
    #vals = np.linspace(0.001, 0.001, num=1, endpoint=True)
    pdss = np.asarray(vals)*2./np.sqrt(3)
    pdps = np.asarray(pdss)*np.sqrt(3)/4.
    #pdss = [1.5]
    #pdps = [0.7]
    #------------------------------------------------------------------------------
    # note that tpp ~ (pps+ppp)/2
    # because 3 or 7 orbital bandwidth is 8*tpp while 9 orbital has 4*(pps+ppp)
    pps = 0.9
    ppp = 0.2
    #pps = 0.001
    #ppp = 0.001

wmin = -10; wmax = 20
eta = 0.2
Lanczos_maxiter = 600

# restriction on variational space
VS_only_up_up = 0
VS_only_up_dn = 1

if_H0_rotate_byU = 1
basis_change_type = 'all_states' # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_find_lowpeak = 0
if if_find_lowpeak==1:
    peak_mode = 'lowest_peak' # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
    if_write_lowpeak_ep_tpd = 1
if_write_Aw = 1
if_savefig_Aw = 1

if_get_ground_state = 1
if if_get_ground_state==1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 10
    
if_compute_Aw_with_Ffg = 0   # Mona's new method for computing <w0|G|v0>
if_compute_Aw_dd_total = 0
if_compute_Aw_pp = 0
if_compute_Aw_dp = 0
if_compute_Aw_Cu_dx2y2_O = 0

Ni_orbs = ['dx2y2','dxy','dxz','dyz','d3z2r2']
#Ni_orbs = ['dx2y2','d3z2r2']
    
if Norb==7:
    O1_orbs  = ['px']
    O2_orbs  = ['py']
elif Norb==9:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
elif Norb==11:
    O1_orbs  = ['px1','py1','pz1']
    O2_orbs  = ['px2','py2','pz2']
O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()
print ("Ni_orbs = ", Ni_orbs)
print ("O1_orbs = ",  O1_orbs)
print ("O2_orbs = ",  O2_orbs)
orbs = Ni_orbs + O_orbs 
#assert(len(orbs)==Norb)

symmetries = ['1A1','1B1','3B1','1A2','3A2','1E','3E']
print ("compute A(w) for symmetries = ",symmetries)


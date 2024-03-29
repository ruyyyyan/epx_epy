'''
Functions for constructing individual parts of the Hamiltonian. The 
matrices still need to be multiplied with the appropriate coupling 
constants t_pd, t_pp, etc..
'''
import parameters as pam
import lattice as lat
import variational_space as vs 
import utility as util
import numpy as np
import scipy.sparse as sps

directions_to_vecs = {'UR': (1,1,0),\
                      'UL': (-1,1,0),\
                      'DL': (-1,-1,0),\
                      'DR': (1,-1,0),\
                      'L': (-1,0,0),\
                      'R': (1,0,0),\
                      'U': (0,1,0),\
                      'D': (0,-1,0),\
                      'T': (0,0,1),\
                      'B': (0,0,-1),\
                      'L2': (-2,0,0),\
                      'R2': (2,0,0),\
                      'U2': (0,2,0),\
                      'D2': (0,-2,0),\
                      'T2': (0,0,2),\
                      'B2': (0,0,-2),\
                      'pzL': (-1,0,1),\
                      'pzR': (1,0,1),\
                      'pzU': (0,1,1),\
                      'pzD': (0,-1,1),\
                      'mzL': (-1,0,-1),\
                      'mzR': (1,0,-1),\
                      'mzU': (0,1,-1),\
                      'mzD': (0,-1,-1)}
tpp_nn_hop_dir = ['UR','UL','DL','DR']

def set_tpd_tpp(Norb,tpd,tpddiff,tpp,pds,pdsdiff,pdp,pdpdiff,pps,ppp):
    # dxz and dyz has no tpd hopping
    if pam.Norb==7:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'px'    : ['L','R'],\
                          'py'    : ['U','D']}
    elif pam.Norb==9 or pam.Norb==10:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D']}
    elif pam.Norb==11:
        tpd_nn_hop_dir = {'d3z2r2': ['L','R','U','D'],\
                          'dx2y2' : ['L','R','U','D'],\
                          'dxy'   : ['L','R','U','D'],\
                          'dxz'   : ['L','R'],\
                          'dyz'   : ['U','D'],\
                          'px1'   : ['L','R'],\
                          'py1'   : ['L','R'],\
                          'pz1'   : ['L','R'],\
                          'px2'   : ['U','D'],\
                          'py2'   : ['U','D'],\
                          'pz2'   : ['U','D']}
        
    if pam.Norb==7:
        tpd_orbs = {'d3z2r2','dx2y2','px','py'}
    elif pam.Norb==9:
        tpd_orbs = {'d3z2r2','dx2y2','dxy','px1','py1','px2','py2'}
    elif pam.Norb==11:
        tpd_orbs = {'d3z2r2','dx2y2','dxy','dxz','dyz','px1','py1','pz1','px2','py2','pz2'}
        
    # hole language: sign convention followed from Fig 1 in H.Eskes's PRB 1990 paper
    #                or PRB 2016: Characterizing the three-orbital Hubbard model...
    # Or see George's email on Aug.19, 2021:
    # dx2-y2 hoping to the O px in the positive x direction should be positive for electrons and for O in the minus x
    # directions should be negative for electrons, i.e. the hoping integral should be minus sign of overlap integral
    # between two neighboring atoms. 
    if pam.Norb==7:
        # d3z2r2 has +,-,+ sign structure so that it is - in x-y plane
        # all tpdx>tpdy
        tpd_nn_hop_fac = {('d3z2r2','L','px'): -(tpd+tpddiff)/np.sqrt(3),\
                          ('d3z2r2','R','px'):  (tpd+tpddiff)/np.sqrt(3),\
                          ('d3z2r2','U','py'):  (tpd-tpddiff)/np.sqrt(3),\
                          ('d3z2r2','D','py'): -(tpd-tpddiff)/np.sqrt(3),\
                          ('dx2y2','L','px'):   (tpd+tpddiff),\
                          ('dx2y2','R','px'):  -(tpd+tpddiff),\
                          ('dx2y2','U','py'):   (tpd-tpddiff),\
                          ('dx2y2','D','py'):  -(tpd-tpddiff),\
                          # below just inverse dir of the above one by one
                          ('px','R','d3z2r2'): -(tpd+tpddiff)/np.sqrt(3),\
                          ('px','L','d3z2r2'):  (tpd+tpddiff)/np.sqrt(3),\
                          ('py','D','d3z2r2'):  (tpd-tpddiff)/np.sqrt(3),\
                          ('py','U','d3z2r2'): -(tpd-tpddiff)/np.sqrt(3),\
                          ('px','R','dx2y2'):   (tpd+tpddiff),\
                          ('px','L','dx2y2'):  -(tpd+tpddiff),\
                          ('py','D','dx2y2'):   (tpd-tpddiff),\
                          ('py','U','dx2y2'):  -(tpd-tpddiff)}
    elif pam.Norb==9:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'): -(pds+pdsdiff)/2.0,\
                          ('d3z2r2','R','px1'):  (pds+pdsdiff)/2.0,\
                          ('d3z2r2','U','py2'):  (pds-pdsdiff)/2.0,\
                          ('d3z2r2','D','py2'): -(pds-pdsdiff)/2.0,\
                          ('dx2y2','L','px1'):   (pds+pdsdiff)*c,\
                          ('dx2y2','R','px1'):  -(pds+pdsdiff)*c,\
                          ('dx2y2','U','py2'):   (pds-pdsdiff)*c,\
                          ('dx2y2','D','py2'):  -(pds-pdsdiff)*c,\
                          ('dxy','L','py1'):  -(pdp+pdpdiff),\
                          ('dxy','R','py1'):   pdp+pdpdiff,\
                          ('dxy','U','px2'):   pdp-pdpdiff,\
                          ('dxy','D','px2'):  -(pdp-pdpdiff),\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'): -(pds+pdsdiff)/2.0,\
                          ('px1','L','d3z2r2'):  (pds+pdsdiff)/2.0,\
                          ('py2','D','d3z2r2'):  (pds-pdsdiff)/2.0,\
                          ('py2','U','d3z2r2'): -(pds-pdsdiff)/2.0,\
                          ('px1','R','dx2y2'):   (pds+pdsdiff)*c,\
                          ('px1','L','dx2y2'):  -(pds+pdsdiff)*c,\
                          ('py2','D','dx2y2'):   (pds-pdsdiff)*c,\
                          ('py2','U','dx2y2'):  -(pds-pdsdiff)*c,\
                          ('py1','R','dxy'):  -(pdp+pdpdiff),\
                          ('py1','L','dxy'):   pdp+pdpdiff,\
                          ('px2','D','dxy'):   pdp-pdpdiff,\
                          ('px2','U','dxy'):  -(pdp-pdpdiff)}
    elif pam.Norb==11:
        c = np.sqrt(3)/2.0
        tpd_nn_hop_fac = {('d3z2r2','L','px1'): -(pds+pdsdiff)/2.0,\
                          ('d3z2r2','R','px1'):  (pds+pdsdiff)/2.0,\
                          ('d3z2r2','U','py2'):  (pds-pdsdiff)/2.0,\
                          ('d3z2r2','D','py2'): -(pds-pdsdiff)/2.0,\
                          ('dx2y2','L','px1'):   (pds+pdsdiff)*c,\
                          ('dx2y2','R','px1'):  -(pds+pdsdiff)*c,\
                          ('dx2y2','U','py2'):   (pds-pdsdiff)*c,\
                          ('dx2y2','D','py2'):  -(pds-pdsdiff)*c,\
                          ('dxy','L','py1'):  -(pdp+pdpdiff),\
                          ('dxy','R','py1'):   pdp+pdpdiff,\
                          ('dxy','U','px2'):   pdp-pdpdiff,\
                          ('dxy','D','px2'):  -(pdp-pdpdiff),\
                          ('dxz','L','pz1'):  -(pdp+pdpdiff),\
                          ('dxz','R','pz1'):   pdp+pdpdiff,\
                          ('dyz','U','pz2'):   pdp-pdpdiff,\
                          ('dyz','D','pz2'):  -(pdp-pdpdiff),\
                          # below just inverse dir of the above one by one
                          ('px1','R','d3z2r2'): -(pds+pdsdiff)/2.0,\
                          ('px1','L','d3z2r2'):  (pds+pdsdiff)/2.0,\
                          ('py2','D','d3z2r2'):  (pds-pdsdiff)/2.0,\
                          ('py2','U','d3z2r2'): -(pds-pdsdiff)/2.0,\
                          ('px1','R','dx2y2'):   (pds+pdsdiff)*c,\
                          ('px1','L','dx2y2'):  -(pds+pdsdiff)*c,\
                          ('py2','D','dx2y2'):   (pds-pdsdiff)*c,\
                          ('py2','U','dx2y2'):  -(pds-pdsdiff)*c,\
                          ('py1','R','dxy'):  -(pdp+pdpdiff),\
                          ('py1','L','dxy'):   pdp+pdpdiff,\
                          ('px2','D','dxy'):   pdp-pdpdiff,\
                          ('px2','U','dxy'):  -(pdp-pdpdiff),\
                          ('pz1','R','dxz'):  -(pdp+pdpdiff),\
                          ('pz1','L','dxz'):   pdp+pdpdiff,\
                          ('pz2','D','dyz'):   pdp-pdpdiff,\
                          ('pz2','U','dyz'):  -(pdp-pdpdiff)}
    ########################## tpp below ##############################
    if pam.Norb==7:
        tpp_nn_hop_fac = {('UR','px','py'): -tpp,\
                          ('UL','px','py'):  tpp,\
                          ('DL','px','py'): -tpp,\
                          ('DR','px','py'):  tpp}
    elif pam.Norb==9:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps)}
    elif pam.Norb==11:
        tpp_nn_hop_fac = {('UR','px1','px2'):  0.5*(ppp-pps),\
                          ('UL','px1','px2'):  0.5*(ppp-pps),\
                          ('DL','px1','px2'):  0.5*(ppp-pps),\
                          ('DR','px1','px2'):  0.5*(ppp-pps),\
                          ('UR','py1','py2'):  0.5*(ppp-pps),\
                          ('UL','py1','py2'):  0.5*(ppp-pps),\
                          ('DL','py1','py2'):  0.5*(ppp-pps),\
                          ('DR','py1','py2'):  0.5*(ppp-pps),\
                          ('UR','px1','py2'): -0.5*(ppp+pps),\
                          ('UL','px1','py2'):  0.5*(ppp+pps),\
                          ('DL','px1','py2'): -0.5*(ppp+pps),\
                          ('DR','px1','py2'):  0.5*(ppp+pps),\
                          ('UR','px2','py1'): -0.5*(ppp+pps),\
                          ('UL','px2','py1'):  0.5*(ppp+pps),\
                          ('DL','px2','py1'): -0.5*(ppp+pps),\
                          ('DR','px2','py1'):  0.5*(ppp+pps),\
                          ('UR','pz1','pz2'):  ppp,\
                          ('UL','pz1','pz2'):  ppp,\
                          ('DL','pz1','pz2'):  ppp,\
                          ('DR','pz1','pz2'):  ppp}
        
    return tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac, tpp_nn_hop_fac
        
def get_interaction_mat(A, sym):
    '''
    Get d-d Coulomb and exchange interaction matrix
    total_spin based on lat.spin_int: up:1 and dn:0
    
    Rotating by 90 degrees, x goes to y and indeed y goes to -x so that this basically interchanges 
    spatial wave functions of two holes and can introduce - sign (for example (dxz, dyz)).
    But one has to look at what such a rotation does to the Slater determinant state of two holes.
    Remember the triplet state is (|up,down> +|down,up>)/sqrt2 so in interchanging the holes 
    the spin part remains positive so the spatial part must be negative. 
    For the singlet state interchanging the electrons the spin part changes sign so the spatial part can stay unchanged.
    
    Triplets cannot occur for two holes in the same spatial wave function while only singlets only can
    But both singlets and triplets can occur if the two holes have orthogonal spatial wave functions 
    and they will differ in energy by the exchange terms
    
    ee denotes xz,xz or xz,yz depends on which integral <ab|1/r_12|cd> is nonzero, see handwritten notes
    
    AorB_sym = +-1 is used to label if the state is (e1e1+e2e2)/sqrt(2) or (e1e1-e2e2)/sqrt(2)
    For syms (in fact, all syms except 1A1 and 1B1) without the above two states, AorB_sym is set to be 0
    
    Here different from all other codes, change A by A/2 to decrease the d8 energy to some extent
    the remaining A/2 is shifted to d10 states, see George's email on Jun.21, 2021
    and also set_edepeOs subroutine
    '''
    B = pam.B
    C = pam.C
    
    # not useful if treat 1A1 and 1B1 as correct ee states as (exex +- eyey)/sqrt(2)
    if sym=='1AB1':
        fac = np.sqrt(6)
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 4,\
                       ('d3z2r2','dx2y2') : 5}
        interaction_mat = [[A+4.*B+3.*C,  4.*B+C,       4.*B+C,           B+C,           B+C,       0], \
                           [4.*B+C,       A+4.*B+3.*C,  C,             3.*B+C,        3.*B+C,       0], \
                           [4.*B+C,       C,            A+4.*B+3.*C,   3.*B+C,        3.*B+C,       0], \
                           [B+C,          3.*B+C,       3.*B+C,        A+4.*B+3.*C,   3.*B+C,       B*fac], \
                           [B+C,          3.*B+C,       3.*B+C,        3.*B+C,        A+4.*B+3.*C, -B*fac], \
                           [0,            0,            0,              B*fac,         -B*fac,      A+2.*C]]
        
    if sym=='1A1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 1
        fac = np.sqrt(2)
        state_order = {('d3z2r2','d3z2r2'): 0,\
                       ('dx2y2','dx2y2')  : 1,\
                       ('dxy','dxy')      : 2,\
                       ('dxz','dxz')      : 3,\
                       ('dyz','dyz')      : 3}
        interaction_mat = [[A+4.*B+3.*C,  4.*B+C,       4.*B+C,        fac*(B+C)], \
                           [4.*B+C,       A+4.*B+3.*C,  C,             fac*(3.*B+C)], \
                           [4.*B+C,       C,            A+4.*B+3.*C,   fac*(3.*B+C)], \
                           [fac*(B+C),    fac*(3.*B+C), fac*(3.*B+C),  A+7.*B+4.*C]]
        
    if sym=='1B1':
        Stot = 0
        Sz_set = [0]
        AorB_sym = -1
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dx2y2'): 0,\
                       ('dxz','dxz')     : 1,\
                       ('dyz','dyz')     : 1}
        interaction_mat = [[A+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A+B+2.*C]]
        
    if sym=='1A2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0}
        interaction_mat = [[A+4.*B+2.*C]]
        
    if sym=='3A2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('dx2y2','dxy'): 0,\
                       ('dxz','dyz')  : 1}
        interaction_mat = [[A+4.*B,   6.*B], \
                           [6.*B,     A-5.*B]]
        
    if sym=='3B1':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        state_order = {('d3z2r2','dx2y2'): 0}
        interaction_mat = [[A-8.*B]]
        
    if sym=='1B2':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxy'): 0,\
                       ('dxz','dyz')   : 1}
        interaction_mat = [[A+2.*C,    2.*B*fac], \
                           [2.*B*fac,  A+B+2.*C]]
        
    if sym=='3B2':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = -1
        state_order = {('d3z2r2','dxy'): 0}
        interaction_mat = [[A-8.*B]]
        
    if sym=='1E':
        Stot = 0
        Sz_set = [0]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}    
        interaction_mat = [[A+3.*B+2.*C,  0,           -B*fac,      0,          0,        -B*fac], \
                           [0,            A+3.*B+2.*C,  0,          B*fac,     -B*fac,     0], \
                           [-B*fac,       0,            A+B+2.*C,   0,          0,        -3.*B], \
                           [0,            B*fac,        0,          A+B+2.*C,   3.*B,      0 ], \
                           [0,           -B*fac,        0,          3.*B,       A+B+2.*C,  0], \
                           [-B*fac,       0,           -3.*B,       0,          0,         A+B+2.*C]]
        
    if sym=='3E':
        Stot = 1
        Sz_set = [-1,0,1]
        AorB_sym = 0
        fac = np.sqrt(3)
        state_order = {('d3z2r2','dxz'): 0,\
                       ('d3z2r2','dyz'): 1,\
                       ('dx2y2','dxz') : 2,\
                       ('dx2y2','dyz') : 3,\
                       ('dxy','dxz')   : 4,\
                       ('dxy','dyz')   : 5}        
        interaction_mat = [[A+B,         0,         -3.*B*fac,    0,          0,        -3.*B*fac], \
                           [0,           A+B,        0,           3.*B*fac,  -3.*B*fac,  0], \
                           [-3.*B*fac,   0,          A-5.*B,      0,          0,         3.*B], \
                           [0,           3.*B*fac,   0,           A-5.*B,    -3.*B,      0 ], \
                           [0,          -3.*B*fac,   0,          -3.*B,       A-5.*B,    0], \
                           [-3.*B*fac,   0,          3.*B,        0,          0,         A-5.*B]]
        
    return state_order, interaction_mat, Stot, Sz_set, AorB_sym

def set_matrix_element(row,col,data,new_state,col_index,VS,element):
    '''
    Helper function that is used to set elements of a matrix using the
    sps coo format.

    Parameters
    ----------
    row: python list containing row indices
    col: python list containing column indices
    data: python list containing non-zero matrix elements
    col_index: column index that is to be appended to col
    new_state: new state corresponding to the row index that is to be
        appended.
    VS: VariationalSpace class from the module variationalSpace
    element: (complex) matrix element that is to be appended to data.

    Returns
    -------
    None, but appends values to row, col, data.
    '''
    row_index = VS.get_index(new_state)
    if row_index != None:
        data.append(element)
        row.append(row_index)
        col.append(col_index)

def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_orbs, tpd_nn_hop_fac):
    '''
    Create nearest neighbor (NN) pd hopping part of the Hamiltonian
    Only hole can hop with tpd

    Parameters
    ----------
    VS: VariationalSpace class from the module variationalSpace
    
    Returns
    -------
    matrix: (sps coo format) t_pd hopping part of the Hamiltonian without 
        the prefactor t_pd.
    
    Note from the sps documentation
    -------------------------------
    By default when converting to CSR or CSC format, duplicate (i,j)
    entries will be summed together
    '''    
    print ("start create_tpd_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpd_keys = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        # double check which cost some time, might not necessary
        assert VS.get_uid(start_state) == VS.lookup_tbl[i]
        
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']

        # hole 1 hops: some d-orbitals might have no tpd
        if orb1 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb1]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)
                if orbs1 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition(x1+vx,y1+vy,x2,y2):
                    continue

                # consider t_pd for all cases; when up hole hops, dn hole should not change orb
                for o1 in orbs1:
                    if o1 not in tpd_orbs:
                        continue
                    # consider Pauli principle
                    if s1==s2 and o1==orb2 and (x1+vx,y1+vy,z1+vz)==(x2,y2,z2):
                        continue

                    tmp_state = vs.create_state(s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb1, dir_, o1])
                    if o12 in tpd_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)

        # hole 2 hops; some d-orbitals might have no tpd
        if orb2 in tpd_orbs:
            for dir_ in tpd_nn_hop_dir[orb2]:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)
                if orbs2 == ['NotOnSublattice']:
                    continue

                if not vs.check_in_vs_condition(x1,y1,x2+vx,y2+vy):
                    continue

                for o2 in orbs2:
                    if o2 not in tpd_orbs:
                        continue
                    # consider Pauli principle
                    if s1==s2 and orb1==o2 and (x1,y1,z1)==(x2+vx, y2+vy, z2+vz):
                        continue

                    tmp_state = vs.create_state(s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = tuple([orb2, dir_, o2])
                    if o12 in tpd_keys:
                        set_matrix_element(row,col,data,new_state,i,VS,tpd_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out


def create_tpp_nn_matrix(VS,tpp_nn_hop_fac): 
    '''
    similar to comments in create_tpp_nn_matrix
    '''   
    print ("start create_tpp_nn_matrix")
    print ("==========================")
    
    dim = VS.dim
    tpp_orbs = tpp_nn_hop_fac.keys()
    data = []
    row = []
    col = []
    for i in range(0,dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']

        # hole1 hops: only p-orbitals has t_pp 
        if orb1 in pam.O_orbs: 
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs1 = lat.get_unit_cell_rep(x1+vx, y1+vy, z1+vz)

                if orbs1!=pam.O1_orbs and orbs1!=pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition(x1+vx,y1+vy,x2,y2):
                    continue

                # consider t_pp for all cases; when one hole hops, the other hole should not change orb
                for o1 in orbs1:
                    # consider Pauli principle
                    if s1==s2 and o1==orb2 and (x1+vx,y1+vy,z1+vz)==(x2,y2,z2):
                        continue

                    tmp_state = vs.create_state(s1,o1,x1+vx,y1+vy,z1+vz,s2,orb2,x2,y2,z2)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    s1n = new_state['hole1_spin']
                    s2n = new_state['hole2_spin']
                    orb1n = new_state['hole1_orb']
                    orb2n = new_state['hole2_orb']
                    x1n, y1n, z1n = new_state['hole1_coord']
                    x2n, y2n, z2n = new_state['hole2_coord']
                    #print x1,y1,orb1,s1,x2,y2,orb2,s2,'tpp hops to',x1n, y1n,orb1n,s1n,x2n, y2n,orb2n,s2n

                    o12 = sorted([orb1, dir_, o1])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

        # hole 2 hops, only p-orbitals has t_pp 
        if orb2 in pam.O_orbs:
            for dir_ in tpp_nn_hop_dir:
                vx, vy, vz = directions_to_vecs[dir_]
                orbs2 = lat.get_unit_cell_rep(x2+vx, y2+vy, z2+vz)

                if orbs2!=pam.O1_orbs and orbs2!=pam.O2_orbs:
                    continue

                if not vs.check_in_vs_condition(x1,y1,x2+vx,y2+vy): 
                    continue

                for o2 in orbs2:
                    # consider Pauli principle
                    if s1==s2 and orb1==o2 and (x1,y1,z1)==(x2+vx, y2+vy, z2+vz):
                        continue

                    tmp_state = vs.create_state(s1,orb1,x1,y1,z1,s2,o2,x2+vx,y2+vy,z2+vz)
                    new_state,ph = vs.make_state_canonical(tmp_state)
                    #new_state,ph = vs.make_state_canonical_old(tmp_state)

                    o12 = sorted([orb2, dir_, o2])
                    o12 = tuple(o12)
                    if o12 in tpp_orbs:
                        set_matrix_element(row,col,data,new_state,i,VS,tpp_nn_hop_fac[o12]*ph)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out


def create_edep_diag_matrix(VS, A, epx, epy):
    '''
    Create diagonal part of the site energies
    '''    
    print ("start create_edep_diag_matrix")
    print ("=============================")
    dim = VS.dim
    data = []
    row = []
    col = []
    idxs = np.zeros(dim)

    for i in range(0,dim):
        diag_el = 0.
        state = VS.get_state(VS.lookup_tbl[i])

        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']

        # d9L
        if orb1 in pam.Ni_orbs and orb2 in pam.O1_orbs:
            diag_el += pam.ed[orb1] + epx
            idxs[i] = 1
        elif orb1 in pam.Ni_orbs and orb2 in pam.O2_orbs:
            diag_el += pam.ed[orb1] + epy
            idxs[i] = 1
        elif orb2 in pam.Ni_orbs and orb1 in pam.O1_orbs: 
            diag_el += pam.ed[orb2] + epx
            idxs[i] = 1
        elif orb2 in pam.Ni_orbs and orb1 in pam.O2_orbs: 
            diag_el += pam.ed[orb2] + epy
            idxs[i] = 1

        # d8
        elif orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs: 
            diag_el += pam.ed[orb1] + pam.ed[orb2] 
            idxs[i] = 1

        # d10L2
        elif orb1 in pam.O1_orbs and orb2 in pam.O1_orbs: 
            diag_el += 2.*epx
            idxs[i] = 1
        elif orb1 in pam.O2_orbs and orb2 in pam.O2_orbs: 
            diag_el += 2.*epy
            idxs[i] = 1
        elif (orb1 in pam.O1_orbs and orb2 in pam.O2_orbs) or (orb2 in pam.O1_orbs and orb1 in pam.O2_orbs): 
            diag_el += epx+epy
            idxs[i] = 1

        data.append(diag_el); row.append(i); col.append(i)
        #print i, diag_el

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    #print min(data)
    #print len(row), len(col)
    
   # for ii in range(0,len(row)):
   #     if data[ii]==0:
   #         print ii
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))
    
    return out
    
def get_double_occu_list(VS):
    '''
    Get the list of states that two holes are both d or p-orbitals
    '''
    dim = VS.dim
    d_list = []
    pUppx_list = []
    pUppy_list = []
    pUpp_list = []
    pVpp_list = []
    pUpdx_list = []
    pUpdy_list = []
    for i in range(0,dim):
        state = VS.get_state(VS.lookup_tbl[i])
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        dist = (x1-x2)**2 + (y1-y2)**2

        if (x1,y1) == (x2,y2):
            if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
                d_list.append(i)
                #print "d_double: no_eh", i, s1,orb1,x1,y1,s2,orb2,x2,y2
            elif x1%2==1 and y1%2==0:
                
                pUppx_list.append(i)
            elif x1%2==0 and y1%2==1:
                pUppy_list.append(i)
            else:
                pUpp_list.append(i)
        elif abs(dist-2)<1.e-5:
            assert(orb1 in pam.O_orbs and orb2 in pam.O_orbs)
            pVpp_list.append(i)
        elif abs(dist-1)<1.e-5:
            if (x1**2+y1**2 ==0) or (x2**2+y2**2 ==0):
                if ((x1+y1)%2==1 and x1%2==1 and y1%2==0) or ((x2+y2)%2==1 and x2%2==1 and y2%2==0):
                
                    pUpdx_list.append(i) 
                if ((x1+y1)%2==1 and x1%2==0 and y1%2==1) or ((x2+y2)%2==1 and x2%2==0 and y2%2==1):
                    pUpdy_list.append(i) 
            
            
    print ("len(d_list)", len(d_list))
    print ("len(pUpp_list)", len(pUpp_list))
    print ("len(pUppx_list)", len(pUppx_list))
    print ("len(pUppy_list)", len(pUppy_list))
    print ("len(pVpp_list)", len(pVpp_list))
    print ("len(pUpdx_list)", len(pUpdx_list))
    print ("len(pUpdy_list)", len(pUpdy_list))
    
    return d_list ,pUppx_list,pUppy_list,pUpp_list, pVpp_list, pUpdx_list ,pUpdy_list


def create_interaction_matrix_ALL_syms(VS,d_double,pUppx_double,pUppy_double,pUpp_double,pVpp_double,pUpdx_double,pUpdy_double,S_val, \
                                       Sz_val, AorB_sym, A, Upp,Udif,Vpp,Upd,Updf):
    '''
    Create Coulomb-exchange interaction matrix of d-multiplets including all symmetries
    
    Loop over all d_double states, find the corresponding sym channel; 
    the other loop over all d_double states, if it has same sym channel and S, Sz
    enter into the matrix element
    
    There are some complications or constraints due to three holes and one Nd electron:
    From H_matrix_reducing_VS file, to set up interaction between states i and j:
    1. i and j belong to the same type, same order of orbitals to label the state (idxi==idxj below)
    2. i and j's spins are same; or L and s should also have same spin
    3. Positions of L and Nd-electron should also be the same
    '''    
    #print "start create_interaction_matrix"
    
    Norb = pam.Norb
    dim = VS.dim
    data = []
    row = []
    col = []
    dd_state_indices = []
    
    channels = ['1A1','1A2','3A2','1B1','3B1','1E','3E','1B2','3B2']

    for sym in channels:
        state_order, interaction_mat, Stot, Sz_set, AorB = get_interaction_mat(A, sym)
        sym_orbs = state_order.keys()
        #print "orbitals in sym ", sym, "= ", sym_orbs

        for i in d_double:
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            is1 = state['hole1_spin']
            is2 = state['hole2_spin']
            io1 = state['hole1_orb']
            io2 = state['hole2_orb']
        
            o12 = sorted([io1,io2])
            o12 = tuple(o12)

            # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
            S12  = S_val[i]
            Sz12 = Sz_val[i]

            # continue only if (o1,o2) is within desired sym
            if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
                continue

            if (io1==io2=='dxz' or io1==io2=='dyz') and AorB_sym[i]!=AorB:
                continue

            # get the corresponding index in sym for setting up matrix element
            idx1 = state_order[o12]
            
            for j in d_double:
                if j<i:
                    continue

                state = VS.get_state(VS.lookup_tbl[j])
                js1 = state['hole1_spin']
                js2 = state['hole2_spin']
                jo1 = state['hole1_orb']
                jo2 = state['hole2_orb']
               
                o34 = sorted([jo1,jo2])
                o34 = tuple(o34)
                S34  = S_val[j]
                Sz34 = Sz_val[j]

                if (jo1==jo2=='dxz' or jo1==jo2=='dyz') and AorB_sym[j]!=AorB:
                    continue

                # only same total spin S and Sz state have nonzero matrix element
                if o34 in sym_orbs and S34==S12 and Sz34==Sz12:
                    idx2 = state_order[o34]

                    #print o12[0],o12[1],S12,Sz12," ",o34[0],o34[1],S34,Sz34," ", interaction_mat[idx1][idx2]
                    #print idx1, idx2

                    val = interaction_mat[idx1][idx2]
                    data.append(val); row.append(i); col.append(j)
                    
                    # make symmetric interaction matrix
                    if j!=i:
                        data.append(val); row.append(j); col.append(i)

    # Create Upp matrix for p-orbital multiplets
    for i in pUppx_double:
        data.append(Upp+Udif); row.append(i); col.append(i)
    for i in pUppy_double:
        data.append(Upp-Udif); row.append(i); col.append(i)
    for i in pUpp_double:
        data.append(Upp); row.append(i); col.append(i)
    for i in pVpp_double:
        data.append(Vpp); row.append(i); col.append(i)
    for i in pUpdx_double:
        data.append(Upd+Updf); row.append(i); col.append(i)
    for i in pUpdy_double:
        data.append(Upd-Updf); row.append(i); col.append(i)


    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    
    # check if hoppings occur within groups of (up,up), (dn,dn), and (up,dn) 
    #assert(check_spin_group(row,col,data,VS)==True)
    out = sps.coo_matrix((data,(row,col)),shape=(dim,dim))

    return out

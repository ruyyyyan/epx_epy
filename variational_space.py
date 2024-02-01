'''
Contains a class for the variational space for the cuprate layer
and functions used to represent states as dictionaries.
Distance (L1-norm) between a hole and Cu-site (0,0) cannot > cutoff Mc
'''

import parameters as pam
import lattice as lat
import bisect
import numpy as np

def create_state(s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2):
    '''
    Creates a dictionary representing a state

    Parameters
    ----------
    s1, s2   : string of spin
    orb_up, orb_dn : string of orb
    x_up, y_up: integer coordinates of hole1
    x_dn, y_dn: integer coordinates of hole2

    Note
    ----
    It is possible to return a state that is not in the 
    variational space because the hole-hole Manhattan distance
    exceeds Mc.
    '''
    assert not (((x1,y1,z1))==(x2,y2,z2) and s1==s2 and orb1==orb2)
    assert(check_in_vs_condition(x1,y1,x2,y2))
    
    state = {'hole1_spin' :s1,\
             'hole1_orb'  :orb1,\
             'hole1_coord':(x1,y1,z1),\
             'hole2_spin' :s2,\
             'hole2_orb'  :orb2,\
             'hole2_coord':(x2,y2,z2)}
    
    return state

def make_state_canonical(state):
    '''
    1. There are a few cases to avoid having duplicate states where 
    the holes are indistinguishable. 
    
    The sign change due to anticommuting creation operators should be 
    taken into account so that phase below has a negative sign
    =============================================================
    Case 1: 
    Note here is different from Mirko's version for only same spin !!
    Now whenever when hole2 is on left of hole 1, switch them and
    order the hole coordinates in such a way that the coordinates 
    of the left creation operator are lexicographically
    smaller than those of the right.
    =======================================================
    Case 2: 
    If two holes locate on the same (x,y) sites (even if including apical pz with z=1)
    a) same spin state: 
      up, dxy,    (0,0), up, dx2-y2, (0,0)
    = up, dx2-y2, (0,0), up, dxy,    (0,0)
    need sort orbital order
    b) opposite spin state:
    only keep spin1 = up state
    
    Different from CuO2 periodic lattice, the phase simply needs to be 1 or -1
    
    2. Besides, see emails with Mirko on Mar.1, 2018:
    Suppose Tpd|state_i> = |state_j> = phase*|canonical_state_j>, then 
    tpd = <state_j | Tpd | state_i> 
        = conj(phase)* <canonical_state_j | Tpp | state_i>
    
    so <canonical_state_j | Tpp | state_i> = tpd/conj(phase)
                                           = tpd*phase
    
    Because conj(phase) = 1/phase, *phase and /phase in setting tpd and tpp seem to give same results
    But need to change * or / in both tpd and tpp functions
    
    Similar for tpp
    '''
    s1 = state['hole1_spin']
    s2 = state['hole2_spin']
    orb1 = state['hole1_orb']
    orb2 = state['hole2_orb']
    x1, y1, z1 = state['hole1_coord']
    x2, y2, z2 = state['hole2_coord']
        
    canonical_state = state
    phase = 1.0
        
    if (x2,y2)<(x1,y1):
        canonical_state = create_state(s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1)
        phase = -1.0
        
    # note that z1 can differ from z2 in the presence of apical pz orbital
    elif (x1,y1)==(x2,y2):           
        if s1==s2:
            o12 = list(sorted([orb1,orb2]))
            if o12[0]==orb2:
                canonical_state = create_state(s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1)
                phase = -1.0  
        elif s1=='dn' and s2=='up':
            canonical_state = create_state('up',orb2,x2,y2,z2,'dn',orb1,x1,y1,z1)
            phase = -1.0

    return canonical_state, phase
    
def calc_manhattan_dist(x1,y1,x2,y2):
    '''
    Calculate the Manhattan distance (L1-norm) between two vectors
    (x1,y1) and (x2,y2).
    '''
    out = abs(x1-x2) + abs(y1-y2)
    return out

def check_in_vs_condition(x1,y1,x2,y2):
    '''
    Restrictions: the distance between one hole and Cu-site (0,0)
    and two-hole distance less than cutoff Mc
    '''     
    if calc_manhattan_dist(x1,y1,0,0) > pam.Mc or \
        calc_manhattan_dist(x2,y2,0,0) > pam.Mc or \
        calc_manhattan_dist(x1,y1,x2,y2) > 2*pam.Mc:
        return False
    else:
        return True

class VariationalSpace:
    '''
    Distance (L1-norm) between any two particles must not exceed a
    cutoff denoted by Mc. 

    Attributes
    ----------
    Mc: Cutoff for the hole-hole 
    lookup_tbl: sorted python list containing the unique identifiers 
        (uid) for all the states in the variational space. A uid is an
        integer which can be mapped to a state (see docsting of get_uid
        and get_state).
    dim: number of states in the variational space, i.e. length of
        lookup_tbl
    filter_func: a function that is passed to create additional 
        restrictions on the variational space. Default is None, 
        which means that no additional restrictions are implemented. 
        filter_func takes exactly one parameter which is a dictionary representing a state.

    Methods
    -------
    __init__
    create_lookup_table
    get_uid
    get_state
    get_index
    '''

    def __init__(self,Mc,filter_func=None):
        self.Mc = Mc
        if filter_func == None:
            self.filter_func = lambda x: True
        else:
            self.filter_func = filter_func
        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print ("VS.dim = ", self.dim)
        #self.print_VS()

    def print_VS(self):
        for i in range(0,self.dim):
            state = self.get_state(self.lookup_tbl[i])
            ts1 = state['hole1_spin']
            ts2 = state['hole2_spin']
            torb1 = state['hole1_orb']
            torb2 = state['hole2_orb']
            tx1, ty1, tz1 = state['hole1_coord']
            tx2, ty2, tz2 = state['hole2_coord']
            #if ts1=='up' and ts2=='up':
            #if torb1=='dx2y2' and torb2=='px':
            print (i, ts1,torb1,tx1,ty1,ts2,torb2,tx2,ty2)
                
    def create_lookup_tbl(self):
        '''
        Create a sorted lookup table containing the unique identifiers 
        (uid) of all the states in the variational space.
        
        Manhattan distance between a hole and the Cu-site (0,0) does not exceed Mc
        Then the hole-hole distance cannot be larger than 2*Mc

        Returns
        -------
        lookup_tbl: sorted python list.
        '''
        Mc = self.Mc
        lookup_tbl = []

        for ux in range(-Mc,Mc+1):
            Bu = Mc - abs(ux)
            for uy in range(-Bu,Bu+1):
                for uz in [0]:
                    orb1s = lat.get_unit_cell_rep(ux,uy,uz)
                    if orb1s==['NotOnSublattice']:
                        continue

                    for vx in range(-Mc,Mc+1):
                        Bv = Mc - abs(vx)
                        for vy in range(-Bv,Bv+1):
                            for vz in [0]:
                                orb2s = lat.get_unit_cell_rep(vx,vy,vz)
                                if orb2s==['NotOnSublattice']:
                                    continue
                                if calc_manhattan_dist(ux,uy,vx,vy)>2*Mc:
                                    continue

                                for orb1 in orb1s:
                                    for orb2 in orb2s:
                                        for s1 in ['up','dn']:
                                            for s2 in ['up','dn']:   
                                                # try screen out same spin states
                                                if pam.VS_only_up_dn==1:
                                                    if s1==s2:
                                                        continue
                                                # try only keep Sz=1 triplet states
                                                if pam.VS_only_up_up==1:
                                                    if not s1==s2=='up':
                                                        continue

                                                # consider Pauli principle
                                                if s1==s2 and orb1==orb2 and ux==vx and uy==vy and uz==vz:
                                                    continue 

                                                #if s1=='dn' and s2=='dn':
                                                #    print "candiate state: ", s1,orb1,ux,uy,s2,orb2,vx,vy

                                                if check_in_vs_condition(ux,uy,vx,vy):
                                                    state = create_state(s1,orb1,ux,uy,uz,s2,orb2,vx,vy,vz)
                                                    canonical_state,_ = make_state_canonical(state)

                                                if self.filter_func(canonical_state):
                                                    uid = self.get_uid(canonical_state)
                                                    lookup_tbl.append(uid)
 
        lookup_tbl = list(set(lookup_tbl)) # remove duplicates
        lookup_tbl.sort()
        #print "\n lookup_tbl:\n", lookup_tbl
        return lookup_tbl
            
    def check_in_vs(self,state):
        '''
        Check if a given state is in VS

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.
        Mc: integer cutoff for the Manhattan distance.

        Returns
        -------
        Boolean: True or False
        '''
        assert(self.filter_func(state) in [True,False])
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
  
        if check_in_vs_condition(x1,y1,x2,y2):
            return True
        else:
            return False

    def get_uid(self,state):
        '''
        Every state in the variational space is associated with a unique
        identifier (uid) which is an integer number.
        
        Rule for setting uid (example below but showing ideas):
        Assuming that i1, i2 can take the values -1 and +1. Make sure that uid is always larger or equal to 0. 
        So add the offset +1 as i1+1. Now the largest value that (i1+1) can take is (1+1)=2. 
        Therefore the coefficient in front of (i2+1) should be 3. This ensures that when (i2+1) is larger than 0, 
        it will be multiplied by 3 and the result will be larger than any possible value of (i1+1). 
        The coefficient in front of (o1+1) needs to be larger than the largest possible value of (i1+1) + 3*(i2+1). 
        This means that the coefficient in front of (o1+1) must be larger than (1+1) + 3*(1+1) = 8, 
        so you can choose 9 and you get (i1+1) + 3*(i2+1) + 9*(o1+1) and so on ....

        Parameters
        ----------
        state: dictionary created by one of the functions which create states.

        Returns
        -------
        uid (integer) or None if the state is not in the variational space.
        '''
        # Need to check if the state is in the VS, because after hopping the state can be outside of VS
        if not self.check_in_vs(state):
            return None
        
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
        N2 = 16*N*N

        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        i1 = lat.spin_int[s1]
        i2 = lat.spin_int[s2]
        orb1 = state['hole1_orb']
        orb2 = state['hole2_orb']
        o1 = lat.orb_int[orb1]
        o2 = lat.orb_int[orb2]
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']

        uid = i1 + 2*i2 + 4*z1 + 8*z2 + 16*o1 + 16*N*o2 +N2*( (y1+s) + (x1+s)*B1 + (y2+s)*(B2+B1+1) + (x2+s)*(B3+B2+B1)*2 )
        
        # check if uid maps back to the original state, namely uid's uniqueness
        tstate = self.get_state(uid)
        ts1 = tstate['hole1_spin']
        ts2 = tstate['hole2_spin']
        torb1 = tstate['hole1_orb']
        torb2 = tstate['hole2_orb']
        tx1, ty1, tz1 = tstate['hole1_coord']
        tx2, ty2, tz2 = tstate['hole2_coord']
        assert((s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)==(ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2))
            
        return uid

    def get_state(self,uid):
        '''
        Given a unique identifier, return the corresponding state. 
        ''' 
        N = pam.Norb 
        s = self.Mc+1 # shift to make values positive
        B1 = 2*self.Mc+4
        B2 = B1*B1
        B3 = B1*B2
        N2 = 16*N*N
        
        x2 = int(uid/(N2*(B3+B2+B1)*2)) - s
        uid_ = uid % (N2*(B3+B2+B1)*2)
        y2 = int(uid_/(N2*(B2+B1+1))) - s
        uid_ = uid_ % (N2*(B2+B1+1))
        x1 = int(uid_/(N2*B1)) - s
        uid_ = uid_ % (N2*B1)
        y1 = int(uid_/N2) - s
        uid_ = uid_ % N2
        o2 = int(uid_/(16*N))
        uid_ = uid_ % (16*N)
        o1 = int(uid_/16)
        uid_ = uid_ % 16
        z2 = int(uid_/8)
        uid_ = uid_ % 8
        z1 = int(uid_/4) 
        uid_ = uid_ % 4
        i2 = int(uid_/2) 
        i1 = uid_%2 
        
        orb2 = lat.int_orb[o2]
        orb1 = lat.int_orb[o1]
        s2 = lat.int_spin[i2]
        s1 = lat.int_spin[i1]
        
        state = create_state(s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2)
        return state

    def get_index(self,state):
        '''
        Return the index under which the state is stored in the lookup
        table.  These indices are consecutive and can be used to
        index, e.g. the Hamiltonian matrix

        Parameters
        ----------
        state: dictionary representing a state

        Returns
        -------
        index: integer such that lookup_tbl[index] = get_uid(state,Mc).
            If the state is not in the variational space None is returned.
        '''
        uid = self.get_uid(state)
        if uid == None:
            return None
        else:
            index = bisect.bisect_left(self.lookup_tbl,uid)
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None

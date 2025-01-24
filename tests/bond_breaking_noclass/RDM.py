import numpy as np

"""
to make the calculatino of 1,2-RDM faster we separate into cases depending on the number of differences between the bra and the ket. The RDMs can then be calculated with less steps. It also avoids calculating RDMs that will have every element zero because of the number of differences between the bra and the ket being too high. 

The algorithm are based on the work in this article https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9648180/pdf/ct2c00738.pdf
"""

# find every difference in occupation states, puts first the difference in ascending order where the bra has occupation 1 then ascending order of the ones with the ket having occupation 1
def find_orbitals_differences(bra,ket):
    differences_index = []
    same_orbital_index = []
    counter = 0
    for index,(occ_bra,occ_ket) in enumerate(zip(bra[1:],ket[1:])):
        if occ_bra != occ_ket:
            if occ_ket:
                differences_index.append(index)
            else:
                differences_index.insert(counter,index)
                counter += 1 
        else:
            if occ_bra == 1:
                same_orbital_index.append(index)

    return len(differences_index),differences_index,same_orbital_index

# get the phase factor and the spin associated with a creation operator acting on the ket at a certain index
def get_pf_spin_operator(index,ket):
    n_dim = len(ket)-1
    occ = ket[1:]
    pf = (-1)**(np.sum(occ[:index]))
    if index < n_dim//2:
        spin = 0
    else:
        spin = 1
    return pf,spin

def kronecker_delta(i,j):
    if i==j: 
        return 1
    else:
        return 0

# computes matrix element of 2-RDM
def twoRDM_element(P,Q,R,S,bra,ket,spin_int=True):
    
    pf_bra = bra[0]
    pf_ket = ket[0]
    pf_overlap = pf_bra*pf_ket

    pf_P,spin_P = get_pf_spin_operator(P,bra)
    pf_Q,spin_Q = get_pf_spin_operator(Q,ket)
    pf_R,spin_R = get_pf_spin_operator(R,bra)
    pf_S,spin_S = get_pf_spin_operator(S,ket)

    pf_all = pf_overlap*pf_P*pf_Q*pf_R*pf_S

    # this part takes into account the doubling of same creatino operators that yield zero. It also changes the phase factor according to the order in which the operators are applied to the ket.    
    if P==R:
        return 0
    elif P>R:
        PR = 1
    elif P<R:
        PR = -1

    if S==Q:
        return 0
    elif Q>S:
        SQ = 1
    elif Q<S:
        SQ = -1

    # the kronecker are used to take into account the spin integral
    if spin_int:
        return pf_all*PR*SQ*kronecker_delta(spin_P,spin_Q)*kronecker_delta(spin_R,spin_S)
    else:
        return pf_all*PR*SQ




# PQRS with a\dagger_P a\dagger_R a_S a_Q
# get 2-RDM for zero differences between bra and ket
def two_rdm_zero_diff(bra,ket,same_orbitals_index,spin_int=True):
    n_dim = len(bra)-1
    two_rdm = np.zeros((n_dim,n_dim,n_dim,n_dim))
    for i in same_orbitals_index:
        for j in same_orbitals_index:
            two_rdm[i,j,j,i] = twoRDM_element(i,j,j,i,bra,ket,spin_int=True)
            two_rdm[i,i,j,j] = twoRDM_element(i,i,j,j,bra,ket,spin_int=True)
    return two_rdm

# PQRS with a\dagger_P a\dagger_R a_S a_Q
# get 2-RDM for two differences between bra and ket
def two_rdm_two_diff(bra,ket,differences_index,same_orbitals_index,spin_int=True):
    n_dim = len(bra)-1
    two_rdm = np.zeros((n_dim,n_dim,n_dim,n_dim))
    
    pf_bra = bra[0]
    pf_ket = ket[0]
    pf_overlap = pf_bra*pf_ket
    
    a = differences_index[0]
    b = differences_index[1]



    for i in same_orbitals_index:
        two_rdm[a,b,i,i] = twoRDM_element(a,b,i,i,bra,ket,spin_int=True)
        two_rdm[a,i,i,b] = twoRDM_element(a,i,i,b,bra,ket,spin_int=True)
        two_rdm[i,b,a,i] = twoRDM_element(i,b,a,i,bra,ket,spin_int=True)
        two_rdm[i,i,a,b] = twoRDM_element(i,i,a,b,bra,ket,spin_int=True)
    return two_rdm

# \gamma_PQRS with a\dagger_P a\dagger_R a_S a_Q
# get 2_RDM for four differences between bra and ket
def two_rdm_four_diff(bra,ket,differences_index,spin_int=True):
    n_dim = len(bra)-1
    two_rdm = np.zeros((n_dim,n_dim,n_dim,n_dim))
    
    pf_bra = bra[0]
    pf_ket = ket[0]
    pf_overlap = pf_bra*pf_ket


    a = differences_index[0]
    b = differences_index[1]
    c = differences_index[2]
    d = differences_index[3]
    
    pf_a,spin_a = get_pf_spin_operator(a,bra) 
    pf_b,spin_b = get_pf_spin_operator(b,bra) 
    pf_c,spin_c = get_pf_spin_operator(c,ket) 
    pf_d,spin_d = get_pf_spin_operator(d,ket) 

    pf_all = pf_overlap*pf_a*pf_b*pf_c*pf_d
    
    # two_rdm[P,Q,R,S] corresponds to a\dagger_P a\dagger_R a_S a_Q
    if spin_int:
        two_rdm[a,c,b,d] = pf_all*kronecker_delta(spin_a,spin_c)*kronecker_delta(spin_b,spin_d) 
        two_rdm[a,d,b,c] = pf_all*(-1)*kronecker_delta(spin_a,spin_d)*kronecker_delta(spin_b,spin_c)
        two_rdm[b,c,a,d] = pf_all*(-1)*kronecker_delta(spin_b,spin_c)*kronecker_delta(spin_a,spin_d)
        two_rdm[b,d,a,c] = pf_all*kronecker_delta(spin_b,spin_d)*kronecker_delta(spin_a,spin_c)
    else:
        two_rdm[a,c,b,d] = pf_all
        two_rdm[a,d,b,c] = pf_all*(-1)
        two_rdm[b,c,a,d] = pf_all*(-1)
        two_rdm[b,d,a,c] = pf_all

    return two_rdm

# get 2-RDM
def get_two_rdm_fast(bra,ket,return_zero=False,spin_int=True):
    n_dim = len(bra)-1
    diff,diff_index,orbitals_index = find_orbitals_differences(bra,ket)
    if diff > 4:
        if return_zero:
            return 0 
        else:
            return np.zeros((n_dim,n_dim,n_dim,n_dim))
    if diff == 4:
        return two_rdm_four_diff(bra,ket,diff_index,spin_int=True)
    if diff == 2:
        return two_rdm_two_diff(bra,ket,diff_index,orbitals_index,spin_int=True)
    if diff == 0:
        return two_rdm_zero_diff(bra,ket,orbitals_index,spin_int=True)


# get 2-RDM for multiconfigurational states
# added possibility of having different mc state for the bra and ket
def get_two_rdm_mc_fast(kets,coeffs,bras=None,bras_coeffs=None,spin_int=True):
    n_dim = len(kets[0]) -1 
    two_rdm_mc = np.zeros((n_dim,n_dim,n_dim,n_dim))
    if bras==None and bras_coeffs==None:
        for bra,ci in zip(kets,coeffs):
            for ket,cj in zip(kets,coeffs):
                two_rdm_mc += get_two_rdm_fast(bra,ket,spin_int=True)*ci*cj
    else:
        for bra,ci in zip(bras,bras_coeffs):
            for ket,cj in zip(kets,coeffs):
                two_rdm_mc += get_two_rdm_fast(bra,ket,spin_int=True)*ci*cj
    return two_rdm_mc

# get the spatial 2-RDM
def get_spatial_two_rdm_fast(two_rdm):
    n_dim = two_rdm.shape[0]
    n_half = n_dim//2
    spatial_two_rdm = np.zeros((n_half,n_half,n_half,n_half))
    for i in range(n_half):
        for j in range(n_half):
            for k in range(n_half):
                for l in range(n_half):
                    spatial_two_rdm[i,j,k,l] += two_rdm[i,j,k,l] + two_rdm[i+n_half,j+n_half,k,l] + two_rdm[i,j,k+n_half,l+n_half] + two_rdm[i+n_half,j+n_half,k+n_half,l+n_half]
    return spatial_two_rdm




# computes matrix element of 1-RDM
def oneRDM_element(P,Q,bra,ket,spin_int=True):

    pf_bra = bra[0]
    pf_ket = ket[0]
    pf_overlap = pf_bra*pf_ket

    pf_P,spin_P = get_pf_spin_operator(P,bra)
    pf_Q,spin_Q = get_pf_spin_operator(Q,ket)

    pf_all = pf_overlap*pf_P*pf_Q

    if spin_int:
        return pf_all*kronecker_delta(spin_P,spin_Q)
    else:
        return pf_all

# get 1-RDM for zero differences between bra and ket
def one_rdm_zero_diff(bra,ket,same_orbitals_index,spin_int=True):
    n_dim = len(bra)-1
    one_rdm = np.zeros((n_dim,n_dim))
    for i in same_orbitals_index:
        one_rdm[i,i] = oneRDM_element(i,i,bra,ket,spin_int=True) 

    return one_rdm

# get 1-RDM for two differences between bra and ket
def one_rdm_two_diff(bra,ket,differences_index,spin_int=True):
    n_dim = len(bra)-1
    one_rdm = np.zeros((n_dim,n_dim))
    
    a = differences_index[0]
    b = differences_index[1]

    one_rdm[a,b] = oneRDM_element(a,b,bra,ket,spin_int=True)

    return one_rdm

# get 1-RDM
def get_one_rdm_fast(bra,ket,return_zero=False,spin_int=True):
    n_dim = len(bra)-1
    diff,diff_index,same_orbitals_index = find_orbitals_differences(bra,ket)
    if diff > 2:
        if return_zero:
            return 0 
        else:
            return np.zeros((n_dim,n_dim))
    if diff == 2:
        return one_rdm_two_diff(bra,ket,diff_index,spin_int=True)
    if diff == 0:
        return one_rdm_zero_diff(bra,ket,same_orbitals_index,spin_int=True)

# get 1-RDM for multiconfigurational states
# added possibility of having different mc state for the bra and ket
def get_one_rdm_mc_fast(kets, coeffs,bras=None,bras_coeffs=None):
    n_dim = len(kets[0]) -1
    one_rdm_mc = np.zeros((n_dim,n_dim))

    if bras==None and bras_coeffs==None:
        for bra,ci in zip(kets,coeffs):
            for ket,cj in zip(kets,coeffs):
                one_rdm_mc += get_one_rdm_fast(bra,ket,spin_int=True)*ci*cj
    else:
        for bra,ci in zip(bras,bras_coeffs):
            for ket,cj in zip(kets,coeffs):
                one_rdm_mc += get_one_rdm_fast(bra,ket,spin_int=True)*ci*cj
    return one_rdm_mc



# get the spatial 1-RDM
def get_spatial_one_rdm_fast(one_rdm):
    n_dim = one_rdm.shape[0]
    n_half = n_dim//2
    spatial_one_rdm = np.zeros((n_half,n_half))
    for i in range(n_half):
        for j in range(n_half):
            spatial_one_rdm[i,j] = one_rdm[i,j] + one_rdm[i+n_half,j+n_half] 
    return spatial_one_rdm

# check RDMs between original code and this one
def check_rdm(rdm_old,rdm_fast):
    print("START CHECKING RDMs")
    n_dim = rdm_old.shape[0]
    dim = len(rdm_old.shape)
    if dim ==2:
        for i in range(n_dim):
            for j in range(n_dim):
                if rdm_old[i,j] != rdm_fast[i,j]:
                    print(rdm_old[i,j],rdm_fast[i,j])
    elif dim == 4:
        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_dim):
                    for l in range(n_dim):
                        if rdm_old[i,k,j,l] != rdm_fast[i,j,k,l]:
                            print(rdm_old[i,k,j,l],rdm_fast[i,j,k,l])
    print("CHECKING RDMs DONE")
    return


















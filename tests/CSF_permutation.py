import numpy as np





def permute_vector(vector,permutation):
    initial_vector = vector.copy()
    vector_local = vector.copy()
    n_dim = len(vector)
    pf = 1
    for i in range(n_dim):
        i_index = permutation.index(i)
        i_occupation = initial_vector[i_index]
        elem_left = np.sum(vector_local[i:i_index])
        pf *= (-1)**(elem_left*i_occupation)
        vector_local.insert(i,i_occupation)
        del vector_local[i_index+1]
    return vector_local,pf

def permute_det(det,permutation):
    n_dim = len(det)-1
    n_half = n_dim//2 
    pf = det[0]
    vector_alpha = det[1:n_half+1]
    vector_beta = det[n_half+1:]
    vector_alpha_permuted,pf_alpha = permute_vector(vector_alpha,permutation)
    vector_beta_permuted,pf_beta = permute_vector(vector_beta,permutation)
    new_det = [pf*pf_alpha*pf_beta] + vector_alpha_permuted + vector_beta_permuted
    return new_det
    
def permute_csf(list_dets,permutation):
    list_dets_local = list_dets.copy()
    for i,det in enumerate(list_dets_local):
        new_det = permute_det(det,permutation)
        list_dets_local[i] = new_det
    return list_dets_local


if __name__ == "__main__":
    pf,vector = permute_vector([1,0,0,1],[1,2,3,0])
    print(pf,vector)
    new_det = permute_det([-1.0,1,0,0,1,0,1,1,0],[1,2,3,0])
    print(new_det)

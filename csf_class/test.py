from state_class import mc_state
import numpy as np


x = mc_state([[1.0,1,0,0,1],[1.0,1,0,0,1]],[0.5,0.5])

x = mc_state([[1.0,1,0,1,0]],[1])

print(x.norm())
print(x.normalize(1e-10))
print(x.dets,x.coeffs)






hf_coeffs = np.array([[ 0.52490465,  1.64273873],[ 0.52490465, -1.64273873]])


lowdin_coeffs = np.array([[ 1.53275533, -0.79042806],[-0.79042806,  1.53275533]])

overlap = np.array([[1.,         0.81471809],[0.81471809, 1.        ]])
x.change_basis(hf_coeffs,lowdin_coeffs,overlap,1e-10)
print(x.dets,x.coeffs)

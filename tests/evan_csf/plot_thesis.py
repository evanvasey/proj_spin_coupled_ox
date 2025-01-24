import numpy as np 
import matplotlib.pyplot as plt
#from plot import coeff_h4

plt.rcParams["font.size"] = 14
plt.rcParams["savefig.bbox"]="tight"

H_energy = -0.46658185


#coeff_h4("data/H4_curve.data")
LC = np.genfromtxt("data/H4_curve.data")
rhf = np.genfromtxt("data/H4_curve_rhf.data")
csf = np.genfromtxt("data/H4_curve_csf.data")
sloc = np.genfromtxt("data/H4_curve_sloc.data")
fci = np.genfromtxt("data/H4_FCI.data")


x = LC[:,0]
n_H = 4
plt.plot(x,np.ones((x.shape))*H_energy*n_H,color="purple")
plt.plot(LC[:,0],LC[:,1],label="$\Phi_{LC}$",color="blue")
plt.plot(rhf[:,0],rhf[:,1],label="$\Phi_{RHF}$",color="green")
plt.plot(csf[:,0],csf[:,1],label="$\Phi_{2}$",color="red")
plt.plot(sloc[:,0],sloc[:,1],label="$\Phi_{4}$",color="orange")
plt.plot(fci[:,0],fci[:,1],"--",linewidth=2,alpha=0.5,color="black",label="FCI")
plt.xlim((0.7,3.5))
plt.ylim((-2.2,-1.0))
plt.xlabel(r"Bond length [$\AA$]")
plt.ylabel(r"Energy [Ha]")

plt.legend()
plt.savefig("thesis_ecurve.pdf")
plt.clf()


coeff = np.genfromtxt("H4_report_coeffs.data")
coeff = np.genfromtxt("data/H4_curve_coeffs.data")
coeff = np.abs(coeff)
plt.plot(coeff[:,0],coeff[:,1],label="$\Phi_{RHF}$",color="green")
plt.plot(coeff[:,0],coeff[:,3],label="$\Phi_{2}$",color="red")
plt.plot(coeff[:,0],coeff[:,5],label="$\Phi_{4}$",color="orange")
plt.xlabel(r"Bond length [$\AA$]")
plt.ylabel(r"Coefficient")
plt.legend()
plt.savefig("thesis_coeffcurve.pdf")
plt.show()

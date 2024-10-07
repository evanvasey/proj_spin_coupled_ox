import numpy as np 
import matplotlib.pyplot as plt 


# H atom energy with sto-3G and pyscf intor methods
H_energy = -0.46658185


plt.rcParams["font.size"] = 14
plt.rcParams["savefig.bbox"]="tight"

def coeff_h4(filename):
    data = np.genfromtxt(filename)
    data = np.abs(data)
    r = data[:,0]
    rhf = (data[:,1] + data[:,2])/2
    csf = (data[:,3] + data[:,4])/2
    sloc = (data[:,5] + data[:,6] + data[:,7] + data[:,8])/4
    new_data = np.column_stack((r,rhf,sloc,csf))
    np.savetxt("H4_report_coeffs.data",new_data)
    return

def plot_E_curve(input_file,output_file,labels,xlim,ylim,n_H=None,true_wavefunction=None,true_wavefunction_index=None):
    data = np.genfromtxt(input_file)
    x = data[:,0]
    for column,legend in zip(data[:,1:].T,labels):
        plt.plot(x,column,label=legend)
    if n_H != None:
        global H_energy

        plt.plot(x,np.ones((x.shape))*H_energy*n_H)
    if true_wavefunction is not None:
       twf = np.genfromtxt(true_wavefunction)
       plt.plot(twf[:,0],twf[:,true_wavefunction_index],"--",alpha=0.5,color="black",label="FCI")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r"Bond length [$\AA$]")
    plt.ylabel(r"Energy [Ha]")
    plt.savefig(output_file)
    plt.show()
    return
def plot_ci_curve(input_file,output_file,labels):
    data = np.genfromtxt(input_file)
    x = data[:,0]
    for column,legend in zip(data[:,1:].T,labels):
        plt.plot(x,np.abs(column),label=legend)
    plt.legend()
    plt.xlabel(r"Bond length [$\AA$]")
    plt.ylabel(r"Coefficients")
    plt.savefig(output_file)
    plt.show()
    return
    

coeff_h4("H4_curve_coeffs.data")

#plot_E_curve("H2_curve.data","H2_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$"],(0.5,3.5),(-1.15,-0.6),n_H=2,true_wavefunction="H2_FCI.data",true_wavefunction_index=1)
#plot_ci_curve("H2_curve_coeffs.data","H2_curve_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$"])

plot_E_curve("H4_curve.data","H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.7,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("H4_curve_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])

plot_E_curve("H4_report.data","H4_report.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$",r"$\Phi_4$"],(0.7,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("H4_report_coeffs.data","H4_report_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$",r"$\Phi_4$"])


plot_E_curve("H6_curve.data","H6_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.5,3.5),(-3.5,-1.0),n_H=6,true_wavefunction="H6_backupFCI.data",true_wavefunction_index=1)
#plot_ci_curve("H4_curve_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])
plot_E_curve("H4_curve.data","H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.5,3.5),(-2.2,-1.0),n_H=4,true_wavefunction="H4_FCI.data",true_wavefunction_index=1)
plot_ci_curve("H4_curve_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])
plot_E_curve("H4_curve_hf.data","H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.5,3.5),(-2.2,-1.0),n_H=4)
plot_ci_curve("H4_curve_hf_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])
plot_E_curve("H4_curve_permute.data","H4_curve_permute.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.5,3.5),(-2.2,-1.0),n_H=4)
plot_ci_curve("H4_curve_permute_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])
plot_E_curve("H4_curve_nohf.data","H4_curve_nohf.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF1}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"],(0.5,3.5),(-2.2,-1.0),n_H=4)
plot_ci_curve("H4_curve_nohf_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$(permute)",r"$\sigma_L$",r"$\sigma_R$",r"$\sigma_T$",r"$\sigma_B$"])
#plot_E_curve("H2_curve.data","H2_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$"],(0.5,3.5),(-1.15,-0.6),n_H=2)
#plot_ci_curve("H2_curve_coeffs.data","H2_curve_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$"])
exit()
plot_E_curve("H4_curve.data","H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$",r"$\Phi_2$"],(0.5,3.5),(-2.2,-1.0),n_H=4)
plot_ci_curve("H4_curve_coeffs.data","H4_curve_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$"])
plot_E_curve("H2_curve.data","H2_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$"],(0.5,3.5),(-1.15,-0.6),n_H=2)
plot_ci_curve("H2_curve_coeffs.data","H2_curve_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$"])
exit()
#plot_E_curve("H4_curve.data","H4_curve.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$"],(0.5,5.5),(-5,5),n_H=4)
plot_E_curve("H4_curve_METHOD1.data","H4_curve_METHOD1.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$"],(0.5,3.5),(-2.2,-1.1),n_H=4)
plot_ci_curve("H4_curve_METHOD1_coeffs.data","H4_curve_METHOD1_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$"])
plot_E_curve("H4_curve_METHOD2.data","H4_curve_METHOD2.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$"],(0.5,3.5),(-2.2,-1.1),n_H=4)
plot_ci_curve("H4_curve_METHOD2_coeffs.data","H4_curve_METHOD2_coeffs.pdf",[r"$\Phi_{RHF1}$",r"$\Phi_{RHF2}$",r"$\Phi_2$",r"$\Phi_2$"])
plot_E_curve("H2_curve_METHOD1.data","H2_curve_METHOD1.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$"],(0.5,5.5),(-1.15,-0.6),n_H=2)
plot_ci_curve("H2_curve_METHOD1_coeffs.data","H2_curve_METHOD1_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$"])
plot_E_curve("H2_curve_METHOD2.data","H2_curve_METHOD2.pdf",[r"$\Phi_{LC}$",r"$\Phi_{RHF}$",r"$\Phi_2$"],(0.5,5.5),(-1.15,-0.6),n_H=2)
plot_ci_curve("H2_curve_METHOD2_coeffs.data","H2_curve_METHOD2_coeffs.pdf",[r"$\Phi_{RHF}$",r"$\Phi_2$"])

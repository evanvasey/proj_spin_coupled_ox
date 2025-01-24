import numpy as np 
import matplotlib
matplotlib.use('Agg')
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
    #plt.show()
    plt.clf()
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
    #plt.show()
    plt.clf()
    return
    


def plot_rhf_states(list_filenames):
    for filename in list_filenames:
        data = np.genfromtxt(filename)
        plt.plot(data[:,0],data[:,1])

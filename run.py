import psi4
import numpy as np
np.set_printoptions(precision=15, linewidth=200, threshold=200, suppress=True)
import matplotlib.pyplot as plt
import time
import sys
from pymc import pymc

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

f = open('out.txt', 'w')
orig_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

be = """
Be
symmetry c1
"""

ne = """
Ne
symmetry c1
"""

# From Crawford-group programming tutorials
h2o = """
O  0.000000000000  -0.143225816552   0.000000000000
H  1.638036840407   1.136548822547  -0.000000000000
H -1.638036840407   1.136548822547  -0.000000000000
units bohr
"""

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'diis': 1})
mol = psi4.geometry(be)
enuc = mol.nuclear_repulsion_energy()
escf, scf_wfn = psi4.energy('SCF', return_wfn=True)

fciqmc = pymc(scf_wfn)
steps = 7000
dt = 0.01
zeta = 0.1
E_freq = 100
fciqmc.propagate(steps, dt)

sys.stdout = orig_stdout
f.close()

#plt.figure(1)
#plt.plot(Nw)
#plt.title('Number of Walkers per Iteration')
#plt.xlabel('iterations')
#plt.ylabel('number of walkers')
#plt.savefig('walkers.png')
#plt.show()


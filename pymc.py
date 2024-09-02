import psi4
import numpy as np
from opt_einsum import contract
from scipy.special import comb
rng = np.random.default_rng()
from helper_CI import Determinant, HamiltonianGenerator
import time

class pymc(object):

    def __init__(self, scf_wfn):

        time_init = time.time()

        ## spatial orbitals
        nirrep = scf_wfn.nirrep()
        frzpi = np.array(list(scf_wfn.frzcpi().to_tuple()))
        doccpi = np.array(list(scf_wfn.doccpi().to_tuple()))
        nmopi = np.array(list(scf_wfn.nmopi().to_tuple()))
        uoccpi = nmopi - doccpi - frzpi

        nmo = sum(nmopi)
        no = sum(doccpi)
        nv = sum(uoccpi)
        nact = no + nv
        nfzc = sum(frzpi)

        ## AO-->MO transformations

        mints = psi4.core.MintsHelper(scf_wfn.basisset())
        C = scf_wfn.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)

        spin_ind = np.arange(2*C.shape[1], dtype=int) % 2

        # Fock matrix (spin orbitals)
        F = npC.T @ scf_wfn.Fa_subset("AO").np @ npC
        F = np.repeat(F, 2, axis=0)
        F = np.repeat(F, 2, axis=1)
        F *= (spin_ind.reshape(-1, 1) == spin_ind)
        self.F = F

        # One-electron integrals (spin orbitals)
        h = npC.T @ (mints.ao_potential().np + mints.ao_kinetic().np) @ npC
        h = np.repeat(h, 2, axis=0)
        h = np.repeat(h, 2, axis=1)
        h *= (spin_ind.reshape(-1, 1) == spin_ind)
        self.h = h

        # Two-electron integrals (spin orbitals)
        ERI = mints.mo_spin_eri(C, C).np  # <pr||qs>
        self.ERI = ERI

        o = slice(0, 2*no)
        v = slice(2*no, 2*nmo)

        self.nmo = nmo
        self.no = no
        self.nv = nv
        self.o = o
        self.v = v
        self.E0 = scf_wfn.energy() - scf_wfn.molecule().nuclear_repulsion_energy()

        ## Set up struct for walkers
        self.wstruct = [('alpha_bits', int), ('beta_bits', int), ('sign', int), ('energy', float)]

        self.Hamiltonian = HamiltonianGenerator(h, ERI)

        ## Set up probability cutoffs

        # Number of singles and doubles
        n_A = n_B = no * nv
        n_s = n_A + n_B
        n_AA = n_BB = comb(no,2) * comb(nv,2)
        n_AB = no * no * nv * nv
        n_d = n_AA + n_BB + n_AB
        n_total = n_s + n_d

        ## Spawning probabilities
        # Singles
        self.P_s = n_s/n_total
        p_gen_i = 1/(2*no)
        self.p_gen_ia = self.P_s * p_gen_i * (1/nv) # Maintains spin

        # Doubles
        self.P_d = n_d/n_total
        p_gen_ij = 1/comb(2*no,2)
        self.p_gen_IJAB = self.p_gen_ijab = self.P_d * 2 * p_gen_ij * 1/(nv*(nv-1))
        self.p_gen_IjAb = self.P_d * 2 * p_gen_ij * (1/(nv*nv))

        # Probability cutoffs for selecting specific excitations for a given parent determinant
        # 0...p(singles_A)...p(singles_B)...p(doubles_AA)... p(doubles_BB)...p(doubles_AB)=1
        self.p_A = n_A/n_total
        self.p_B = self.p_A + n_B/n_total
        self.p_AA = self.p_B + n_AA/n_total
        self.p_BB = self.p_AA + n_BB/n_total
        self.p_AB = self.p_BB + n_AB/n_total

        #print("P_s = ", P_s, "P_d = ", P_d)
        #print("p_gen_ia = ", p_gen_ia)
        #print("p_gen_IJAB = ", p_gen_IJAB, "p_gen_IjAb = ", p_gen_IjAb)
        #print(p_A, p_B, p_AA, p_BB, p_AB)

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (nmo, nact, no, nv))
        print("PyMC object initialized in %.3f seconds." % (time.time() - time_init))


    # Return lists of the occupied and virtual orbitals in a given alpha-/beta-string
    def find_occvir(self, D):
        num_bits = self.nmo
        occ = []; vir = []
        for i in range(num_bits):
            if D%2==0:
                vir.append(i)
            else:
                occ.append(i)
            D >>= 1

        return np.array(occ), np.array(vir)


    # Spawn new walker
    def spawn(self, walkers, dt):

        spawned = np.empty([0], dtype=self.wstruct)

        for w in walkers:
            p_rand = rng.random()

            if p_rand < self.p_A:
                target_det = 'A'
            elif p_rand < self.p_B:
                target_det = 'B'
            elif p_rand < self.p_AA:
                target_det = 'AA'
            elif p_rand < self.p_BB:
                target_det = 'BB'
            elif p_rand < self.p_AB:
                target_det = 'AB'

            w_a = w[0]
            w_b = w[1]
            aocc, avir = self.find_occvir(w_a)
            bocc, bvir = self.find_occvir(w_b)

            if target_det == 'AB':
                i = np.random.choice(aocc, 1)[0]
                j = np.random.choice(bocc, 1)[0]
                a = np.random.choice(avir, 1)[0]
                b = np.random.choice(bvir, 1)[0]
                det_a = (2**i + 2**a) ^ w_a
                det_b = (2**j + 2**b) ^ w_b
                p_gen = self.p_gen_IjAb

            elif target_det == 'AA':
                ij = np.random.choice(aocc, 2, replace=False)
                i = ij[0]; j = ij[1]
                ab = np.random.choice(avir, 2, replace=False)
                a = ab[0]; b = ab[1]
                det_a = (2**i + 2**a + 2**j + 2**b) ^ w_a
                det_b = w_b
                p_gen = self.p_gen_IJAB

            elif target_det == 'BB':
                ij = np.random.choice(bocc, 2, replace=False)
                i = ij[0]; j = ij[1]
                ab = np.random.choice(bvir, 2, replace=False)
                a = ab[0]; b = ab[1]
                det_a = w_a
                det_b = (2**i + 2**a + 2**j + 2**b) ^ w_b
                p_gen = self.p_gen_ijab

            elif target_det == 'A':
                i = np.random.choice(aocc, 1)[0]
                a = np.random.choice(avir, 1)[0]
                det_a = (2**i + 2**a) ^ w_a
                det_b = w_b
                p_gen = self.p_gen_ia

            elif target_det == 'B':
                i = np.random.choice(bocc, 1)[0]
                a = np.random.choice(bvir, 1)[0]
                det_a = w_a
                det_b = (2**i + 2**a) ^ w_b
                p_gen = self.p_gen_ia

            det_new = Determinant(det_a, det_b)
            det_parent = Determinant(w_a, w_b)
            Kij = self.Hamiltonian.calcMatrixElement(det_new,det_parent)

            p_spawn = dt * abs(Kij)/p_gen

            #print('\nTarget:', target_det, "Parent:", det_parent, "Spawn:", det_new, "Kij:", Kij, "p_gen:", p_gen, "p_spawn:", p_spawn, "p_rand:", p_rand)

            if p_spawn > p_rand:

                if Kij < 0:
                    sign = w[2] # sign of parent walker
                else:
                    sign = -1 * w[2]

                E = self.Hamiltonian.calcMatrixElement(det_new, det_new) - self.E0

                spawn = np.array([(det_a, det_b, sign, E)], dtype=self.wstruct)

                spawned = np.append(spawned, spawn)

        return spawned

    # Kill or clone parents
    def death(self, walkers, dt, S):
        dead = np.empty([0], dtype=self.wstruct)
        cloned = np.empty([0], dtype=self.wstruct)

        keep = np.ones(walkers.shape[0], dtype=bool) # mask for keeping/killing walkers
        for pos, w in enumerate(walkers):
            p_step = rng.random()
            p_d = dt * (w[3] - S)
            if abs(p_d) > p_step:
                if p_d > 0:
                    keep[pos] = False
                    dead = np.append(dead, w)
                    #print('\nDeath:', w)
                    #print(p_step, p_d)
                elif p_d < 0:
                    cloned = np.append(cloned, w)
                    #print('\nCloned:', w)
                    #print(p_step, p_d)

        # Delete dead walkers
        walkers = walkers[keep]
        # Append cloned walkers
        walkers = np.append(walkers, cloned)

        return walkers, dead, cloned


    # Annihilate opposite-signed walkers from a given list
    def annihilation(self, walkers):

        if len(walkers) < 2:
            return(walkers, 0)

        initial_length = len(walkers)
        walkers = np.sort(walkers)

        # Find sequences of walkers on identical determinants
        walker_sets = []
        sequence_started = 0
        for i in range(len(walkers)-1):
            if walkers[i][0] == walkers[i+1][0] and walkers[i][1] == walkers[i+1][1]:
                if sequence_started == 0:
                    sequence_started = 1
                    sequence_start_pos = i
                    sequence_length = 1
                else:
                    sequence_length += 1
            else:
                if sequence_started == 1:
                    walker_sets.append([sequence_start_pos, sequence_length])
                sequence_started = 0

        if sequence_started == 1:
            walker_sets.append([sequence_start_pos, sequence_length])

        # Replace walkers on identical determinants 
        replacement_walker_sets = []
        for set in reversed(walker_sets):
            this_set = walkers[set[0]:(set[0]+set[1])]

            sum = 0
            for walker in this_set:
                sum += walker[2]

            replacement_walker = np.array((this_set[0][0], this_set[0][1], np.sign(sum), this_set[0][3]), dtype=self.wstruct)

            # Delete this section of the input walker list and replace with survivors, if necessary
            walkers = np.delete(walkers, slice(set[0], set[0]+set[1]))
            replacement_set = np.empty([0], dtype=self.wstruct)
            for i in range(abs(sum)):
                replacement_set = np.append(replacement_set, replacement_walker)

            if sum != 0:
                walkers = np.insert(walkers, set[0], replacement_set, axis=0)

        final_length = len(walkers)

        return(walkers)


    # Find number of walkers matching a given Determinant (assumes walkers are already sorted)
    def find_matching_walkers(self, det, walkers):
        det_a = det.alphaObtBits; det_b = det.betaObtBits
        n = 0
        found = 0
        for w in walkers:
            if det_a == w[0] and det_b == w[1]:
                n += 1
                found = 1
            else:
                if found == 1:
                    return n
                n = 0

        if found == 1: # We ended the list on a match
            return n

        # No match found
        return 0


    def projected_energy(self, walkers, ref_det, sd_list, E_sd_list):
        walkers = np.sort(walkers)

        N0 = self.find_matching_walkers(ref_det, walkers)
        if N0 == 0:
            raise Exception("No walkers on the reference??")

        E = 0.0
        n = 0
        # find number (n) of det in walkers and add n*<0|H|det> to energy
        for idx, det in enumerate(sd_list):
            n = self.find_matching_walkers(det, walkers)
            E += E_sd_list[idx] * n/N0

        return E


    def propagate(self, steps=1000, dt=0.001, zeta=0.1, A=10, E_freq=100):

        S = 0.0
        E0 = self.E0
        no = self.no
        wstruct = self.wstruct

        # Generate initial set of walkers on the reference
        N0_init = 10
        walkers = np.empty([0], dtype=wstruct)
        for w in range(N0_init):
            walkers = np.append(walkers, np.array((2**no-1, 2**no-1, 1, 0.0), dtype=wstruct))

        # Prepare list of singles and doubles relative to the reference and pre-compute their matrix elements with the reference
        ref_det = Determinant(alphaObtBits=2**no-1, betaObtBits=2**no-1)
        sd_list = ref_det.generateSingleAndDoubleExcitationsOfDet(self.nmo)
        H_sd_list = []
        for sd_det in sd_list:
            H_sd_list.append(self.Hamiltonian.calcMatrixElement(sd_det, ref_det))

        Nw = []
        nspawned = 0
        ndead = 0
        ncloned = 0
        t = time.time()

        print("Starting iterative QMC sequence...")
        for i in range(steps):

            t_iter = time.time()

            Nw.append(len(walkers))

#            if i >= plateau and i % A == 0: # adjust S every A steps
#                S = S - (zeta/(A*dt)) * np.emath.log(Nw[i]/(Nw[i - A]))
            if i % E_freq == 0: # calculate E every E_freq steps
                E = self.projected_energy(walkers, ref_det, sd_list, H_sd_list)

            # Spawning
            spawned = self.spawn(walkers, dt)
            nspawned += len(spawned)

            # Diagonal Death/Cloning
            walkers, dead, cloned = self.death(walkers, dt, S)
            ndead += len(dead)
            ncloned += len(cloned)

            # Annihilation
            walkers = np.append(walkers, spawned)
            nw = len(walkers)
            walkers = self.annihilation(walkers)
            nw_annihilated = nw - len(walkers)


            print(f"Iter = {i:d} #Walkers = {Nw[i]:d} #Spawned = {len(spawned):d} #Killed = {len(dead):d} #Annihilated = {nw_annihilated:d} S = {S:f} E = {E:f} Iter Time = {time.time()-t_iter:.3f}")

        print('Total Spawned:', nspawned)
        print('Total Killed:', ndead)
        print('Total Cloned:', ncloned)
        print(f"\nTotal time for QMC iterations: {(time.time()-t)/60:.3f} minutes.\n")


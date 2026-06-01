import numpy as np
import copy

class Phonon:
    def __init__(
            self, int1e: np.ndarray, int2e: np.ndarray, 
            norb: int, nelec: int, dm_calc: callable
        ):

        self.int1e = int1e
        self.int2e = int2e
        self.norb = norb
        self.nelec = nelec
        self.dm_calc = dm_calc
        self.int1e_save = copy.deepcopy(int1e)
        self.int2e_save = copy.deepcopy(int2e)

        self.K1 = 10
        self.K2 = 10
        self.do_print = False
        self.use_multiplicative_bonding = True
        self.use_negative_transfer_notation = False
        self.use_positive_transfer = False
        self.use_test_impl = False
    
    def init_params(self):
        self.bonding = np.ones_like(self.int1e)
        self.site_potential = np.ones((self.norb))
        return

    def oneshot(self, pressure: float= 0.0):
        self.init_params()
        dm = self.dm_calc(self.int1e, self.int2e, self.norb, self.nelec)
        for i in range(dm.shape[0]):
            for j in range(dm.shape[1]):
                if i != j:
                    if self.use_multiplicative_bonding:
                        tdm = dm[i, j] * self.int1e[i, j]
                    else:
                        tdm = dm[i, j]
                    if self.use_negative_transfer_notation:
                        tdm *= -1
                    coeff = [self.K1, tdm, 0, -pressure]
                    roots = np.roots(coeff)
                    if self.do_print:
                        print(f"roots @ {i} {j}=", roots)
                    if self.use_test_impl:
                        if tdm < 0:
                            self.use_positive_transfer = False
                        else:
                            self.use_positive_transfer = True
                        if self.use_positive_transfer:
                            max_root_idx = np.argmax(roots.real)
                            max_root = roots[max_root_idx]
                            imag_roots = np.iscomplex(roots[max_root_idx])
                            if self.do_print:
                                print(f"max_root @ {i} {j}=", max_root, "imaginary part =", imag_roots)
                                print(f"tdm @ {i} {j}=", tdm)
                            self.bonding[i, j] = max_root
                        else:
                            min_root_idx = np.argmin(roots.real)
                            min_root = roots[min_root_idx]
                            imag_roots = np.iscomplex(roots[min_root_idx])
                            if self.do_print:
                                print(f"min_root @ {i} {j}=", min_root, "imaginary part =", imag_roots)
                                print(f"tdm @ {i} {j}=", tdm)
                            self.bonding[i, j] = min_root
                    else:
                        max_root_idx = np.argmax(roots.real)
                        max_root = roots[max_root_idx].real
                        imag_roots = np.iscomplex(roots[max_root_idx])
                        if self.do_print:
                            print(f"max_root @ {i} {j}=", max_root, "imaginary part =", imag_roots)
                            print(f"tdm @ {i} {j}=", tdm)
                        self.bonding[i, j] = max_root
            self.site_potential = np.diag(dm) / self.K2
        return

    def run(self, pressure: float=0.0) -> np.ndarray:
        self.oneshot(pressure=pressure)
        self.int1e = self.int1e * self.bonding
        self.int1e -= np.diag(self.site_potential) 
        return self.int1e
    
    def clear(self):
        self.int1e = copy.deepcopy(self.int1e_save)
        self.int2e = copy.deepcopy(self.int2e_save)
        return
    

from pyscf import fci

def dm_calculator(int1e: np.ndarray, int2e: np.ndarray, norb: int, nelec: int) -> np.ndarray:
    cis = fci.direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1 = cis.make_rdm1(c, norb, nelec)
    # print("dm1=", dm1)
    return dm1

if __name__ == "__main__":
    norb = 6
    nelec = 6
    int1e = np.zeros((norb, norb))
    int2e = np.zeros((norb, norb, norb, norb))
    t1 = 1.0
    V = 1.2
    U = 3

    for i in range(norb):
        int1e[i, (i+1)%norb] = t1
        int1e[(i+1)%norb, i] = t1
        int2e[i, i, (i+1)%norb, (i+1)%norb] = V
        int2e[(i+1)%norb, (i+1)%norb, i, i] = V
        int2e[i, i, i, i] = U


    # print(int1e)

    ph = Phonon(int1e, int2e, norb, nelec, dm_calculator)
    ph.run(pressure=5.)
    # ph.clear()

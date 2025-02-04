import unittest
import torch
import math
import random
import numpy as np
from vpnet import *

def phi_diff(Phi: torch.Tensor, params: torch.Tensor):
    '''Autograd based function differentiation.'''
    num_coeffs, num_samples = Phi.shape
    num_params = params.size(0)
    dPhi = torch.zeros((num_params, num_coeffs, num_samples), 
                        dtype=params.dtype, device=params.device)
    for j in range(0, num_coeffs):
        for k in range(0, num_samples):
            params.grad = None # abuse autograd
            Phi[j, k].backward(retain_graph=True)
            dPhi[:, j, k] = params.grad
    return dPhi

def polyval(p, x):
    '''Polynomial evaluation using Horner's method.'''
    y = torch.tensor(0, dtype=x.dtype, device=x.device)
    for c in p:
        y = y * x + c
    return y

# Coefficients of Hermite polynomials
hermite_poly = [
    [1],
    [2, 0],
    [4, 0, -2],
    [8, 0, -12, 0],
    [16, 0, -48, 0, 12],
    [32, 0, -160, 0, 120, 0],
    [64, 0, -480, 0, 720, 0, -120],
    [128, 0, -1344, 0, 3360, 0, -1680, 0],
    [256, 0, -3584, 0, 13440, 0, -13440, 0, 1680],
    [512, 0, -9216, 0, 48384, 0, -80640, 0, 30240, 0],
]

def hermite_explicit(num_coeffs: int, num_samples: int, params: torch.Tensor) -> torch.Tensor:
    '''Explicit computation of Hermite functions.'''
    m2 = num_samples // 2
    t = torch.arange(-m2, m2 + 1, dtype=params.dtype, device=params.device) if num_samples % 2 else \
        torch.arange(-m2, m2, dtype=params.dtype, device=params.device)
    x = params[0] * (t - params[1] * m2)
    w = torch.exp(-0.5 * x ** 2)
    Phi = torch.zeros((num_coeffs, num_samples), dtype=params.dtype, device=params.device)
    f = 1
    for j in range(num_coeffs):
        if j > 1:
            f *= j
        Phi[j, :] = torch.sqrt(params[0]) * polyval(hermite_poly[j], x) * w / math.sqrt(2 ** j * f * math.sqrt(math.pi))
    return Phi

def hermite2_explicit(num_coeffs: int, num_samples: int, params: torch.Tensor) -> torch.Tensor:
    '''Explicit computation of Hermite functions.'''
    m2 = num_samples // 2
    t = torch.arange(-m2, m2 + 1, dtype=params.dtype, device=params.device) if num_samples % 2 else \
        torch.arange(-m2, m2, dtype=params.dtype, device=params.device)
    x = params[0] ** 2 * (t - params[1] * m2)
    w = torch.exp(-0.5 * x ** 2)
    Phi = torch.zeros((num_coeffs, num_samples), dtype=params.dtype, device=params.device)
    f = 1
    for j in range(num_coeffs):
        if j > 1:
            f *= j
        Phi[j, :] = params[0] * polyval(hermite_poly[j], x) * w / math.sqrt(2 ** j * f * math.sqrt(math.pi))
    return Phi

class TestHermiteSystem(unittest.TestCase):
    def setUp(self):
        # Test systems
        self.test_systems = 4
        self.num_samples = [5, 42, 100, 101]
        self.num_coeffs  = [2, 3, 10, 10]
        self.fun_systems = []
        for ii in range(self.test_systems):
            self.fun_systems.append(HermiteSystem(self.num_samples[ii], self.num_coeffs[ii]))
        # Test parameters
        self.test_params = 4
        self.dilation = [0.1, 0.1, 1, 1]
        self.translation = [0, 0.5, 0, 0.5]

    def test_fun_system(self):
        '''Validation of Phi and dPhi of Hermite system.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            for ii in range(self.test_systems):
                for jj in range(self.test_params):
                    with self.subTest(num_samples=self.num_samples[ii], num_coeffs=self.num_coeffs[ii], \
                                      dilation=self.dilation[jj], translation=self.translation[jj], \
                                      dtype=str(dtype), device=str(device)):
                        params = torch.tensor([self.dilation[jj], self.translation[jj]], \
                                               dtype=dtype, device=device, requires_grad=True)
                        Phi, dPhi = self.fun_systems[ii](params.detach())
                        # reference computation
                        Phi0 = hermite_explicit(self.num_coeffs[ii], self.num_samples[ii], params)
                        self.assertTrue(torch.allclose(Phi0, Phi))
                        dPhi0 = phi_diff(Phi0, params)
                        self.assertTrue(torch.allclose(dPhi0, dPhi))

class TestHermiteSystem2(unittest.TestCase):
    def setUp(self):
        # Test systems
        self.test_systems = 4
        self.num_samples = [5, 42, 100, 101]
        self.num_coeffs  = [2, 3, 10, 10]
        self.fun_systems = []
        for ii in range(self.test_systems):
            self.fun_systems.append(HermiteSystem2(self.num_samples[ii], self.num_coeffs[ii]))
        # Test parameters
        self.test_params = 4
        self.dilation = [0.1, 0.1, 1, 1]
        self.translation = [0, 0.5, 0, 0.5]

    def test_fun_system(self):
        '''Validation of Phi and dPhi of HermiteV2 system.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            for ii in range(self.test_systems):
                for jj in range(self.test_params):
                    with self.subTest(num_samples=self.num_samples[ii], num_coeffs=self.num_coeffs[ii], \
                                      dilation=self.dilation[jj], translation=self.translation[jj], \
                                      dtype=str(dtype), device=str(device)):
                        params = torch.tensor([self.dilation[jj], self.translation[jj]], \
                                               dtype=dtype, device=device, requires_grad=True)
                        Phi, dPhi = self.fun_systems[ii](params.detach())
                        # reference computation
                        Phi0 = hermite2_explicit(self.num_coeffs[ii], self.num_samples[ii], params)
                        self.assertTrue(torch.allclose(Phi0, Phi))
                        dPhi0 = phi_diff(Phi0, params)
                        self.assertTrue(torch.allclose(dPhi0, dPhi))

class TestRealMTSystem(unittest.TestCase):
    def setUp(self):
        # Test systems
        self.test_systems = 8
        self.num_samples = [5, 5, 42, 42, 100, 100, 101, 101]
        self.mults = [[1], [4], [4, 3], [1, 5], [4, 3, 2], [5, 1, 8], [2, 2, 2], [3, 4, 1]]
        self.fun_systems = []
        for ii in range(self.test_systems):
            self.fun_systems.append(RealMTSystem(self.num_samples[ii], self.mults[ii]))

    def test_fun_system(self):
        '''Validation of dPhi of real MT system.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            for ii in range(self.test_systems):
                params = torch.rand(2 * len(self.mults[ii]), dtype=dtype, device=device)
                params[1::2] = (2 * params[1::2] - 1) * math.pi
                params.requires_grad_()
                with self.subTest(num_samples=self.num_samples[ii], mult=self.mults[ii], params=params, \
                                  dtype=str(dtype), device=str(device)):
                    Phi, dPhi = self.fun_systems[ii](params)
                    # reference diff
                    dPhi0 = phi_diff(Phi, params)
                    self.assertTrue(torch.allclose(dPhi0, dPhi))

class TestMultiSystem(unittest.TestCase):
    def setUp(self):
        # Test systems
        num_samples = 100
        self.fun_system = MultiSystem(
            HermiteSystem(num_samples, 4),
            HermiteSystem(num_samples, 2),
            HermiteSystem(num_samples, 5),
        )
        # Test parameters
        self.params = [0.5, 0, 0.3, 0.5, 1.0, 0.5]

    def test_fun_system(self):
        '''Validation of Multi-Hermite systems.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            params = torch.tensor(self.params, dtype=dtype, device=device, requires_grad=True)
            Phi, dPhi = self.fun_system(params.detach())
            # reference computation
            c_params = 0
            c_coeffs = 0
            for f in self.fun_system.fun_systems:
                p0 = params[c_params:c_params+f.num_params]
                p0.retain_grad()
                Phi0 = hermite_explicit(f.num_coeffs, f.num_samples, p0)
                self.assertTrue(torch.allclose(Phi0, Phi[c_coeffs:c_coeffs+f.num_coeffs, :]))
                dPhi0 = phi_diff(Phi0, p0)
                self.assertTrue(torch.allclose(dPhi0, dPhi[c_params:c_params+f.num_params, c_coeffs:c_coeffs+f.num_coeffs, :]))
                dPhi[c_params:c_params+f.num_params, c_coeffs:c_coeffs+f.num_coeffs, :] = 0
                c_params += f.num_params
                c_coeffs += f.num_coeffs
            self.assertTrue(torch.equal(torch.zeros_like(dPhi), dPhi))

class TestHermiteVP(unittest.TestCase):
    def setUp(self):
        self.batch = 10
        self.n_in = 100
        self.n_channels = 2
        self.n_vp = 8
        self.params_init = [1.0, 0.01]
        self.fun_system = HermiteSystem(self.n_in, self.n_vp)

    def test_vp(self):
        '''Hermite VP evaluation and size comparison.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            with self.subTest(device=str(device)):
                params_init = torch.tensor(self.params_init, requires_grad=True,
                                           dtype=dtype, device=device)
                x = torch.rand((self.batch, self.n_channels, self.n_in),
                                dtype=dtype, device=device, requires_grad=True)
                coeffs, x_hat, res, r2 = VPFun.apply(x, params_init, self.fun_system)
                loss = (coeffs ** 2).sum() + 0.1*(x_hat ** 2).sum() + 2*(res ** 2).sum() - r2.sum()
                loss.backward()
                self.assertTrue(x.shape == x.grad.shape)
                self.assertTrue(params_init.shape == params_init.grad.shape)
    
    def test_vp_gradcheck(self):
        '''Hermite VP gradcheck.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            with self.subTest(device=str(device)):
                params_init = torch.tensor(self.params_init, requires_grad=True,
                                           dtype=dtype, device=device)
                x = torch.rand((self.batch, self.n_channels, self.n_in),
                                dtype=dtype, device=device, requires_grad=True)
                torch.autograd.set_detect_anomaly(True)
                self.assertTrue(torch.autograd.gradcheck(VPFun.apply, (x, params_init, self.fun_system)))
                self.assertTrue(torch.autograd.gradcheck(VPIteration.apply, (x, params_init, self.fun_system)))

class TestRealMTVP(unittest.TestCase):
    def setUp(self):
        self.batch = 10
        self.n_in = 100
        self.n_channels = 2
        self.mults = [1, 2, 2]
        self.params_init = [0.9, 0, 0.6, -1, 0.7, 1]
        self.fun_system = RealMTSystem(self.n_in, self.mults)
    
    def test_vp(self):
        '''Real MT VP evaluation and size comparison.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            with self.subTest(device=str(device)):
                params_init = torch.tensor(self.params_init, requires_grad=True,
                                           dtype=dtype, device=device)
                x = torch.rand((self.batch, self.n_channels, self.n_in),
                                dtype=dtype, device=device, requires_grad=True)
                coeffs, x_hat, res, r2 = VPFun.apply(x, params_init, self.fun_system)
                loss = (coeffs ** 2).sum() + 0.1*(x_hat ** 2).sum() + 2*(res ** 2).sum() - r2.sum()
                loss.backward()
                self.assertTrue(x.shape == x.grad.shape)
                self.assertTrue(params_init.shape == params_init.grad.shape)
    
    def test_vp_gradcheck(self):
        '''Real MT VP gradcheck.'''
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        dtype = torch.double
        for device in devices:
            with self.subTest(device=str(device)):
                params_init = torch.tensor(self.params_init, requires_grad=True,
                                           dtype=dtype, device=device)
                x = torch.rand((self.batch, self.n_channels, self.n_in),
                                dtype=dtype, device=device, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(VPFun.apply, (x, params_init, self.fun_system)))
                self.assertTrue(torch.autograd.gradcheck(VPIteration.apply, (x, params_init, self.fun_system)))

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # torch.autograd.set_detect_anomaly(True)
    unittest.main(verbosity=2)

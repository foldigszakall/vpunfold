import math
import torch
from typing import Any, Callable

'''Function systems and Variable Projection operators.'''

class FunSystem:
    '''Abstract function system base class. Callable.'''
    def __init__(self, num_samples: int, num_coeffs: int, num_params: int) -> None:
        '''
        Constructs a function system.
        Subclasses should call this in their constructor as of
            super().__init__(num_samples, num_coeffs, num_params)
        
        Input:
            num_samples: int    Number of time sampling points of the input
            num_coeffs: int     Number of output coefficients (number of functions)
            num_params: int     Number of nonlinear system parameters
        '''
        self.num_samples = num_samples
        self.num_coeffs = num_coeffs
        self.num_params = num_params

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes the sampled function system and its derivatives with respect to
        the nonlinear system parameters. Subclasses should implement this method.
        
        Input:
            params: torch.Tensor    Tensor of nonlinear system parameters.
                                    Expected size: (num_params)
        Output:
            Phi: torch.Tensor       Tensor of sampled function system.
                                    Size: (num_coeffs, num_samples)
                                    Phi[i,j] represents the jth basic function
                                    sampled at the ith time instance.
            dPhi: torch.Tensor      Tensor of the function system derivatives.
                                    Size: (num_params, num_coeffs, num_samples)
                                    dPhi[p,i,j] represents the derivative of 
                                    the jth basic function with respect to the
                                    pth system parameter param[p] sampled at
                                    the ith time instance.
        '''
        raise NotImplementedError()

class HermiteSystem(FunSystem):
    '''
    Adaptive Hermite functions: classical Hermite functions parametrized
    by dilation and translation.
    '''
    def __init__(self, num_samples: int, num_coeffs: int) -> None:
        '''
        Construct an adaptive Hermite function system. System parameters
        consist of dilation and translation, i.e. property
        num_params = 2 fixed.
        
        See also FunSystem.__init__
        '''
        assert num_samples > 0
        assert num_coeffs > 1
        super().__init__(num_samples, num_coeffs, 2)

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes adaptive Hermite functions and derivatives. System parameters 
        params is expected to contain dilation and translation in this order.
        The functions are uniformly sampled.
        
        See also FunSystem.__call__
        '''
        dilation, translation = params[:2]
        m2 = self.num_samples // 2
        t = torch.arange(-m2, m2 + 1, dtype=params.dtype, device=params.device) if self.num_samples % 2 else \
            torch.arange(-m2, m2, dtype=params.dtype, device=params.device)
        x = dilation * (t - translation * m2)
        w = torch.exp(-0.5 * x ** 2)
        pi_sqrt = 1 / torch.sqrt(torch.sqrt(torch.tensor(math.pi, dtype=params.dtype, device=params.device)))
        n_sqrt = torch.sqrt(2 * torch.arange(0, self.num_coeffs, dtype=params.dtype, device=params.device))
        # stack to avoid gradient computation error on inplace modification
        Phi = self.num_coeffs * [None]
        dPhi = self.num_coeffs * [None]
        Phi[0] = torch.sqrt(dilation) * pi_sqrt * w
        dPhi[0] = -x * Phi[0]
        Phi[1] = n_sqrt[1] * x * Phi[0]
        dPhi[1] = -x * Phi[1] + n_sqrt[1] * Phi[0]
        for j in range(2, self.num_coeffs):
            Phi[j] = (2 * x * Phi[j - 1] - n_sqrt[j - 1] * Phi[j - 2]) / n_sqrt[j]
            dPhi[j] = -x * Phi[j] + n_sqrt[j] * Phi[j - 1]
        Phi = torch.stack(Phi)
        dPhi = torch.stack(dPhi)
        dPhi = torch.stack((dPhi * (t - translation * m2) + 0.5 * Phi / dilation, -dPhi * dilation * m2))
        return Phi, dPhi

    def __repr__(self):
        return f'HermiteSystem(num_samples={self.num_samples}, num_coeffs={self.num_coeffs})'

class HermiteSystem2(FunSystem):
    '''
    Adaptive Hermite functions with dilation parameters squared. See also HermiteSystem.
    '''
    def __init__(self, num_samples: int, num_coeffs: int) -> None:
        '''
        See HermiteSystem.__init__
        '''
        assert num_samples > 0
        assert num_coeffs > 1
        super().__init__(num_samples, num_coeffs, 2)

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes adaptive Hermite functions and derivatives. Dilation parameter
        is squared compared to HermiteSystem for numerical stability.
        
        See HermiteSystem.__call__
        '''
        dilation, translation = params[:2]
        m2 = self.num_samples // 2
        t = torch.arange(-m2, m2 + 1, dtype=params.dtype, device=params.device) if self.num_samples % 2 else \
            torch.arange(-m2, m2, dtype=params.dtype, device=params.device)
        x = dilation ** 2 * (t - translation * m2)
        w = torch.exp(-0.5 * x ** 2)
        pi_sqrt = 1 / torch.sqrt(torch.sqrt(torch.tensor(math.pi, dtype=params.dtype, device=params.device)))
        n_sqrt = torch.sqrt(2 * torch.arange(0, self.num_coeffs, dtype=params.dtype, device=params.device))
        # stack to avoid gradient computation error on inplace modification
        Phi = self.num_coeffs * [None]
        dPhi = self.num_coeffs * [None]
        Phi[0] = dilation * pi_sqrt * w
        dPhi[0] = -x * Phi[0]
        Phi[1] = n_sqrt[1] * x * Phi[0]
        dPhi[1] = -x * Phi[1] + n_sqrt[1] * Phi[0]
        for j in range(2, self.num_coeffs):
            Phi[j] = (2 * x * Phi[j - 1] - n_sqrt[j - 1] * Phi[j - 2]) / n_sqrt[j]
            dPhi[j] = -x * Phi[j] + n_sqrt[j] * Phi[j - 1]
        Phi = torch.stack(Phi)
        dPhi = torch.stack(dPhi)
        dPhi = torch.stack((dPhi * 2 * dilation * (t - translation * m2) + Phi / dilation,
                            -dPhi * dilation ** 2 * m2))
        return Phi, dPhi

    def __repr__(self):
        return f'HermiteSystem2(num_samples={self.num_samples}, num_coeffs={self.num_coeffs})'

class MultiSystem(FunSystem):
    '''
    Combination of multiple function systems.
    '''
    def __init__(self, *fun_systems: FunSystem) -> None:
        '''
        Construct the combination of a function system list.
        
        Input:
            *fun_systems: FunSystem     Variable arguments of function systems.
        See also FunSystem.__init__
        '''
        assert len(fun_systems) > 0
        num_samples = fun_systems[0].num_samples
        assert all([f.num_samples == num_samples for f in fun_systems])
        num_coeffs = sum([f.num_coeffs for f in fun_systems])
        num_params = sum([f.num_params for f in fun_systems])
        super().__init__(num_samples, num_coeffs, num_params)
        self.fun_systems = fun_systems
        self.num_systems = len(fun_systems)

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes and combines the functions and derivatives of the systems.
        Parameters are expected to be concatenated in the order of the systems.
        
        See also FunSystem.__call__
        '''
        c_params = 0
        Phi = self.num_systems * [None]
        dPhi = self.num_systems * [None]
        for i, f in enumerate(self.fun_systems):
            Phi_i, dPhi_i = f(params[c_params:c_params+f.num_params])
            Phi[i] = Phi_i
            dPhi[i] = dPhi_i.flatten(1)
            c_params += f.num_params
        Phi = torch.cat(Phi)
        dPhi = torch.block_diag(*dPhi).reshape(self.num_params, self.num_coeffs, self.num_samples)
        return Phi, dPhi

    def __repr__(self):
        return 'MultiSystem(\n' + ',\n'.join([repr(f) for f in self.fun_systems]) + '\n)'

class RealMTSystem(FunSystem):
    '''
    Real Malmquist--Takenaka (MT) system.
    
    The system is parametrized by a sequence of inverse poles with given
    multiplicities (mults). Every inverse pole is represented with its
    complex magnitude and argument. 
    '''
    def __init__(self, num_samples: int, mults: list[int]) -> None:
        '''
        Constructs a real MT function system. Complex system parameters
        (inverse poles) are represented with their complex magnitude and 
        argument. Number of coefficients and parameters depend on the
        multiplicities:
            num_params = 2 * len(mults)
            num_coeffs = 1 + 2 * sum(mults)
        
        Input:
            num_samples: int    Number of time sampling points of the input
            mults: list[int]    List of multiplicities of inverse poles.
        See also FunSystem.__init__
        '''
        assert num_samples > 0
        assert len(mults) > 0
        assert all([m > 0 for m in mults])
        super().__init__(num_samples, 1 + 2 * sum(mults), 2 * len(mults))
        self.mults = mults
        # index precomputation
        num_coeffs_c = self.num_coeffs // 2         # sum(mults)
        i = 0
        k0 = self.num_params * [None]
        k1 = self.num_params * [None]
        k2 = self.num_params * [None]
        for j, m in enumerate(self.mults):
            k0_j = num_coeffs_c * [0]
            k1_j = num_coeffs_c * [0]
            k2_j = num_coeffs_c * [0]
            for k in range(m):
                k0_j[i] = 1
                k1_j[i] = k
                k2_j[i] = k + 1
                i += 1
            for ii in range(i, num_coeffs_c):
                k1_j[ii] = m
                k2_j[ii] = m
            k0_j = torch.tensor(k0_j)
            k1_j = torch.tensor(k1_j)
            k2_j = torch.tensor(k2_j)
            k0[2 * j] = k0_j
            k0[2 * j + 1] = torch.zeros_like(k0_j)
            k1[2 * j] = -k1_j
            k1[2 * j + 1] = -k1_j
            k2[2 * j] = k2_j
            k2[2 * j + 1] = -k2_j
        self._k0 = torch.stack(k0).unsqueeze(-1)    # (num_params, num_coeffs, 1)
        self._k1 = torch.stack(k1).unsqueeze(-1)    # (num_params, num_coeffs, 1)
        self._k2 = torch.stack(k2).unsqueeze(-1)    # (num_params, num_coeffs, 1)

    def __call__(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes the real MT functions and derivatives. System parameters 
        (inverse poles) are represented with their complex magnitude and
        argument, in this order, i.e. params is expected to have the
        structure
            [magnitude0, argument0, magnitude1, argument1, ...]
        The functions are uniformly sampled between -pi and pi.
        
        See also FunSystem.__call__
        '''
        t = (2 * torch.arange(self.num_samples, dtype=params.dtype, device=params.device) / self.num_samples - 1) * math.pi
        z = torch.exp(1j * t)
        sqrt2 = torch.sqrt(torch.tensor(2, dtype=params.dtype, device=params.device))
        # complex intermediate computation
        mults = torch.tensor(self.mults, device=params.device)
        # Phi
        r = torch.repeat_interleave(params[0::2], mults).unsqueeze(-1)
        a = torch.repeat_interleave(params[1::2], mults).unsqueeze(-1)
        eia = torch.exp(1j * a)
        R1 = z - r * eia                # (num_coeffs // 2, num_samples)
        R2 = 1 - r * eia.conj() * z     # (num_coeffs // 2, num_samples)
        B0 = sqrt2 * torch.sqrt(1 - r ** 2) / R2
        B1 = R1 / R2
        B = torch.cumprod(B1, dim=0)
        B = torch.cat((torch.ones((1, self.num_samples), dtype=params.dtype, device=params.device), B))
        B = B[:-1, :]
        Phi = B0 * B * z                # (num_coeffs // 2, num_samples)
        # dPhi
        r0 = params[0::2]
        r = torch.stack((r0, r0)).T.reshape(self.num_params).unsqueeze(-1).unsqueeze(-1)
        a0 = params[1::2]
        a = torch.stack((a0, a0)).T.reshape(self.num_params).unsqueeze(-1).unsqueeze(-1)
        eia = torch.exp(1j * a)
        R1 = z - r * eia                # (num_params, 1, num_samples)
        R2 = 1 - r * eia.conj() * z     # (num_params, 1, num_samples)
        d0 = -r / (1 - r ** 2)          # (num_params, 1, 1)
        d1 = eia / R1                   # (num_params, 1, num_samples)
        d2 = eia.conj() * z / R2        # (num_params, 1, num_samples)
        dm = torch.stack((torch.ones_like(r0), 1j * r0)).T.reshape(self.num_params).unsqueeze(-1).unsqueeze(-1)
        k0 = self._k0.to(dtype=params.dtype, device=params.device)
        k1 = self._k1.to(dtype=params.dtype, device=params.device)
        k2 = self._k2.to(dtype=params.dtype, device=params.device)
        dPhi = (k0 * d0 + k1 * d1 + k2 * d2) * dm * Phi # (num_params, num_coeffs // 2, num_samples)
        # Phi_0, dPhi_0
        Phi = torch.cat((torch.ones((1, self.num_samples), dtype=params.dtype, device=params.device), 
                         Phi.real, Phi.imag))           # (num_coeffs, num_samples)
        dPhi = torch.cat((torch.zeros((self.num_params, 1, self.num_samples),
                                       dtype=params.dtype, device=params.device), 
                          dPhi.real, dPhi.imag), dim=1) # (num_params, num_coeffs, num_samples)
        return Phi, dPhi

    def __repr__(self):
        return f'RealMTSystem(num_samples={self.num_samples}, mults={self.mults})'

def bbmm(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    '''
    Batch-batch matrix multiplication.
    
    Input:
        t1: torch.Tensor    Input tensor of size (batch1,*,num_samples).
        t2: torch.Tensor    Input tensor of size (batch2,num_samples,**).
    Output:
        out: torch.Tensor   Output tensor of size (batch2,batch1,*,**) computed
                            as the product of t1 and t2 along the last 
                            dimension of t1 and the second (first non-batch) 
                            dimension of t2.
    '''
    t2 = t2.permute(*range(1, t2.ndim), 0)    # (num_samples,**,batch2)
    out = torch.tensordot(t1, t2, dims=1)     # (batch1,*,**,batch2)
    out = out.permute(-1, *range(out.ndim-1)) # (batch2,batch1,*,**)
    return out

class VPFun(torch.autograd.Function):
    '''Variable Projection operators with analytic derivatives.'''
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, params: torch.Tensor,
                fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) \
                -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Computes the orthogonal projection of the input.
        
        Input:
            x: torch.Tensor         Input tensor of size (batch,*,num_samples).
            params: torch.Tensor    Tensor of nonlinear system parameters.
                                    Size: (num_params)
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                    Function system and derivative builder.
                                    Expected to return Phi and dPhi as of
                                    FunSystem.__call__
        Output:
            coeffs: torch.Tensor    Coefficients of orthogonal projection.
                                    coeffs = Phi^+ x
                                    Size: (batch,*,num_coeffs)
            x_hat: torch.Tensor     Orthogonal projection, VP approximation.
                                    x_hat = Phi coeffs
                                    Size: (batch,*,num_samples)
            res: torch.Tensor       Residual vector.
                                    res = x - x_hat
                                    Size: (batch,*,num_samples)
            r2: torch.Tensor        L2 error of approximation.
                                    r2 = ||res||^2
                                    Size: (batch,*)
        '''
        phi, dphi = fun_system(params)  # phi: (num_coeffs,num_samples)
                                        # dphi: (num_params,num_coeffs,num_samples)
        phip = torch.linalg.pinv(phi)   # (num_samples,num_coeffs)
        coeffs = x @ phip               # (batch,*,num_coeffs), coefficients
        x_hat = coeffs @ phi            # (batch,*,num_samples), approximations
        res = x - x_hat                 # (batch,*,num_samples), residual vectors
        r2 = (res ** 2).sum(dim=-1)     # (batch,*), L2 errors
        ctx.save_for_backward(phi, phip, dphi, coeffs, res)
        return coeffs, x_hat, res, r2

    @staticmethod
    def backward(ctx: Any, d_coeff: torch.Tensor, d_x_hat: torch.Tensor,
                 d_res: torch.Tensor, d_r2: torch.Tensor) \
                 -> tuple[torch.Tensor, torch.Tensor, None]:
        '''
        Computes the backpropagation gradients.
        
        Input:
            d_coeffs: torch.Tensor  Backpropagated gradient of coeffs.
                                    Size: (batch,*,num_coeffs)
            d_x_hat: torch.Tensor   Backpropagated gradient of x_hat.
                                    Size: (batch,*,num_samples)
            d_res: torch.Tensor     Backpropagated gradient of res.
                                    Size: (batch,*,num_samples)
            d_r2: torch.Tensor      Backpropagated gradient of r2.
                                    Size: (batch,*)
        Output:
            dx: torch.Tensor        Gradient of input x.
                                    Size: (batch,*,num_samples)
            d_params: torch.Tensor  Gradient of params.
                                    Size: (num_params)
            None                    [Argument if not differentiable.]
        '''
        phi, phip, dphi, coeffs, res = ctx.saved_tensors
        # Intermediate Jacobians:
        #   Jac1 = dPhi coeff
        #   Jac2 = Phi^+^T dPhi^T res
        #   Jac3 = dPhi^T Phi^+^T c
        jac1 = bbmm(coeffs, dphi)               # (num_params,batch,*,num_samples)
        jac2 = bbmm(res, dphi.mT) @ phip.T      # (num_params,batch,*,num_samples)
        jac3 = bbmm(coeffs @ phip.T, dphi.mT)   # (num_params,batch,*,num_coeffs)
        # Jacobians
        jac_coeff = jac3 + (-jac1 + jac2 - jac3 @ phi) @ phip   # (num_params,batch,*,num_coeffs)
        jac_x_hat = jac1 - jac1 @ phip @ phi + jac2             # (num_params,batch,*,num_samples)
        jac_res = -jac_x_hat                                    # (num_params,batch,*,num_samples)
        jac_r2 = -2 * (jac1 * res).sum(dim=-1)                  # (num_params,batch,*)
        # gradients
        dx = d_coeff @ phip.T + \
             d_x_hat @ phip @ phi + \
             d_res - d_res @ phip @ phi + \
             2 * d_r2.unsqueeze(-1) * res
        d_params = (jac_coeff * d_coeff).flatten(1).sum(dim=1) + \
                   (jac_x_hat * d_x_hat).flatten(1).sum(dim=1) + \
                   (jac_res * d_res).flatten(1).sum(dim=1) + \
                   (jac_r2 * d_r2).flatten(1).sum(dim=1)
        return dx, d_params, None

class VPIteration:
    '''
    Variable Projection iteration step for higher level computation.
    Backpropagation gradients are automatically computed based on VPFun.
    Use similar to torch.autograd.Function.
    '''
    @staticmethod
    def apply(x: torch.Tensor, params: torch.Tensor,
              fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) \
              -> torch.Tensor:
        '''
        Computes one step of the Variable Projection iteration.
        
        Input:
            x: torch.Tensor         Input tensor of size (batch,*,num_samples).
            params: torch.Tensor    Tensor of nonlinear system parameters.
                                    Size: (num_params)
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                    Function system and derivative builder.
                                    Expected to return Phi and dPhi as of
                                    FunSystem.__call__
        Output:
            iter: torch.Tensor      Iteration step.
                                    iter = -2 * res^T @ dPhi @ coeffs
                                    Size: (num_params)
        '''
        phi, dphi = fun_system(params)
        fun_fixed = lambda params: (phi, dphi)
        coeffs, _, res, _ = VPFun.apply(x, params, fun_fixed)
        return -2 * (bbmm(coeffs, dphi) * res).sum(dim=-1).flatten(1).mean(dim=1)

class VPLayer(torch.nn.Module):
    '''Variable Projection layer for neural networks.'''
    def __init__(self, params_init: torch.Tensor,
                 fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) -> None:
        '''
        Constructs a VP layer using the given function system, performing a
        single VP orthogonal projection.
        
        Input:
            params_init: torch.Tensor   Initial values of nonlinear system parameters.
                                        Size: (num_params)
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                        Function system and derivative builder.
                                        Expected to return Phi and dPhi as of
                                        FunSystem.__call__
        See also VPFun
        '''
        super().__init__()
        self.params_init = params_init
        self.params = torch.nn.Parameter(params_init.clone().detach())
        self.fun_system = fun_system

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Forward operator, see VPFun'''
        return VPFun.apply(x, self.params, self.fun_system)

    def extra_repr(self) -> str:
        return f'params_init={self.params_init}, fun_system={self.fun_system}'

class VPIterationLayer(torch.nn.Module):
    '''VP iteration layer for deep unfolding.'''
    def __init__(self, weight_init: torch.Tensor, weight_learn: bool,
                 fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) -> None:
        '''
        Constructs a VP iteration layer using the given function system,
        performing a single VP iteration step with given step size (weight).
        
        Input:
            weight_init: torch.Tensor   Initial weight value. Size: ()
            weight_learn: bool          If True, then weight parameter is learnable,
                                        otherwise it is fixed to weight_init.
            fun_system: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                                        Function system and derivative builder.
                                        Expected to return Phi and dPhi as of
                                        FunSystem.__call__
        See also VPIteration
        '''
        super().__init__()
        self.weight_init = weight_init
        self.weight_learn = weight_learn
        self.weight = weight_init.clone().detach()
        if weight_learn:
            self.weight = torch.nn.Parameter(self.weight)
        self.fun_system = fun_system

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        '''Forward operator, see VPIteration'''
        return params - self.weight * VPIteration.apply(x, params, self.fun_system)

    def extra_repr(self) -> str:
        return f'weight_init={self.weight_init}, weight_learn={self.weight_learn}, fun_system={self.fun_system}'


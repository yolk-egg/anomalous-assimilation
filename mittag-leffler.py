'''
BSD 3-Clause License

Copyright (c) 2017, Konrad Hinsen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import numpy as np
from scipy.special import gamma
from ml_internal import LTInversion

def ml(z, alpha, beta=1., gama=1.):
    eps = np.finfo(np.float64).eps
    if np.real(alpha) <= 0 or np.real(gama) <= 0 or np.imag(alpha) != 0. \
       or np.imag(beta) != 0. or np.imag(gama) != 0.:
        raise ValueError('ALPHA and GAMA must be real and positive. BETA must be real.')
    if np.abs(gama-1.) > eps:
        if alpha > 1.:
            raise ValueError('GAMMA != 1 requires 0 < ALPHA < 1')
        if (np.abs(np.angle(np.repeat(z, np.abs(z) > eps))) <= alpha*np.pi).any():
            raise ValueError('|Arg(z)| <= alpha*pi')

    return np.vectorize(ml_, [np.float64])(z, alpha, beta, gama)

def ml_(z, alpha, beta, gama):
    # Target precision 
    log_epsilon = np.log(1.e-15)
    # Inversion of the LT
    if np.abs(z) < 1.e-15:
        return 1/gamma(beta)
    else:
        return LTInversion(1, z, alpha, beta, gama, log_epsilon)


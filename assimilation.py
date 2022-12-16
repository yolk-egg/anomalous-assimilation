import numpy as np
import matplotlib.pyplot as plt
import scipy
import shutil

import os

from scipy.special import gamma
from scipy.integrate import simpson

def step_int_RK(u, L, ts, dt, order=3/2):
  """
  Computes the step du associated with a timestep dt (Runge-Kutta).
  float array u: (N, X, T) = (ensemble members, spatial coordinate, time coordinate)
  function L: float array -> float array, the function that acts on the spatial coordinate
  float array array ts: (T) the times integrated so far
  float dt: the step size
  float order: the order of the time derivative

  returns 
  float array du: (N, X), the step
  """

  p = order-1
  s1 = np.power(ts, p)
  s2 = np.power(np.append(ts, ts[-1] + dt/2), p)
  s4 = np.power(np.append(ts, ts[-1] + dt), p)
  k1 = simpson(np.flip(L(u), axis=-1), s1, axis=-1) / (gamma(p+1))
  u2 = np.append(u, k1[..., np.newaxis] * dt / 2, axis=-1)
  k2 = simpson(np.flip(L(u2), axis=-1), s2, axis=-1) / (gamma(p+1))
  u3 = np.append(u, k2[..., np.newaxis] * dt / 2, axis=-1)
  k3 = simpson(np.flip(L(u3), axis=-1), s2, axis=-1) / (gamma(p+1))
  u4 = np.append(u, k3[..., np.newaxis] * dt, axis=-1)
  k4 = simpson(np.flip(L(u4), axis=-1), s4, axis=-1) / (gamma(p+1))

  output = (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
  return output

def step_int(u, L, ts, dt, order=3/2):
  """
  Computes the step du associated with a timestep dt (Euler).
  float array u: (N, X, T) = (ensemble members, spatial coordinate, time coordinate)
  function L: float array -> float array, the function that acts on the spatial coordinate
  float array array ts: (T) the times integrated so far
  float dt: the step size
  float order: the order of the time derivative

  returns 
  float array du: (N, X), the step
  """

  p = order-1
  s = np.power(ts, p)
  output = dt * simpson(np.flip(L(u), axis=-1), s, axis=-1) / (gamma(p+1))
  return output

def prop_state(ts, z, alpha, L_k):
  """
  Propagates the state z over timesteps ts assuming the Fourier transform of z 
  obeys the equation d^alpha z(k, t)/ dt^alpha = L_k(z)
  float array ts (t): all the times to evaluate the state at
  float array z(n, x, t1 < t-2): the state at some already calculated times (in
    real space)
  float alpha: the time derivative order
  float array (n, x, t) -> float array (n, x, t) L_k: the operator
    L_x in Fourier space
  
  returns
  float array z (n, x, t): the state evaluated at all the times in ts
  """

  # ts must be evenly spaced
  dt = ts[1]-ts[0]

  # We assume z has been integrated over some of the times already
  n_prop = ts.shape[0] - z.shape[2]
  assert n_prop >= 0

  # It's more stable in Fourier space
  out_size = np.array(z.shape)
  out_size[-1] = ts.shape[0]
  z_fft = np.zeros(out_size, dtype='complex128')
  z_fft[..., 0:z.shape[2]] = np.fft.fft(z, axis=1)

  for i in range(n_prop):
    z_step = z_fft[:, :, z.shape[2]+i-1] + step_int(z_fft[:, :, 0:z.shape[2]+i], L_k, ts[0:z.shape[2] + i], dt, alpha)
    z_fft[..., z.shape[2]+i] = z_step

  return np.real(np.fft.ifft(z_fft, axis=1))

def assimilation(z, x, ts, obs_steps, alpha, L_k, n_members, H, P, B, mode='boolean'):
  """
  Perform a data assimilation of the system d^alpha z(x, t)/ dt^alpha = L_x(z)
  float array z: initial truth
  float array x: the spatial coordinates
  float array ts: the times over which to assimilate
  int array obs_steps: the timesteps at which observations are made
  float alpha: time derivative order
  float array (n, x, t) -> float array (n, x, t) L_k: the operator
    L_x in Fourier space
  int n_members: the number of ensemble members
  float array (d, x) H: observation matrix, truth table if mode=boolean
  float array (d, d) P: observation covariance, 1d array if mode=boolean
  float array (x, x) B: initial state covariance, 1d array if mode=boolean
  string mode: the assimilation mode, only 'boolean' is implemented ('full' or 
    something should be when the observations are not just a subset of the space
    and the covariances are not diagonal, in which case H, P, B will be matrices)

  Outputs:
  float array z_true (x, t): the truth
  float array z_mean (x, t): the mean of the ensemble
  float array z_cov (x, x, t): the covariance matrix of the ensemble
  """
  rng = np.random.default_rng()
  # prepare initial state
  N_t = np.size(ts)
  N_x = np.size(x)

  # Calculate wavevectors
  ks = np.fft.fftfreq(N_x, d=x[1]-x[0])

  z_true = z[np.newaxis, :, np.newaxis]

  if mode=='boolean':
    z_est = z[np.newaxis, :, np.newaxis] + rng.normal(0, 1, (n_members, N_x, 1)) * B[np.newaxis, : , np.newaxis]


  # Do the observation stuff
  for obs_step in obs_steps:
    # Propagate to an observation time
    z_est = prop_state(ts[0:obs_step], z_est, alpha, lambda u: L_k(u, ks))
    z_true = prop_state(ts[0:obs_step], z_true, alpha, lambda u: L_k(u, ks))
    # Perform the observation
    # we first need the covariance matrix
    Q = np.cov(z_est[:, :, -1].transpose())
    # Compute the Kalman Gain
    if mode == 'boolean':
      QHt = np.copy(Q[:, H])
      HQHt = np.copy(QHt[H, :])
      print(obs_step)
      HQHt[np.diag_indices_from(HQHt)] += P
      K = QHt @ np.linalg.inv(HQHt)
      # Make an observation
      d = z_true[0, H, -1]
      d += rng.normal(0, 1, d.shape) * P

      # Compute the increments
      y = d[np.newaxis, :] - z_est[:, H, -1]
      # Compute the new estimate of the state
      z_est[:, :, -1] += np.matmul(y, K.transpose())


  # Propagate to the final time
  z_est = prop_state(ts, z_est, alpha, lambda u: L_k(u, ks))
  z_true = prop_state(ts, z_true, alpha, lambda u: L_k(u, ks))

  # Calculate statistics
  z_mean = np.mean(z_est, axis=0)
  z_cov = np.array([np.cov(z_est[:, :, i].transpose()) for i in range(N_t)]).transpose(1, 2, 0)
  return (z_true[0, :, :], z_mean, z_cov)

def animation(path, x, z_mean, skip=5, z_cov=None, z_true=None, dpi=200):
  '''
  Make a zip file containing frames of an animation (this is very inefficient, 
  ssshhh)
  Animate with ffmpeg or however you please:
  $ ffmpeg -i %05.png [name].webm
  string path: name of the folder that will store the files. This folder should
    not already exist, it will be deleted after the zip is created
  float array x: the x coordinates
  float array z_mean (x, t): the function to plot at each time
  int skip: the number of timesteps to skip between each frame
  float array z_cov (x, x, t): the covariance matrix of the function values
  float array z_true (x, t): the truth of the function (will be dashed, black)
  int dpi: the dpi to save as
  '''
  assert not os.path.isdir(path)
  os.makedirs(path)
  z_mean = z_mean[:, ::skip]
  if z_cov is not None:
    z_cov = z_cov[:, :, ::skip]
  if z_true is not None:
    z_true = z_true[:, ::skip]
  N_t = z_mean.shape[1]
  a, b = (np.min(z_mean), np.max(z_mean))
  for i in range(N_t):
    fig, ax = plt.subplots()
    ax.plot(x, z_mean[:, i])
    if z_cov is not None:
      ax.fill_between(x, z_mean[:, i] + np.sqrt(np.diagonal(z_cov[:, :, i])), z_mean[:, i] - np.sqrt(np.diagonal(z_cov[:, :, i])), alpha=0.3)
    if z_true is not None:
      ax.plot(x, z_true[:, i], c='k', linestyle='--')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u$')
    ax.set_ylim(a, b)
    plt.savefig('{}/{:05d}'.format(path, i), dpi=dpi)
    plt.show()
  shutil.make_archive("anim", "zip", path)
  shutil.rmtree(path)
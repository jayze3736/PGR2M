import numpy as np


def sine_schedule(t, T, tau_plus, tau_minus):
  sin_term = np.sin(np.pi / 2 * t / T)
  result = tau_minus + (tau_plus - tau_minus) * sin_term
  return result

def cosine_schedule(t, T, tau_plus, tau_minus):
  cos_term = np.cos(np.pi * t / T) 
  result = (tau_plus - tau_minus) * (1 + cos_term) / 2 + tau_minus
  return result

def linear_schedule_with_min_threshold(t, T, tau_plus, tau_minus):
  T_safe = max(T, 1)
  linear_part = tau_plus - (tau_plus - tau_minus) * (t / T_safe)
  result = np.where(t < T, linear_part, tau_minus)
  
  return result
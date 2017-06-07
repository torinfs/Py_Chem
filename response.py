# Function to build the A and B response matrices
# Torin Stetina
# June 1st, 2017

import numpy as np


def spin_eri(eriMO, sdim):

  # ** Original algorithm based on function
  # ** from joshuagoings.com/2013/05/27/tdhf-cis-in-python/
  #
  # Makes spin adapted 2 electron integrals from
  # eriMO that also can be represented as the
  # double bar integral < pq || rs > 

  seri = np.zeros((sdim,sdim,sdim,sdim))
  for p in range(0,sdim):  
    for q in range(0,sdim):  
      for r in range(0,sdim):  
        for s in range(0,sdim):
          v1 = eriMO[p//2,r//2,q//2,s//2] * (p%2 == r%2) * (q%2 == s%2)  
          v2 = eriMO[p//2,s//2,q//2,r//2] * (p%2 == s%2) * (q%2 == r%2)  
          seri[p,q,r,s] = v1 - v2 

  return seri
    

def responseAB(eriMO, eps, Nelec):
  
  # Get spin adapted ERIs
  dim = len(eriMO)
  #seri = spin_eri(eriMO, sdim)
  
  # Extend epsilon array for spin
  #spin_eps = np.zeros((sdim))
  #for i in range(0,sdim):
  #  spin_eps[i] = eps[i//2]
  #spin_eps = np.diag(spin_eps)
  
  # Compute A and B matrix elements 
  A = np.zeros((Nelec/2*(dim-Nelec/2),Nelec/2*(dim-Nelec/2)))
  B = np.zeros((Nelec/2*(dim-Nelec/2),Nelec/2*(dim-Nelec/2)))
  ia = -1
  for i in range(0,Nelec/2):
    for a in range(Nelec/2,dim):
      ia += 1
      jb  = -1
      for j in range(0,Nelec/2):
        for b in range(Nelec/2,dim):
          jb += 1
          
          # A = (e_a - e_i) d_{ij} d{ab} * < aj || ib >
          A[ia,jb] = (eps[a] - eps[i]) \
                    * (i == j) * (a == b) + eriMO[a,j,i,b]
          
          # B = < ab || ij >
          B[ia,jb] = eriMO[a,b,i,j]
  
  return A, B


def TDHF(eriMO, eps, Nelec):
  
  # Get A and B matrices
  A, B = responseAB(eriMO, eps, Nelec)
  
  # Solve non-Hermetian eigenvalue problem
  M = np.bmat([[A, B],[-B, -A]])
  E_td, C_td = np.linalg.eig(M)
  print 'Excitation Energies (TDHF) = ', E_td       








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
    

def responseAB(eriMO, eps, Nelec, R):
  # R = Restriced (is a boolean)  
  # eriMO = MO transformed ERIs
  # eps = orbital energies
  # Nelec = # of electrons


  # Get spin adapted ERIs if Unrestricted
  dim = len(eriMO)
  if not R:
    sdim = 2*dim
    seri = spin_eri(eriMO, sdim)
    
    # Extend epsilon array for spin
    spin_eps = np.zeros((sdim))
    for i in range(0,sdim):
      spin_eps[i] = eps[i//2]
    spin_eps = np.diag(spin_eps)
    
    A = np.zeros((Nelec*(sdim-Nelec),Nelec*(sdim-Nelec)))
    B = np.zeros((Nelec*(sdim-Nelec),Nelec*(sdim-Nelec)))
  
  elif R:
    A = np.zeros((Nelec/2*(dim-Nelec/2),Nelec/2*(dim-Nelec/2)))
    B = np.zeros((Nelec/2*(dim-Nelec/2),Nelec/2*(dim-Nelec/2)))
  
  # Compute A and B matrix elements 
  ia = -1
  for i in range(0,Nelec/2):
    for a in range(Nelec/2,dim):
      ia += 1
      jb  = -1
      for j in range(0,Nelec/2):
        for b in range(Nelec/2,dim):
          jb += 1
          
          if R:
            # A = (e_a - e_i) d_{ij} d{ab} * < aj || ib >
            # < aj || ib > = < aj | ib > - < aj | bi >
            #              = ( ai | jb ) - ( ab | ji ) 
            A[ia,jb] = (eps[a] - eps[i]) \
                      * (i == j) * (a == b) + 2*eriMO[a,i,j,b] - eriMO[a,b,j,i]
          
            # B = < ab || ij >
            # < ab || ij > = < ab | ij > - < ab | ji >
            #              = ( ai | bj ) - ( aj | bi )
            B[ia,jb] = 2*eriMO[a,i,b,j] - eriMO[a,j,b,i]
          
          elif not R:
            # A = (e_a - e_i) d_{ij} d{ab} * < aj || ib >
            A[ia,jb] = (spin_eps[a,a] - spin_eps[i,i]) \
                      * (i == j) * (a == b) + seri[a,j,i,b]
          
            # B = < ab || ij >
            B[ia,jb] = seri[a,b,i,j]

  return A, B


def TDHF(eriMO, eps, Nelec, R):
  
  # Get A and B matrices
  A, B = responseAB(eriMO, eps, Nelec, R)

  # Solve non-Hermetian eigenvalue problem
  M = np.bmat([[A, B],[-B, -A]])
  E_td, C_td = np.linalg.eig(M)
  print 'Excitation Energies (TDHF) = \n'
  for i in range(len(E_td)):
    if E_td[i] > 0.00:
      #print E_td[i], 'a.u.' 
      print 27.211396132*E_td[i], 'eV' 








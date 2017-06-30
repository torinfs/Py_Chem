#!/usr/bin/env python
# -*- coding: utf-8 -*- 
#
# Function to build the A and B response matrices
# Torin Stetina
# June 1st, 2017

import numpy as np


def spin_eri(eriMO, sdim):

  # ** Original algorithm based on function
  # ** from joshuagoings.com/2013/05/27/tdhf-cis-in-python/
  #
  # Makes spin adapted 2 electron integrals from
  # eriMO in RHF that also can be represented as the
  # double bar integral < pq || rs > 
  # WARNING: Converts to dirac notation

  seri = np.zeros((sdim,sdim,sdim,sdim))
  for p in range(0,sdim):  
    for q in range(0,sdim):  
      for r in range(0,sdim):  
        for s in range(0,sdim):
          v1 = eriMO[p//2,r//2,q//2,s//2] * (p%2 == r%2) * (q%2 == s%2)  
          v2 = eriMO[p//2,s//2,q//2,r//2] * (p%2 == s%2) * (q%2 == r%2)  
          seri[p,q,r,s] = v1 - v2 
  
  return seri
    

def responseAB_RHF(eriMO, eps, Nelec, S):
  # S = singlet (is a boolean)  
  # eriMO = MO transformed ERIs
  # eps = orbital energies
  # Nelec = # of electrons


  dim = len(eriMO)

  if not S: # Spin-Adapted (triplets)
    sdim = 2*dim
    seri = spin_eri(eriMO, sdim)
    
    # Extend epsilon array for spin
    spin_eps = np.zeros((sdim))
    for i in range(0,sdim):
      spin_eps[i] = eps[i//2]
    spin_eps = np.diag(spin_eps)
    
    A = np.zeros((Nelec*(sdim-Nelec),Nelec*(sdim-Nelec)))
    B = np.zeros((Nelec*(sdim-Nelec),Nelec*(sdim-Nelec)))
  
    # Compute A and B matrix elements 
    ia = -1
    for i in range(0,Nelec):
      for a in range(Nelec,sdim):
        ia += 1
        jb  = -1
        for j in range(0,Nelec):
          for b in range(Nelec,sdim):
            jb += 1
            
            # A = (e_a - e_i) d_{ij} d{ab} * < aj || ib >
            A[ia,jb] = (spin_eps[a,a] - spin_eps[i,i]) \
                      * (i == j) * (a == b) + seri[a,j,i,b]
            
            # B = < ab || ij >
            B[ia,jb] = seri[a,b,i,j]


  elif S: # Singlets only
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
            
            # A = (e_a - e_i) d_{ij} d{ab} * < aj || ib >
            # < aj || ib > = < aj | ib > - < aj | bi >
            #              = ( ai | jb ) - ( ab | ji ) 
            A[ia,jb] = (eps[a] - eps[i]) \
                      * (i == j) * (a == b) + 2*eriMO[a,i,j,b] - eriMO[a,b,j,i]
           
            # B = < ab || ij >
            # < ab || ij > = < ab | ij > - < ab | ji >
            #              = ( ai | bj ) - ( aj | bi )
            B[ia,jb] = 2*eriMO[a,i,b,j] - eriMO[a,j,b,i]

  return A, B


def responseAB_UHF(eriMO, eps, Nelec):
  # eriMO = MO transformed ERIs in Block form
  # eps = [eps_a, eps_b]
  # Nelec = [Na, Nb]
  dim = len(eriMO[0])
  
  # Compute A and B matrix elements 
  wx = -1
  block_A = []
  block_B = []

  for w in range(2):
    for x in range(2):
      wx += 1
      ia = -1
      A  = np.zeros((Nelec[w]*(dim-Nelec[w]),Nelec[x]*(dim-Nelec[x])))
      B  = np.zeros((Nelec[w]*(dim-Nelec[w]),Nelec[x]*(dim-Nelec[x])))
      rg = [w,x]
      for i in range(0,Nelec[rg[0]]):
        for a in range(Nelec[rg[0]],dim):
          ia += 1
          jb  = -1
          for j in range(0,Nelec[rg[1]]):
            for b in range(Nelec[rg[1]],dim):
              jb += 1

              # A = (e_a - e_i) d_{ij} d{ab} d{σσ'} + (aiσ|jbσ') - d{σσ'}(abσ|jiσ)
              A[ia,jb] = (eps[w][a] - eps[w][i]) \
                   * (i == j) * (a == b) * (w == x) \
                    + eriMO[wx][a,i,j,b] - (w == x) * eriMO[wx][a,b,j,i]

              # B = (aiσ|bjσ') - d{σσ'}(ajσ|biσ)
              B[ia,jb] = eriMO[wx][a,i,b,j] - (w == x)*eriMO[wx][a,j,b,i]

      block_A.append(A)   
      block_B.append(B)
  
  A = np.bmat([[block_A[0], block_A[1]],[block_A[2], block_A[3]]])
  B = np.bmat([[block_B[0], block_B[1]],[block_B[2], block_B[3]]])
  return A, B

def TDHF(eriMO, eps, Nelec, R):
  
  # Get A and B matrices
  A, B = responseAB_UHF(eriMO, eps, Nelec)

  # Solve non-Hermetian eigenvalue problem
  M = np.bmat([[A, B],[-B, -A]])
  E_td, C_td = np.linalg.eig(M)

  Energies = []  
  print 'Excitation Energies (TDHF) = '
  for i in range(len(E_td)):
    if E_td[i] > 0.00:
      Energies.append(E_td[i]) 

  Energies = sorted(Energies)
  for i in range(len(Energies)):
    print 27.211396132*Energies[i], 'eV' 






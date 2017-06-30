# Transforms integrals in the AO basis to the MO basis
# Torin Stetina
# May 31st, 2017

import numpy as np

def ao2mo(eriAO, cAO, R): 
  # eriAO = AO electron repulsion matrix
  # cAO   = AO Coefficient matrix from SCF, [C_a, C_b] if UHF
  # R     = Restriced (is a boolean)  
  dim = len(eriAO)
  
  if R: 

    # N^5 smart AO to MO transformation
    temp  = np.zeros((dim,dim,dim,dim))
    temp2 = np.zeros((dim,dim,dim,dim))
    temp3 = np.zeros((dim,dim,dim,dim))
    eriMO = np.zeros((dim,dim,dim,dim))
    for p in range(0,dim):
      for mu in range(0,dim):
        temp[p,:,:,:] += cAO[mu,p] * eriAO[mu,:,:,:]
      for q in range(0,dim):
        for nu in range(0,dim):
          temp2[p,q,:,:] += cAO[nu,q] * temp[p,nu,:,:]
        for r in range(0,dim):
          for lam in range(0,dim):
            temp3[p,q,r,:] += cAO[lam,r] * temp2[p,q,lam,:]
          for s in range(0,dim):
            for kap in range(0,dim):
              eriMO[p,q,r,s] += cAO[kap,s] * temp3[p,q,r,kap]

    return eriMO

  elif not R:

    # N^5 smart AO to MO transformation
    # make block MOs explicitly:
    # | M_aa,aa M_aa,bb |
    # | M_bb,aa M_bb,bb |
    block_MOs = [] # FIX below, only need 2 for loops
    for a in range(2):
      for b in range(2):
        temp  = np.zeros((dim,dim,dim,dim))
        temp2 = np.zeros((dim,dim,dim,dim))
        temp3 = np.zeros((dim,dim,dim,dim))
        eriMO = np.zeros((dim,dim,dim,dim))
        for p in range(0,dim):
          for mu in range(0,dim):
            temp[p,:,:,:] += cAO[a][mu,p] * eriAO[mu,:,:,:]
          for q in range(0,dim):
            for nu in range(0,dim):
              temp2[p,q,:,:] += cAO[a][nu,q] * temp[p,nu,:,:]
            for r in range(0,dim):
              for lam in range(0,dim):
                temp3[p,q,r,:] += cAO[b][lam,r] * temp2[p,q,lam,:]
              for s in range(0,dim):
                for kap in range(0,dim):
                  eriMO[p,q,r,s] += cAO[b][kap,s] * temp3[p,q,r,kap]

        block_MOs.append(eriMO)

    return block_MOs 



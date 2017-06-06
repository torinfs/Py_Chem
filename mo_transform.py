# Transforms integrals in the AO basis to the MO basis
# Torin Stetina
# May 31st, 2017

import numpy as np

def ao2mo(eriAO, cAO): 
  dim = len(eriAO)
  temp = np.zeros((dim,dim,dim,dim))
  temp2 = np.zeros((dim,dim,dim,dim))
  temp3 = np.zeros((dim,dim,dim,dim))
  eriMO = np.zeros((dim,dim,dim,dim))
  
  # N^5 smart AO to MO transformation
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

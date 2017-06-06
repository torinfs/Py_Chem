# MP2 correlation energy post scf and AO to MO transformation
# Torin Stetina
# June 1st, 2017

# ONLY RHF CURRENTLY
def mp2(eriMO, eps, Nelec):
  mp2E = 0
  dim = len(eriMO)
  for i in range(0, Nelec/2): 
    for a in range(Nelec/2, dim):
      for j in range(0, Nelec/2):
        for b in range(Nelec/2, dim):
          mp2E += eriMO[i,a,j,b]*(2*eriMO[i,a,j,b] \
                  - eriMO[i,b,j,a])/ (eps[i] + eps[j] \
                                   - eps[a] - eps[b])

  return mp2E

# Self-Consistent Field Main Script - Unrestricted
# Torin Stetina
# June 5th 2017

import time
import numpy as np
from numpy import genfromtxt
from scipy.linalg import eig, eigh
import scipy.io

# Internal imports
from mo_transform import ao2mo
from mp2 import mp2
from response import*
#import sys
##sys.path.insert(0, '/path/to/application/app/folder')


###############################
##     F U N C T I O N S     ##
###############################

def getIntegrals(molecule):
  # Vnn = Nuclear repulsion energy value
  # Vne = Electron-Nuclear attraction matrix
  # T   = Kinetic energy matrix
  # S   = Overlap matrix
  # ERI = Electron-Electron repulsion tensor
  direc = 'test_systems/' + molecule

  Vnn = genfromtxt('./' + direc + '/Vnn.dat',dtype=None)
  Vne = genfromtxt('./' + direc + '/Vne.dat',dtype=None)
  T   = genfromtxt('./' + direc + '/T.dat',dtype=None)
  S   = genfromtxt('./' + direc + '/S.dat',dtype=None)
  ERI = scipy.io.loadmat( direc + '/ERI.mat', squeeze_me=False) 
  return Vnn, Vne, T, S, ERI['ERI']


def deltaP(P,P_old):  
  # Calculate change in density matrix  
  return max(abs(P.flatten()-P_old.flatten()))


###############################
##       M   A   I   N       ##
###############################

start_time = time.time()

#### TEST SYSTEMS #### 

## Test Molecules
#mol, Nelec, name, basis = 'H2_STO3G', 2, 'H2', 'STO-3G'
#mol, Nelec, name, basis, mult = 'HeHplus_STO3G', 2, 'HeH+', 'STO-3G', 0
#mol, Nelec, name, basis, mult = 'H2O_STO3G', 10, 'Water', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'Li_STO3G', 3, 'Lithium', 'STO-3G', 2
mol, Nelec, name, basis, mult = 'O2_STO3G', 16, 'Oxygen', 'STO-3G', 3

######################



# Define number of MOs
#nMOs = Nelec //2  # RHF

# Get integrals from files
Vnn, Vne, T, S, ERI = getIntegrals(mol)

# Number of basis functions dim
dim = len(S)

# Build Core Hamiltonian
h = T + Vne

# Set up initial Fock with core guess, and density at 0
F_a, F_b = h, h
P_a, P_b = np.zeros((dim,dim)), np.zeros((dim,dim))
C_a, C_b = np.zeros((dim,dim)), np.zeros((dim,dim))

# Form transformation matrix
s, Y = eigh(S)
s = np.diag(s**(-0.5))
X = np.dot(Y, np.dot(s, Y.T))

# Initialize variables
delta  = 1.0
conver = 1.0e-10
count  = 0

Na = Nelec//2 + mult-1 
Nb = Nelec//2

# Start main SCF loop
while delta > conver and count < 300:
  count += 1
  E0 = 0
  Vee_a = np.zeros((dim,dim))
  Vee_b = np.zeros((dim,dim))
  for m in range(0,dim):
    for n in range(0,dim):
      for k in range(0,dim):
        for l in range(0,dim):
          Vee_a[m,n] += (P_a[k,l] + P_b[k,l]) * ERI[m,n,l,k] \
                                  - P_a[k,l]  * ERI[m,k,l,n]
          Vee_b[m,n] += (P_a[k,l] + P_b[k,l]) * ERI[m,n,l,k] \
                                  - P_b[k,l]  * ERI[m,k,l,n]
      
      E0 += 0.5 *( (P_a[m,n] + P_b[m,n])*h[m,n] + P_a[m,n]*(h[m,n] + Vee_a[m,n])\
                                                + P_b[m,n]*(h[m,n] + Vee_b[m,n]))
  
  E0 += Vnn

  # Update Fock alpha and beta
  F_a = h + Vee_a
  F_b = h + Vee_b
 
  # Orthonormalize Fock alpha and beta
  F_a_oao = np.dot(X.T,np.dot(F_a, X))
  F_b_oao = np.dot(X.T,np.dot(F_b, X))
  eps_a, C_a_oao = eigh(F_a_oao)
  eps_b, C_b_oao = eigh(F_b_oao)
  C_a = np.dot(X, C_a_oao)
  C_b = np.dot(X, C_b_oao)
  
  # Normalize C
  norm_a = np.sqrt(np.diag(np.dot(np.dot(np.transpose(C_a),S),C_a) ))
  norm_b = np.sqrt(np.diag(np.dot(np.dot(np.transpose(C_b),S),C_b) ))
  for i in range(0,dim):
    C_a[:,i] = C_a[:,i]/norm_a[i]
    C_b[:,i] = C_b[:,i]/norm_b[i]
  
  # Update Density Matrix
  P_old = P_a + P_b   
  P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
  P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
  
  # Compute change in density matrix
  delta = deltaP((P_a + P_b), P_old)



# Get Spin Contamination
spin2_exact = ((mult-1)/2.0) * ((mult-1)/2.0 + 1)
spin_overlap = 0.0
print C_a, '\n'
print C_b, '\n'
for i in range(0,Na):
  for j in range(Na+1,Nelec):
    spin_overlap += abs(np.inner(C_a[:,i], C_b[:,j-Na+1]))**2

spin2_tot = spin2_exact + Nb - spin_overlap


elapsed_time = time.time() - start_time

### Print results ###
print ''
print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
print '                   R e s u l t s            \n'
print 'Molecule: ' + name 
print 'Basis: ' + basis
print 'E(SCF) = ' + str(E0) + ' a.u.'
print 'SCF iterations: ' + str(count)
print 'Elapsed time: ' + str(elapsed_time) + ' sec'
print ''
print 'Fock Matrix (alpha) = \n' + np.array_str(F_a)
print 'Fock Matrix (beta) = \n' + np.array_str(F_b)
print 'Density Matrix (alpha) = \n' + np.array_str(P_a)
print 'Density Matrix (beta) = \n' + np.array_str(P_b)
print 'Orbital Energies (alpha) = \n' + str(eps_a) + '\n'
print 'Orbital Energies (beta) = \n' + str(eps_b) + '\n'
print spin2_exact, Nb, spin_overlap, '\n'
print '<S^2>_UHF =', spin2_tot, '\n'
print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
print ''

# Convert AO to MO orbital basis
#print '-------------------------'
#eriMO = ao2mo(ERI, C)
#mp2(eriMO, eps, Nelec)
#print responseAB(eriMO, eps, Nelec)
#TDHF(eriMO, eps, Nelec, False)
#print '-------------------------'











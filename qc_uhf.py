# Self-Consistent Field Main Script - Unrestricted
# Torin Stetina
# June 27th 2017

import time
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
from scipy.linalg import eig, eigh, inv, expm
import scipy.io

# Internal imports
from mo_transform import ao2mo
from mp2 import*
from response import*
import sys
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


def calcAlphaBeta(Nelec, mult):
  # Nelec = # of electrons in system
  # mult  = multiplicity of system

  if Nelec % 2 == 0:
    Na, Nb = Nelec//2, Nelec//2
    Na += (mult-1)/2
    Nb -= (mult-1)/2
  elif Nelec % 2 != 0:
    Na, Nb = Nelec//2, Nelec//2
    Na += (mult-1)/2 + 1
    Nb -= (mult-1)/2
  return Na, Nb


def buildFock(dim,P_a,P_b,h,ERI):
  # N^4 loop through basis fuctions to build the Fock matrix
  # and calculate the Hartree energy
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

      E0 += 0.5 *((P_a[m,n] + P_b[m,n])*h[m,n] + P_a[m,n]*(h[m,n] + Vee_a[m,n])\
                                               + P_b[m,n]*(h[m,n] + Vee_b[m,n]))
  E0 += Vnn
  F_a = h + Vee_a
  F_b = h + Vee_b

  return E0, F_a, F_b


def deltaP(P,P_old):  
  # Calculate change in density matrix  
  return max(abs(P.flatten()-P_old.flatten()))


def is_pos_def(x):
  return np.all(np.linalg.eigvals(x) > 0)


def getOVfvector(F, Nelec, dim):
  # Get occupied-virtual block of Fock matrix,
  # then convert into vector for QC eigenvalue problem
  F_ov = np.zeros((Nelec, dim-Nelec))
  F_ov = F[:Nelec, Nelec:]
  f = np.zeros(((Nelec)*(dim-Nelec)))
  ia = -1
  for i in range(0,Nelec):
    for a in range(0, dim - (Nelec)):
      ia += 1
      f[ia] = F_ov[i,a]
  
  return f
      

###############################
##       M   A   I   N       ##
###############################

start_time = time.time()

#### TEST SYSTEMS #### 

## Test Molecules
#mol, Nelec, name, basis, mult = 'H2_STO3G', 2, 'H2', 'STO-3G', 1
mol, Nelec, name, basis, mult = 'HeHplus_STO3G', 2, 'HeH+', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'CO_STO3G', 14, 'CO', 'STO-3G', 1 
#mol, Nelec, name, basis, mult = 'H2O_STO3G', 10, 'Water', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'Methanol_STO3G', 18, 'Methanol', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'Li_STO3G', 3, 'Lithium', 'STO-3G', 2
#mol, Nelec, name, basis, mult = 'O2_STO3G', 16, 'Oxygen', 'STO-3G', 3

######################



# Get integrals from files
Vnn, Vne, T, S, ERI = getIntegrals(mol)

# Number of basis functions dim, Nalpha, Nbeta
dim = len(S)
Na, Nb = calcAlphaBeta(Nelec, mult)

# Build Core Hamiltonian
h = T + Vne

# Set up initial Fock with core guess, and density at 0
eps_h, C_h = eigh(h)
F_a, F_b = h, h
P_a, P_b = np.zeros((dim,dim)), np.zeros((dim,dim))
C_a, C_b = np.zeros((dim,dim)), np.zeros((dim,dim))

# Form transformation matrix
s, Y = eigh(S)
s = np.diag(s**(-0.5))
X = np.dot(Y, np.dot(s, Y.T))

# Initialize variables
delta  = 1.0
conver = 1.0e-08
count  = 0
E0     = 0
P_old  = P_a + P_b
doNR   = True
doQN   = False


# Start main SCF loop
while delta > conver and count < 1000:
  count += 1
  old_E = E0
  
  # Build Fock Matrix and Hartree energy
  E0, F_a, F_b = buildFock(dim,P_a,P_b,h,ERI)
  print 'E0', E0

  # Fock mo basis transform
  F_a_mo = np.dot(C_a.T, np.dot(F_a, C_a))
  F_b_mo = np.dot(C_b.T, np.dot(F_b, C_b))  


  # ---  Quadratic Convergence  ---

  if count == 1: # Bypass first loop

    # Orthonormalize Fock alpha and beta
    F_a_oao = np.dot(X.T,np.dot(F_a, X))
    F_b_oao = np.dot(X.T,np.dot(F_b, X))
    eps_a, C_a_oao = eigh(F_a_oao)
    eps_b, C_b_oao = eigh(F_b_oao)
    C_a = np.dot(X, C_a_oao)
    C_b = np.dot(X, C_b_oao)
   
    # Initialize direction vectors
    D_a = np.array([])
    D_b = np.array([])
    
  
  elif count > 1:
    f_a   = getOVfvector(F_a_mo, Na, dim)
    f_b   = getOVfvector(F_b_mo, Nb, dim)
    deriv = max(LA.norm(f_a),LA.norm(f_b))
    print 'deriv ', deriv

    # Do Newton-Raphson step
    #if (deriv < 1e-02):
    if doNR:
      eriMO    = ao2mo(ERI, [C_a, C_b], False)
      eps_a    = np.diag(F_a_mo)
      eps_b    = np.diag(F_b_mo)
      A, B     = responseAB_UHF(eriMO, [eps_a,eps_b],[Na,Nb])
      M        = np.bmat([[A, B],[B, A]])

      # Solve Ax = b
      f_ab = np.append(f_a,f_b)
      D = np.linalg.solve(A+B,f_ab)
      D_a = D[:Na*(dim-Na)]
      D_b = D[Na*(dim-Na):]
      C_a_old = C_a
      C_b_old = C_b
      P_old = P_a + P_b   
      old_E = E0
      for alp in np.arange(1,0,-0.01):
        K_a = np.zeros((dim, dim))
        K_b = np.zeros((dim, dim))
        ia = -1
        for i in range(0, Na):
          for a in range(Na,dim): 
            ia += 1
            K_a[i,a] =  D_a[ia]
        ia = -1
        for i in range(0, Nb):
          for a in range(Nb,dim): 
            ia += 1
            K_b[i,a] =  D_b[ia]

        K_a = (-K_a + K_a.T)
        K_b = (-K_b + K_b.T)
        C_a = np.dot(C_a_old , expm(-alp*K_a))
        C_b = np.dot(C_b_old , expm(-alp*K_b))
        P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
        P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
        
        E0, F_a, F_b = buildFock(dim,P_a,P_b,h,ERI)
        if old_E > E0:
          print alp
          break
    
    elif doQN:
      print 'here'
    
      
    # Do steepest descent (Not working)
    '''
    else:
      D_a_old, D_b_old = D_a, D_b
      #D_a, D_b, = 0.0947*f_a, 0.0947*f_b #triplet water
      D_a, D_b, = f_a, f_b
      C_a_old = C_a
      C_b_old = C_b
      temp_a = np.zeros((dim, dim))
      temp_b = np.zeros((dim, dim))
      ia = -1
      for i in range(0, Na):
        for a in range(Na,dim): 
          ia += 1
          temp_a[i,a] =  D_a[ia]
      ia = -1
      for i in range(0, Nb):
        for a in range(Nb,dim): 
          ia += 1
          temp_b[i,a] =  D_b[ia]
      for alp in np.arange(1,0,-0.01):
        C_a = C_a_old * expm(alp*temp_a)
        C_b = C_b_old * expm(alp*temp_b)
        F_a_mo = np.dot(C_a.T, np.dot(F_a, C_a))
        F_b_mo = np.dot(C_b.T, np.dot(F_b, C_b))  
        eps_a    = np.diag(F_a_mo)
        eps_b    = np.diag(F_b_mo)
        if (LA.norm(eps_a) < LA.norm(eps_a_old)) and (LA.norm(eps_b) < LA.norm(eps_b_old)):
          break
      print 'alp',alp
      D_a, D_b = alp*D_a, alp*D_b
    '''
    ''' 
    # Create orbital rotation matrix K and U (OLD_QC!)
    K_a = np.zeros((dim, dim))
    K_b = np.zeros((dim, dim))
    ia = -1
    for i in range(0, Na):
      for a in range(Na,dim): 
        ia += 1
        K_a[i,a] =  D_a[ia]
    ia = -1
    for i in range(0, Nb):
      for a in range(Nb,dim): 
        ia += 1
        K_b[i,a] =  D_b[ia]
    K_a = (-K_a + K_a.T)
    K_b = (-K_b + K_b.T)
    U_a = expm(-K_a)
    U_b = expm(-K_b)

    # Rotate MO coeffs
    C_a = np.dot(C_a, U_a)
    C_b = np.dot(C_b, U_b)
    '''
  # Normalize C
  norm_a = np.sqrt(np.diag(np.dot(np.dot(np.transpose(C_a),S),C_a) ))
  norm_b = np.sqrt(np.diag(np.dot(np.dot(np.transpose(C_b),S),C_b) ))
  for i in range(0,dim):
    C_a[:,i] = C_a[:,i]/norm_a[i]
    C_b[:,i] = C_b[:,i]/norm_b[i]
  
  # Update Density Matrix
  # P_old = P_a + P_b   
  P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
  P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
  
  # Compute change in density matrix
  delta = deltaP((P_a + P_b), P_old)



# Get Spin Expectation Value
# <S^2> = 1/4 * [(Tr[Pmz*S])^2 + 2*Tr[Pmz*S*Pmz*S]]
Pmz = P_a - P_b
spin_expect = 1/4.0 * ((np.trace(np.dot(Pmz,S)))**2 + 2*np.trace(np.dot(Pmz,S).dot(Pmz).dot(S)))

# Final diagnolization to keep F_mo diagonal (Pseudocanonicalization)
F_a_oao = np.dot(X.T,np.dot(F_a, X))
F_b_oao = np.dot(X.T,np.dot(F_b, X))
eps_a, C_a_oao = eigh(F_a_oao)
eps_b, C_b_oao = eigh(F_b_oao)
C_a = np.dot(X, C_a_oao)
C_b = np.dot(X, C_b_oao)

elapsed_time = time.time() - start_time



### Print results ###
print ''
print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
print '                   R e s u l t s            \n'
print 'Molecule: ' + name 
print 'Basis: ' + basis
print 'Multiplicity: ' + str(mult)
print 'E(SCF) = ' + str(E0) + ' a.u.'
print '<S'+u'\xb2'+ '> =', spin_expect
print 'SCF iterations: ' + str(count)
print 'Elapsed time: ' + str(elapsed_time) + ' sec\n'
print 'MO Coeffs (alpha) = \n' + np.array_str(C_a)
#print 'Fock Matrix (alpha) = \n' + np.array_str(F_a)
#print 'Fock Matrix (beta) = \n' + np.array_str(F_b)
#print 'Density Matrix (alpha) = \n' + np.array_str(P_a)
#print 'Density Matrix (beta) = \n' + np.array_str(P_b)
print 'Orbital Energies (alpha) = \n' + str(eps_a) + '\n'
print 'Orbital Energies (beta) = \n' + str(eps_b) + '\n'
print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
print ''

# Convert AO to MO orbital basis
#print '-------------------------'
eriMO_u = ao2mo(ERI, [C_a, C_b], False)
#mp2(eriMO, eps, Nelec)
#print responseAB_UHF(eriMO, [eps_a, eps_b], Nelec)
TDHF(eriMO_u, [eps_a, eps_b], [Na, Nb], False)
#print '-------------------------'











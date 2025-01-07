import numpy as np
import random
from scipy.linalg import eigh

#np.set_printoptions(formatter={'float': lambda x: "{0:8.4f}".format(x)},linewidth=500)

P =[[ 2.0000,  0.6286,  0.0256], 
    [ 0.6286,  1.9987,  0.5238],
    [ 0.0256,  0.5238,  1.8248]]

S =[[1.0000,  0.3142,  0.0136],
    [0.3142,  1.0000,  0.2542],
    [0.0136,  0.2542,  1.0000]]

nshell = 3
nbas   = 3
P = np.asarray(P)
S = np.asarray(S)
l  = 0
nm = 2*l+1
imap = [i for i in range(nshell)]

P_av = np.zeros((nshell,nshell))
S_av = np.zeros((nshell,nshell))

for i in range(nshell):
  for j in range(nshell):
    ifirst = imap[i]
    jfirst = imap[j]
    for im in range(nm):
      P_av[i,j] += P[ifirst+im,jfirst+im]
      S_av[i,j] += S[ifirst+im,jfirst+im]

P_av/=nm
S_av/=nm

print("P","\n",P)
print("S","\n",S)
print("P_av","\n",P_av)
print("S_av","\n",S_av)

eigvals=np.zeros(nshell)
eigvecs=np.zeros((nshell,nshell))

eigvals,eigvecs = eigh(P,S)
print("eigvals","\n",eigvals)
print("eigvecs","\n",eigvecs)

eigvals=np.zeros(nshell)
eigvecs=np.zeros((nshell,nshell))

eigvals,eigvecs = eigh(P_av,S_av)
print("eigvals","\n",eigvals)
print("eigvecs","\n",eigvecs)

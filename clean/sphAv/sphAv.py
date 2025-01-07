#Purpose is to perform spherically symmetry averaging of hypothetical pair of p-shells

import numpy as np
from scipy.linalg import eigh
 
#np.set_printoptions(formatter={'float': lambda x: "{0:8.4f}".format(x)},linewidth=500)

#Generate matrix here
nbas   = 6
nshell = 2
l      = 1
nm     = 2*l+1
imap   = np.zeros(nshell).astype(int)

for i in range(nshell):
  if i==0:
    imap[i] = 0
  else:
    imap[i] = i+nm-1

print("imap",imap)

P = np.array([[ 1.9944,  0.0000,  0.0000,  0.5919,  0.0000,  0.0000],
              [ 0.0000,  1.9944,  0.0000,  0.0000,  0.5919,  0.0000],
              [ 0.0000,  0.0000,  1.9944,  0.0000,  0.0000,  0.5919],
              [ 0.5919,  0.0000,  0.0000,  1.4625,  0.0000,  0.0000],
              [ 0.0000,  0.5919,  0.0000,  0.0000,  1.4625,  0.0000],
              [ 0.0000,  0.0000,  0.5919,  0.0000,  0.0000,  1.4625]])

S =np.array([[ 1.0000,  0.0000,  0.0000,  0.2685,  0.0000,  0.0000],  
             [ 0.0000,  1.0000,  0.0000,  0.0000,  0.2685,  0.0000],
             [ 0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.2685],
             [ 0.2685,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
             [ 0.0000,  0.2685,  0.0000,  0.0000,  1.0000,  0.0000],
             [ 0.0000,  0.0000,  0.2685,  0.0000,  0.0000,  1.0000]])

#P=np.asarray(P)
#S=np.asarray(S)
print("P","\n",P)
print("S","\n",S)

eigval,eigvec = eigh(P,S)
print("eigval","\n",eigval)
print("eigvec","\n",eigvec)

P_al   = np.zeros((nshell,nshell))
S_al   = np.zeros((nshell,nshell))

#Here averaging with shellxshell density (a,l) block
for i in range(nshell):
  for j in range(nshell):
    ifirst = imap[i]
    jfirst = imap[j]
    for im in range(nm):
      P_al[i,j] += P[ifirst+im,jfirst+im]
      S_al[i,j] += S[ifirst+im,jfirst+im]

P_al/=nm 
S_al/=nm

print("P_al","\n",P_al)
print("S_al","\n",S_al)

eigval = np.zeros(nshell)
eigvec = np.zeros((nshell,nshell))

eigval,eigvec = eigh(P_al,S_al)
print("eigval","\n",eigval)
print("eigvec","\n",eigvec)

P=np.zeros((nbas,nbas))
S=np.zeros((nbas,nbas))
print("P","\n",P)
print("S","\n",S)
for i in range(nshell):
  for j in range(nshell):
    ifirst = imap[i]
    jfirst = imap[j]
    for im in range(nm):
       P[ifirst+im,jfirst+im] = P_al[i,j]
       S[ifirst+im,jfirst+im] = S_al[i,j]

print("P","\n",P)
print("S","\n",S)

eigval,eigvec = eigh(P,S)
print("eigval","\n",eigval)
print("eigvec","\n",eigvec)

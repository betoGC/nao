############################################################################
########################## TO - DO  ########################################
############################################################################

np.set_printoptions(threshold=sys.maxsize,formatter={'float': lambda x: "{0:20.4f}".format(x)},linewidth=400)

#def nao_step_2(Pp,Sp):
#  N_st    = np.zeros((nbas,nbas), dtype=dtype)
#  W_st    = np.zeros(nbas, dtype=dtype)
#  rho_av  = np.zeros((nbas,nbas), dtype=dtype)
#  s_av    = np.zeros((nbas,nbas), dtype=dtype)
#
#  print("===== After load P check zeros =====","\n",Pp)   
# 
#  #Calling spherical symmetry averaging
#  rho_av,s_av = load_average(Pp,Sp)
#  #Calling general eigenvalue problem solver
#  W_st,N_st = GenEig(rho_av,s_av)
#
#  return W_st,N_st,rho_av,s_av
#
#def load_average(Pp,Sp):
#  #Ap contains the density to average
#  #Bp contains the overlap to average
#  Ap   = np.zeros((nbas,nbas),dtype=dtype)
#  Bp   = np.zeros((nbas,nbas),dtype=dtype)
#  W_av = np.zeros(nbas,dtype=dtype)
#  N_av = np.zeros((nbas,nbas),dtype=dtype)
#  idx  = [False for i in range(nbas)]
#  idx  = np.array(idx).astype(bool)
#
#  #visiting atom,l blocks
#  for iatom in range(natom):
#    lstop = lmax[iatom]+1
#    for lctrl in range(lstop):
#      ml_up = 2*lctrl+1
#      for ibas in range(nbas):
#        if center[ibas] == iatom and func_am[ibas] == lctrl:
#          idx[ibas] = True
#        else:
#          idx[ibas] = False
#      
#      ntrue = idx.sum()
#      print("lctrl",lctrl,"ntrue",ntrue)
#      cA    = np.zeros((ntrue,ntrue))
#      cB    = np.zeros((ntrue,ntrue))
#      cn_av = np.zeros((ntrue,ntrue))
#      cw_av = np.zeros(ntrue)
#      #Copying local block of angularity lctrl
#      if lctrl >= 0:
#        print("lctrl",lctrl)
#        nr = 0
#        for ibas in range(nbas):
#         if idx[ibas] == True:
#            nc = 0
#            for jbas in range(nbas):
#              if idx[jbas] == True:
#                cA[nr,nc] = Pp[ibas,jbas].copy()     
#                cB[nr,nc] = Sp[ibas,jbas].copy()
#                nc+=1
#            nr+=1
#        #Here averaging on a local block of same angularity
#        nsh = int(ntrue/ml_up)                #count shells of same angularity read 
#        print("nsh",nsh)
#        shell_idx = [0 for i in range(nsh)]   #Start index for shell of given l in a block
#        for ish in range(nsh):
#          shell_idx[ish] = ish*ml_up
#        print("shell_idx in load_average",shell_idx)
#        sum_ra=np.zeros(ml_up, dtype=dtype)
#        sum_rb=np.zeros(ml_up, dtype=dtype)
#        sum_ca=np.zeros(ml_up, dtype=dtype)
#        sum_cb=np.zeros(ml_up, dtype=dtype)
#        w_al  =np.zeros(nsh, dtype=dtype)
#        n_al  =np.zeros((nsh,nsh), dtype=dtype)
#        p_al  =np.zeros((nsh,nsh), dtype=dtype)
#        s_al  =np.zeros((nsh,nsh), dtype=dtype)
#        print("===== cA before average =====","\n",cA)
#        print("===== cB before average =====","\n",cB)
#        for i in range(nsh):
#          for j in range(nsh):
#            ifirst = shell_idx[i]
#            jfirst = shell_idx[j]
#            for im in range(ml_up):
#              sum_ra[im] = np.sum(cA[ifirst:ifirst+ml_up,jfirst+im])
#              sum_rb[im] = np.sum(cB[ifirst:ifirst+ml_up,jfirst+im])
#              sum_ca[im] = np.sum(cA[ifirst+im,jfirst:jfirst+ml_up])
#              sum_cb[im] = np.sum(cB[ifirst+im,jfirst:jfirst+ml_up])
#            for im in range(ml_up):
#              for jm in range(ml_up):
#                p_al[i,j] += cA[ifirst+im,jfirst+jm].copy()/ml_up 
#                s_al[i,j] += cB[ifirst+im,jfirst+jm].copy()/ml_up
#            for im in range(ml_up):
#              for jm in range(ml_up):
#                if im == jm:
#                  cA[ifirst+im,jfirst+im] = (sum_ra[im]+sum_ca[im])/(2.0)
#                  cB[ifirst+im,jfirst+im] = (sum_rb[im]+sum_cb[im])/(2.0)
#                else:
#                  cA[ifirst+im,jfirst+jm] = 0.
#                  cB[ifirst+im,jfirst+jm] = 0.
#        print("===== cA after average =====","\n",cA)
#        print("===== cB after average =====","\n",cB)
#        print("===== p_al after average =====","\n",p_al)
#        print("===== s_al after average =====","\n",s_al)
#        x,y = np.linalg.eig(s_al)
#        xxidx = x.argsort()[::-1]
#        x = x[xxidx]
#        y = y[:,xxidx]
#        x = np.power(x,-0.5000000)
#        x = np.diag(x)
#        x = y @ x @ y.T
#        p_al = x.T @ p_al @ x
#        s_al = x.T @ s_al @ x
#        w_al,n_al = Jacobi(p_al)
#        alidx = w_al.argsort()[::-1]
#        w_al = w_al[alidx] 
#        n_al = n_al[:,alidx] 
#        n_al = x @ n_al
#        print("===== w_al =====","\n",w_al)
#        print("===== n_al =====","\n",n_al)
#        for i in range(nsh):
#          for j in range(nsh):
#            ifirst = shell_idx[i]
#            jfirst = shell_idx[j]
#            for im in range(ml_up):
#              cw_av[ifirst+im] = w_al[i]
#              for jm in range(ml_up):
#                if im == jm:
#                  cn_av[ifirst+im,jfirst+jm] = n_al[i,j]
#        #Copying back averaged local block in the original matrix
#        nr=0
#        for ibas in range(nbas):
#          if idx[ibas] == True:
#            W_av[ibas] = cw_av[nr].copy()
#            nc=0
#            for jbas in range(nbas):
#              if idx[jbas] == True:
#                Ap[ibas,jbas] = cA[nr,nc].copy()
#                Bp[ibas,jbas] = cB[nr,nc].copy()
#                N_av[ibas,jbas] = cn_av[nr,nc].copy()
#                nc += 1
#            nr += 1
#  #print("===== W_av after average =====")
#  #print(W_av)
#  #print("===== N_av after average =====")
#  #prntMtrx(N_av)
#  return Ap,Bp,W_av,N_av
#
#def GenEig(P_av,S_av):
#  W_ge = np.zeros(nbas)
#  N_ge = np.zeros((nbas,nbas), dtype=dtype)
#  idx  = [False for i in range(nbas)]
#  idx  = np.array(idx).astype(bool)
#  for iatom in range(natom):
#    lstop = lmax[iatom]+1             #python counting interval [0,n)
#    for lctrl in range(lstop):        #visit angularity in order s=0,p=1,...
#      ml_up = 2*lctrl+1
#      for ibas in range(nbas):
#        if center[ibas] == iatom and func_am[ibas] == lctrl:
#          idx[ibas] = True
#        else:
#          idx[ibas] = False
#      ntrue=idx.sum()
#      nsh = int(ntrue/ml_up)
#      shell_idx = [0 for i in range(nsh)]   #starting index in local block
#      for ish in range(nsh):
#        shell_idx[ish] = ish*ml_up
#      #print("shell_idx",shell_idx)
#      #print("Number of functions in (iatom,l):(",iatom,lctrl,") block = ",ntrue)
#      cP_av  = np.zeros((ntrue,ntrue))
#      cS_av  = np.zeros((ntrue,ntrue))
#      cP_old = np.zeros((ntrue,ntrue))
#      cS_old = np.zeros((ntrue,ntrue))
#      cr = 0
#      for ibas in range(nbas):
#        if idx[ibas] == True:
#          cc = 0
#          for jbas in range(nbas):
#            if idx[jbas] == True:
#              cP_av[cr,cc] = P_av[ibas,jbas].copy()
#              cS_av[cr,cc] = S_av[ibas,jbas].copy()
#              cc+=1 
#          cr+=1 
#      #print("=====iatom,l block",iatom,lctrl,"cP_av =====","\n",cP_av)
#      #print("=====iatom,l block",iatom,lctrl,"cS_av =====","\n",cS_av)
#      cP_old = cP_av.copy()
#      cS_old = cS_av.copy()
#      x,y = Jacobi(cS_av)
#      x = np.power(x,-0.5)
#      x = np.diag(x)
#      x = y @ x @ y.T
#      cP_av = x.T @ cP_av @ x
#      eigvals,eigvecs = Jacobi(cP_av)
#      jdx = eigvals.argsort()[::-1]
#      eigvals = eigvals[jdx]
#      eigvecs = eigvecs[:,jdx]
#      eigvecs = x@eigvecs
#      #print("===== iatom,lblock eigvals =====",iatom,lctrl,"\n",eigvals)
#      #print("===== iatom,lblock eigvecs =====",iatom,lctrl,"\n",eigvecs)
#      cr = 0
#      for ibas in range(nbas):
#        if idx[ibas] == True:
#          cc = 0
#          for jbas in range(nbas):
#            if idx[jbas] == True:
#              if ntrue > 0:
#                W_ge[ibas] = eigvals[cr].copy()
#                N_ge[ibas,jbas] = eigvecs[cr,cc].copy()
#              cc+=1
#          cr+=1 
#      print("===== Density simultaneous Diagonal blocks atom,l =====","\n",np.matmul(eigvecs.T,np.matmul(cP_old,eigvecs)))
#      print("===== Overlap simultaneous Diagonal blocks atom,l =====","\n",np.matmul(eigvecs.T,np.matmul(cS_old,eigvecs)))
#  return W_ge,N_ge
#
#def nmb_count(atNum,iatom):
#  iatnum=atNum[iatom]
#  if iatnum >= 87:
#    exit("Principal quantum number n > 6 to be implemented")
#  elif iatnum >= 55:
#    imbcount = 43
#  elif iatnum >= 37:
#    imbcount = 27
#  elif iatnum >= 19:
#    imbcount = 18
#  elif iatnum >= 11:
#    imbcount = 9
#  elif iatnum >= 5:
#    imbcount = 5
#  elif iatnum >= 3:
#    imbcount = 2
#  else:
#    imbcount = 1
#  print("iatom,iatnum,imbcount",iatom,iatnum,imbcount)
#  return imbcount 
#
#def createNMBmask(w,nbas,ibasismap,atNum):
#  #Purpose: Define logical array indexing to NMB according to occupations
#  wsort = w.copy()
#  mask = [False for i in range(nbas)] #Initializing
#  imbcount=0
#  fidx=0
#  for iatom in range(natom):
#    imbcount = nmb_count(atNum,iatom)
#    ifirst = ibasismap[iatom]                              #start of the funcs for this atom
#    ilast  = ibasismap[iatom+1]-1                          #end of the funcs for this atom
#    res=np.sort(wsort[ifirst:ilast+1],kind='heapsort')     #occupations vector
#    res=res[::-1]                                          #ordering largest to lowest
#    fidx=imbcount-1
#    thresh = res[int(fidx)].copy()-1.0e-4                  #occupations up to the numerical threshold
#    #print("imbcount,fidx,res,thres",imbcount,fidx,res,thresh)
#    #print("wsort",wsort)
#    #print("res",res)
#    for i in range(ifirst,ilast+1):                        #Seek for the largest occupation on each atom
#      value = w[i].copy()                                  #NMB basis functions are selected
#      if (value >= thresh) or (imbcount>0):                #untill the imbcount decreases to zero
#        mask[i] = True
#        imbcount -= 1
#      else:
#        mask[i] = False
#    #print(mask)
#  return mask
#
#def partBasis(mask_pb,ntrue,nfalse,imap_pb):
#  #The goal is to make a partition of the basis in two sets 
#  #imap contains indexing in order to split the basis
#  imap_pb[:]=0
#  nb=len(mask_pb)
#  ntrue = 0 
#  for ibas in range(nb):
#    if (mask_pb[ibas]):
#      ntrue += 1 
#      imap_pb[ibas]=ntrue
#  nfalse = 0
#  for ibas in range(nb):
#    if (not mask_pb[ibas]):
#      nfalse += 1 
#      imap_pb[ibas]=ntrue+nfalse
#  return imap_pb,ntrue,nfalse
#
#def mapMatrix(A,imap):
#  #Define orbitals shuffle from imap as a unitary tranformation 
#  #Forward: newM=A*oldM*A.T
#  #Reverse: oldM=A.T*newM*A
#  nb=len(A[:,0])
#  dummy=np.zeros((nb,nb))
#  for j in range(nb):
#    dummy[:,j] = A[:,imap[j]-1]
#  return dummy
#
#def symmOrthMat(Sw,w):
#  #Weigthed symmetric orthogonalization
#  #The weigths w_i are occupations stored in W
#  #the eigenvalues colected from \tilde(P)N = W\tilde(S)N 
#  nthresh = 1.0e-12
#  rank=len(Sw[0,:])
#  Ow=np.zeros((rank,rank))
#  for i in range(rank):
#    for j in range(rank):
#      Ow[i,j] = w[i]*Sw[i,j]*w[j]
#  print("Ow","\n",Ow)
#  print("same?","\n",Sw*np.outer(w,w))
#  x,y=eigh(Ow)
#  jdx = x.argsort()[::-1]
#  x = x[jdx]
#  y = y[:,jdx]
#  for i in range(rank):
#    if (x[i] < nthresh):
#      print("Warning: Found small eigenvalue in Ow")
#      #exit()
#  print("x","\n",x)
#  print("y","\n",y)
#  x=np.power(x,-0.5)
#  x=np.diag(x)
#  x=np.matmul(y,np.matmul(x,y.T))
#  print("Sw^(-1/2)*Sw*Sw^(-1/2) = E ?","\n",np.matmul(np.matmul(x,Ow),x.T))
#  for i in range(rank):
#    for j in range(rank):
#      x[i,j]=w[i]*x[i,j]
#  print("w(wSw)^(-1/2)","\n",x)
#  return x
#
#def GenOverlap(N_go,S_go):
#  #Purpose is to compute the MOs overlap matrix
#  nrows=len(N_go[:,0])
#  ncols=len(N_go[0,:])
#  nfunc=len(S_go[0,:])
#  print("nrows,ncols,nfunc",nrows,ncols,nfunc)
#  GenS = np.zeros((nrows,ncols))
#  for i in range(nrows):
#    for j in range(ncols):
#      factor_go = 0.0
#      for alpha in range(nrows):
#        for beta in range(ncols):
#          factor_go += N_go[alpha,i]*N_go[beta,j]*S_go[alpha,beta]
#      GenS[i,j] = factor_go
#  return GenS
#
#def S_ij(coeff1,coeff2,S_ab):
#  #Purpose is to compute a specific i,j-th element of MOs overlap matrix 
#  nfun = len(S_ab[:,0])
#  number = 0.0
#  for alpha in range(nfun):
#    for beta in range(nfun):
#      number += coeff1[alpha]*coeff2[beta]*S_ab[alpha,beta] #fixme \sum_ab coeff1_ai^{*}.coeff2_bj.S_ab
#  return number

# Function to print the matrix in chunks of 10 columns
def prntMtrx(matrix, chunk_size=10):
  num_columns = matrix.shape[1]
  print("")
  for start in range(0, num_columns, chunk_size):
    end = start + chunk_size
    header = [f"             [Col{i+1:>3}]" for i in range(start, min(end, num_columns))]
    print("     "+" ".join(header))
    for row_index, row in enumerate(matrix[:, start:end]):
      print(f"[Row{row_index+1:>4}] " + " ".join(f"{val:20.4f}" for val in row))

#Function to reorder a matrix P or S to Gaussian m_l ordering for real spherical harmonics
def stdGorder(A):
  #A contains the matrix to reorder
  Ap  = A.copy()
  cA  = np.zeros((nbas,nbas), dtype=dtype)
  idx = [False for i in range(nbas)]
  idx = np.array(idx).astype(bool)

  cA = Ap.copy()
  #visiting atom,l blocks
  for iatom in range(natom):
    lstop = lmax[iatom]+1
    for lctrl in range(lstop):
      ml_up = 2*lctrl+1
      for ibas in range(nbas):
        if center[ibas] == iatom and func_am[ibas] == lctrl:
          idx[ibas] = True
        else:
          idx[ibas] = False

      ntrue = np.sum(idx)
      nsh = int(ntrue/ml_up)                #count shells of same angularity
      shell_idx = [0 for i in range(nsh)]   #starting index in local block
      for ish in range(nsh):
        shell_idx[ish] = ish*ml_up
      #print("shell_idx in stdGorder",shell_idx)

      order=np.zeros(ntrue).astype(int)
      nb = 0
      for ibas in range(nbas):
        if center[ibas] == iatom and func_am[ibas] == lctrl:
          if idx[ibas] == True:
            order[nb] = ibas
            nb += 1
      #print("GaussianOrdering","iatom",iatom,"ntrue",ntrue,"lctrl",lctrl)
      #print("order before",order)
      if lctrl == 0 and lctrl < lstop:
        shuffle = np.zeros(nbas)
        shuffle = shuffle.astype(int)
        shuffle = [i for i in range(1,nbas+1)] #sequence for full matrix dimensions
        E = np.identity(nbas)                  #
        M = mapMatrix(E,shuffle)               #permutes the identity to define the indexing M matrix
        cA = np.matmul(M.T,np.matmul(cA,M))
      elif lctrl == 1:
        shuffle = np.zeros(nbas)
        shuffle = shuffle.astype(int)
        stdOrder = np.array([[0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0]], dtype=np.float64)
        for ish in range(nsh):
          first = shell_idx[ish]
          last = first+ml_up
          order[first:last] = np.matmul(stdOrder,order[first:last])
        # print("p-shells","f",first,"l",last,"order",order)
        n=0
        for ibas in range(nbas):
          if idx[ibas] == False:
            shuffle[ibas] = ibas+1
          else:
            shuffle[ibas] = order[n]+1
            n+=1
        E  = np.identity(nbas)
        M  = mapMatrix(E,shuffle) 
        #print("Mapping the matrix","\n",M)
        cA = np.matmul(M.T,np.matmul(cA,M))
      elif lctrl == 2: 
        shuffle = np.zeros(nbas)
        shuffle = shuffle.astype(int)
        stdOrder = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        for ish in range(nsh):
          first = shell_idx[ish]
          last = first+ml_up
          order[first:last] = np.matmul(stdOrder,order[first:last])
        order = order.astype(int)  
        n=0
        for ibas in range(nbas):
          if idx[ibas] == False:
            shuffle[ibas] = ibas+1
          else:
            shuffle[ibas] = order[n]+1
            n+=1
        E  = np.identity(nbas)
        M  = mapMatrix(E,shuffle)  
        cA = np.matmul(M.T,np.matmul(cA,M))
      elif lctrl == 3: 
        shuffle = np.zeros(nbas)
        shuffle = shuffle.astype(int)
        stdOrder = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        for ish in range(nsh):
          first = shell_idx[ish]
          last = first+ml_up
          order[first:last] = np.matmul(stdOrder,order[first:last])
        order = order.astype(int)  
        n=0
        for ibas in range(nbas):
          if idx[ibas] == False:
            shuffle[ibas] = ibas+1
          else:
            shuffle[ibas] = order[n]+1
            n+=1
        E  = np.identity(nbas)
        M  = mapMatrix(E,shuffle)  
        cA = np.matmul(M.T,np.matmul(cA,M))
  return cA      

#def Jacobi(a,tol = 1.0e-8): # Jacobi method
#
#    def maxElem(a): # Find largest off-diag. element a[k,l]
#        n = len(a)
#        aMax = 0.0
#        k = 0
#        l = 0
#        for i in range(n-1):
#            for j in range(i+1,n):
#                if abs(a[i,j]) >= aMax:
#                    aMax = abs(a[i,j])
#                    k = i; l = j
#        return aMax,k,l
#
#    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
#        n = len(a)
#        aDiff = a[l,l] - a[k,k]
#        if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
#        else:
#            phi = aDiff/(2.0*a[k,l])
#            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
#            if phi < 0.0: t = -t
#        c = 1.0/sqrt(t**2 + 1.0); s = t*c
#        tau = s/(1.0 + c)
#        temp = a[k,l]
#        a[k,l] = 0.0
#        a[k,k] = a[k,k] - t*temp
#        a[l,l] = a[l,l] + t*temp
#        for i in range(k):      # Case of i < k
#            temp = a[i,k]
#            a[i,k] = temp - s*(a[i,l] + tau*temp)
#            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
#        for i in range(k+1,l):  # Case of k < i < l
#            temp = a[k,i]
#            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
#            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
#        for i in range(l+1,n):  # Case of i > l
#            temp = a[k,i]
#            a[k,i] = temp - s*(a[l,i] + tau*temp)
#            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
#        for i in range(n):      # Update transformation matrix
#            temp = p[i,k]
#            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
#            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
#        
#    n = len(a)
#    maxRot = 5*(n**2)       # Set limit on number of rotations
#    p = identity(n)*1.0     # Initialize transformation matrix
#    for i in range(maxRot): # Jacobi rotation loop 
#        aMax,k,l = maxElem(a)
#        if aMax < tol: return diagonal(a),p
#        rotate(a,p,k,l)
#    print('Jacobi method did not converge')

######## Here we start performing an OPT-RHF/3-21G calculation for H2######
######## FIXME:Replace with a callback to any ESS  ####################
import math
#import psi4 
import glob
import os
import numpy as np
import scipy as scipy
import sys 
from scipy.linalg import eigh
from numpy import array,identity,diagonal
from math import sqrt

dtype = np.float64
in_path  = './XYZb/'
out_path = './OUTb/'

files = os.listdir(in_path)

for a_molecule in files:
  clean()
  filen = in_path+a_molecule
  fileo = out_path+a_molecule+'.out'
  with open(filen,'r') as f:
    geometry_data = f.read()
    mol = psi4.geometry(f"""
    0 1
    {geometry_data}
    units angstrom
    """
    )
  f.close()
  mol.fix_orientation(True)
  mol.fix_com(True)
  mol.reset_point_group('c1')
  mol.update_geometry()
  natom=mol.natom()
  atNum=np.zeros(natom).astype(int)
  atNuCh=np.zeros(natom)
  coords=np.zeros((natom,3))
  with open(fileo, 'w') as file_out:
    sys.stdout = file_out
    print("molecule",a_molecule)
    for iatom in range(natom):
      atNum[iatom] = mol.ftrue_atomic_number(iatom)
      atNuCh[iatom] = mol.Z(iatom)
      coords[iatom,0] = mol.fx(iatom)
      coords[iatom,1] = mol.fy(iatom)
      coords[iatom,2] = mol.fz(iatom)
    #set basis sto-3g

    set reference rhf
    set puream true
    set print_mos true
    set s_orthogonalization symmetric
    set g_convergence GAU_VERYTIGHT    

    ops = {'basis': 'userbas',
          'scf_type': 'direct',
          'e_convergence': 1.00e-8,
          'd_convergence': 1.00e-8,
    }

    psi4.set_options(ops)
    energy,wfn = psi4.energy('SCF',return_wfn=True)
    
    nmo=wfn.nmo()
    nso=wfn.nso()
    bsb=wfn.get_basisset('ORBITAL')
    bsb.print_detail_out()
    bsb.print_out()
    nbas=bsb.nbf()
    mints = psi4.core.MintsHelper(wfn.basisset())
    Spsi=mints.ao_overlap()
    Upsi=psi4.core.Matrix(nmo,nmo)
    seig=psi4.core.Vector(nmo)
    S=np.asarray(Spsi)
    for iatom in range(natom):
      print("System:, atNum:",atNum[iatom], "coords", coords[iatom])
    print("===== Original AO-S matrix =====")
    prntMtrx(S) 
    #Coefficients, Density, and definition of RDMA
    CApsi = wfn.Ca()
    CBpsi = wfn.Cb()
    DApsi = wfn.Da_subset("AO")
    DBpsi = wfn.Db_subset("AO")
    RDMA  = psi4.core.Matrix(nmo,nmo);
    RDMB  = psi4.core.Matrix(nmo,nmo);
    DA    = np.asarray(DApsi)
    DB    = np.asarray(DBpsi)
    CA    = np.asarray(CApsi)
    CB    = np.asarray(CBpsi)
    #################Setting up variables for the NAO transformation ##############
    nbas=len(DA[:,0])
    N_st                  = np.zeros((nbas,nbas), dtype=dtype)                
    W_st                  = np.zeros(nbas)
    ibasismap_st          = np.zeros(natom+1).astype(int)        #contains the indexes of the first basis function on a given atom
    nshell                = np.zeros(natom).astype(int)          #contains the number of different shells on a given atom
    lmax                  = np.zeros(natom).astype(int)          #contains the max subshell angularity found in an atom
    center                = np.zeros(nbas).astype(int)
    func_am               = np.zeros(nbas).astype(int)
    ibasismap_st[natom]   = nbas
    ##############Setting up indexing arrays for spherical_averaging#############                               
    print("nbas",nbas)
    ibas = 0
    for iatom in range(natom):
      nshell[iatom]=bsb.nshell_on_center(iatom)
      ibasismap_st[iatom] = ibas
      #print("iatom:",iatom,"nshell[iatom]:",nshell[iatom])
      for ishell in range(nshell[iatom]):
        nfun = bsb.shell(iatom,ishell).nfunction
        for ifun in range(nfun):
          center[ibas]   = iatom 
          func_am[ibas]  = bsb.shell(iatom,ishell).am
          #print("func_am[ibas]",func_am[ibas])
          ibas+=1
    #print("ibasismap",ibasismap)
    nsh         = np.sum(nshell)                               
    shellmap    = np.zeros(nsh)
    ilmap       = np.zeros(nsh)
    ibasisremap = np.zeros(nsh)
    for ishell in range(nsh):
      shellmap[ishell] = bsb.shell_to_basis_function(ishell)
      ilmap[ishell]    = bsb.shell(ishell).am
 
    ish=0
    for iatom in range(natom):
      lsh=nshell[iatom]
      lmax[iatom] = max(ilmap[ish:ish+lsh])
      ish = ish+nshell[iatom]
      #print("iatom:",iatom,"max angularity:",lmax[iatom])

    ########### Finished to set up variables ####################################

    #print("======= Density in the non-orthogonal representation ============")
    P_psi=np.zeros((nbas,nbas), dtype=dtype)
    result=np.zeros((nbas,nbas), dtype=dtype)
    P_psi=DA+DB
    print(" ===== CA ===== ")
    prntMtrx(CA) 
    print(" ===== CB ===== ")
    prntMtrx(CB) 
    print(" ===== P_psi ===== Dalpha+Dbeta ")
    prntMtrx(P_psi)
    
    #Rebuilt density in the non-orthogonal HF basis
    occ=np.sum(atNum)/2 #Expected occupied orbitals in RHF
    occ = int(occ)
    for i in range(occ):
      result+=np.einsum('i,j->ij',CA[:,i],CA[:,i])
    
    Pmn = 0.0
    Pmn = 2.0*result
    
    #are both the same matrix?
    print("===== P_mu,nu local =====")
    prntMtrx(Pmn) 
    print("===== P_mu,nu psi4 =====")
    prntMtrx(P_psi) 
    ### Bond order matrix projection into AO basis P=SDS pre-NAO-step1 ###
    result2=np.matmul(S,np.matmul(P_psi,S))
    P=result2.copy()

    #worktrsh = 1.0e-8
    #for i in range(nbas):
    #  for j in range(nbas):
    #    if abs(P[i,j]) < worktrsh:
    #      P[i,j] = 0.
    #    if abs(S[i,j]) < worktrsh:
    #      S[i,j] = 0.
 
    #These are the matrices in psi4 std m_l ordering
    print("====== P^(AO) = SPS =====")
    prntMtrx(P)
    print("====== S^(AO) = S =====")
    prntMtrx(S)

    print("====== P^(AO) std gaussian m_l ordering SPS =====")
    #P=stdGorder(P)
    prntMtrx(P)

    print("====== S^(AO) std gaussian m_l ordering =====")
    #S=stdGorder(S)
    prntMtrx(S)
    continue
    ############ NAO-step2 - symmetry averaged matrices ###################
    # 
    #S_aoao        =  np.zeros((nbas,nbas), dtype=dtype)
    #P_aoao        =  np.zeros((nbas,nbas), dtype=dtype)
    #S_avav        =  np.zeros((nbas,nbas), dtype=dtype)
    #P_avav        =  np.zeros((nbas,nbas), dtype=dtype)

    #eigvalP_aoao  =  np.zeros(nbas)
    #eigvecP_aoao  =  np.zeros((nbas,nbas), dtype=dtype)

    #eigvalP_avav  =  np.zeros(nbas)
    #eigvecP_avav  =  np.zeros((nbas,nbas), dtype=dtype)

    #N_avpnao=np.zeros((nbas,nbas), dtype=dtype)
    #W_pnao=np.zeros(nbas)
    #pnboN = np.zeros((nbas,nbas), dtype=dtype)
    #pnboW = np.zeros(nbas, dtype=dtype)
    #P_aoao = P.copy()
    #S_aoao = S.copy()

    ##W_pnao,N_avpnao,P_avav,S_avav = nao_step_2(P_aoao,S_aoao)
    #P_avav,S_avav,pnboW,pnboN = load_average(P_aoao,S_aoao)

    #print("===== pnboW =====","\n",pnboW)
    #print("===== pnboN =====")
    #prntMtrx(pnboN) 

    #eigvalP_aoao,eigvecP_aoao = GenEig(P_aoao,S_aoao)
    #eigvalP_avav,eigvecP_avav = GenEig(P_avav,S_avav)

    #print("eigvalP_aoao","\n",eigvalP_aoao)
    #print("eigvalP_avav","\n",eigvalP_avav)
    #print("===== eigvecP_aoao =====")
    #prntMtrx(eigvecP_aoao)
    #print("===== eigvecP_avav =====")
    #prntMtrx(eigvecP_avav)
    #
    #continue
    ######Assemble transformation from AO to avAO
    #T1       = np.zeros((nbas,nbas), dtype=dtype)
    #T1       = np.matmul(eigvecP_avav,np.linalg.inv(eigvecP_aoao))
    #invT1    = np.matmul(eigvecP_aoao,np.linalg.inv(eigvecP_avav))

    #print("===== T1 =====")
    #prntMtrx(T1)
    #print("===== invT1*T1 =====")
    #prntMtrx(np.dot(invT1,T1))
    #print("===== Trial P_aoao -> P_avav =====")
    #prntMtrx(invT1 @ P_aoao @ T1)
    #print("===== P_avav =====")
    #prntMtrx(P_avav)
    #print("===== Trial S_aoao -> S_avav =====")
    #prntMtrx(invT1 @ S_aoao @ T1)
    #print("===== S_avav =====")
    #prntMtrx(S_avav)
    #
    #W_pnao   = eigvalP_avav.copy()
    #N_avpnao = eigvecP_avav.copy()
    #print("===== W_pnao =====","\n",W_pnao)
    #print("===== N_avpnao =====")
    #prntMtrx(N_avpnao) 

    ##N_avpnao=stdGorder(N_avpnao)
    #T2 = np.zeros((nbas,nbas), dtype=dtype)
    #T2 = N_avpnao.copy()
    ######Here Transformation of P, and S to the preNAO basis, end of pre_nao_step_2
    #Ppnao = np.zeros((nbas,nbas), dtype=dtype)
    #Spnao = np.zeros((nbas,nbas), dtype=dtype)
    #Ppnao = T2.T @ P_avav @ T2
    #Spnao = T2.T @ S_avav @ T2
    #print("===== Spnao =====")
    ##Spnao=stdGorder(Spnao)
    #prntMtrx(Spnao)
    #print("===== Ppnao =====")
    ##Ppnao=stdGorder(Ppnao)
    #prntMtrx(Ppnao)
    #continue
    ###########################################################################
    ##################### ===== Starting with step 3 ===== ####################
    #dummy    = np.zeros(nbas)
    #dummy    = W_pnao.copy()
    #Pscsc    = np.zeros((nbas,nbas), dtype=dtype)
    #Sscsc    = np.zeros((nbas,nbas), dtype=dtype)
    #N_aosc   = np.matmul(T1,N_avpnao)
    #N_aopnao = np.matmul(T1,N_avpnao)
    #      
    #nM=0
    #nR=0
    #      
    #aMask = [False for i in range(nbas)] #Initializing the boolean array for having a map
    #aMask=createNMBmask(dummy,nbas,ibasismap,atNum)
    #print("aMask beginning step 3\n",aMask)
    #      
    #ibasisremap = np.zeros(nbas).astype(int)             
    #ibasisremap,nM,nR = partBasis(aMask,nM,nR,ibasisremap)
    #     
    #A   =  np.identity(nbas)
    #M1  =  mapMatrix(A,ibasisremap)
    #print("Mapping matrix to (NMB|NRB) separation M1\n",M1)
    #print(" === N_aosc     === before (N_s,M1)","\n",N_aosc)
    #print(" === N_aopnao === before (N_pnao,M1)","\n",N_aopnao)
    #     
    #dummy  = np.matmul(dummy,M1.T)
    #N_aosc   = np.matmul(N_aosc,M1.T)
    #N_aopnao = np.matmul(N_aopnao,M1.T)
    #Ppnao  = np.matmul(M1,np.matmul(Ppnao,M1.T))
    #Spnao  = np.matmul(M1,np.matmul(Spnao,M1.T))
    #     
    #print("Minimal basis nM=",nM,"Minimal Rydberg nR=",nR)
    #print("dummy after remapping","\n",dummy)
    #print(" === N_aosc === After (N_s,M1)","\n",N_aosc)
    #print(" === N_pnao === After (N_pnao,M1)","\n",N_aopnao)
    #print(" === Ppnao  === After (M1,Ppnao,M1.T)","\n",Ppnao)
    #print(" === Spnao  === After (M1,Spnao,M1.T)","\n",Spnao)
    #print("NMB set","\n",N_aosc[:,0:nM])
    #print("NRB set","\n",N_aosc[:,nM:nbas])
    ####### Obtain the schmidt orthogonalization between NMB and NRB sets #####
    #fullS=GenOverlap(N_aosc,S)
    #print("===== Check general overlapping matrix =====","\n",fullS)
    ##    
    #Os = np.identity(nbas)
    #for j in range(nbas):
    #  for i in range(nM):
    #    if i<j:
    #      proj=S_ij(N_aopnao[:,j],N_aosc[:,i],S)
    #      norm=S_ij(N_aosc[:,i],N_aosc[:,i],S)
    #      if norm < 1.0e-15:
    #        print("Linear dependency found in i,j pair at step 3 Schmidt:",i,j)
    #        exit()
    #      Os[i,j] = proj/norm
    #      N_aosc[:,j] = N_aosc[:,j]-N_aosc[:,i]*Os[i,j]
    #for i in range(nM,nbas):
    #  norm_c = S_ij(N_aosc[:,i],N_aosc[:,i],S)
    #  norm_c = np.sqrt(1.0/norm_c)
    #  N_aosc[:,i] *= norm_c 
    #print(" ====== Recovering the NMB functions ===== ")
    #N_aosc[:,0:nM] = N_aopnao[:,0:nM].copy()
    #print(" ====== N^(aosc) Recovered NMB, orthogonal Rydberg-basis) ====== ","\n",N_aosc)
    #T3 = np.matmul(N_aosc,np.linalg.inv(N_aopnao))
    #print(" Os=N_aosc*inv(N_aopnao)*N_aopnao??? ","\n",np.matmul(T3,N_aopnao))
    #print(" ===== Os upper triangular transformation, proj/norm) ===== ","\n",Os)
    #fullS = GenOverlap(N_aosc,S)
    #print(" ===== Check general overlapping matrix ===== ","\n",fullS)
    #Pscsc  = np.matmul(np.matmul(N_aosc.T,P),N_aosc)
    #Sscsc  = np.matmul(np.matmul(N_aosc.T,S),N_aosc)
    #print(" ===== Ppnao^Sch projection ===== ","\n",Pscsc)
    #print(" ===== Spnao^Sch projection ===== ","\n",Sscsc)
    #Pscsc  = np.matmul(M1.T,np.matmul(Pscsc,M1))
    #Sscsc  = np.matmul(M1.T,np.matmul(Sscsc,M1))
    #N_aosc   = np.matmul(N_aosc,M1)
    #N_aopnao = np.matmul(N_aopnao,M1)
    #dummy  = np.matmul(dummy,M1)
    ##     
    #print(" ===== P^(ScSc) remap M1 basis ===== ","\n",Pscsc)
    #print(" ===== S^(ScSc) remap M1 basis ===== ","\n",Sscsc)
    #print(" ===== dummy^Sch remap M1 basis ===== ","\n",dummy)
    #print(" ===== N^(aoSch)  ===== ","\n",N_aosc)
    #print("Up to here Schmidt process; step_3; go to step4")
    ###############Recovering the natural character of N ##################
    #avPscsc = np.zeros((nbas,nbas), dtype=dtype)
    #avSscsc = np.zeros((nbas,nbas), dtype=dtype)
    #Pryry   = np.zeros((nbas,nbas), dtype=dtype)
    #Sryry   = np.zeros((nbas,nbas), dtype=dtype)

    #Wryd,Navscryd,avPscsc,avSscsc,ibasismap,nshell,lmax,ilmap,shellmap,center,func_am = nao_step_2(Pscsc,Sscsc)
    #w1,n1 = GenEig(Pscsc,Sscsc,nshell,lmax,ilmap,shellmap,center,func_am)
    #w2,n2 = GenEig(avPscsc,avSscsc,nshell,lmax,ilmap,shellmap,center,func_am)

    #print("w1",w1)
    #print("w2",w2)
    #T4 = np.matmul(n2,np.linalg.inv(n1))
    #print("T4.T*T4","\n",np.matmul(T4.T,T4))
    #print("Trial avPscsc?","\n",np.matmul(np.matmul(T4.T,Pscsc),T4))
    #print("avPscsc?","\n",avPscsc)

    #dummy = Wryd.copy()
    #T5 = Navscryd.copy()  
    #Pryry = np.matmul(T5.T,np.matmul(avPscsc,T5))
    #Sryry = np.matmul(T5.T,np.matmul(avSscsc,T5))
    #print(" ===== Pryry remap basis Nryd ===== ","\n",Pryry)
    #print(" ===== Sryry remap basis Nryd ===== ","\n",Sryry)
    #print(" ===== W^Ryd ===== ","\n",dummy)
    #print(" ===== N^avscRyd  ===== ","\n",Navscryd)
    #print(" ========================================================================= ")
    ###################End of step 3 #######################################
    ###################Let's beggin step 4 #################################
    #print(" ====================Beginning step 4 ==================================== ")
    #ne=np.sum(atNum)
    #print("Number of electrons",ne)  
    #check = 0.0
    #control  = abs(ne-check)
    ##control2 = abs(np.sum(atNum[0:natom])-np.sum(dummy[0:nM]))
    #thresh = 1.0e-6
    #print("Threshold for diff trace(P):",thresh)
    ##    
    ##while control > thresh:
    #print("Ne- control=",control)
    #aMask = [False for i in range(nbas)]          #Initializing the boolean array for having a map
    #aMask=createNMBmask(dummy,nbas,ibasismap,atNum)
    #print("aMask beginning step 4\n",aMask)
    ###
    #ibasisremap = np.zeros(nbas).astype(int)      #contains index of functions of an atom l-block
    #ibasisremap,nM,nR = partBasis(aMask,nM,nR,ibasisremap)
    #E  = np.identity(nbas)
    #N_NMB = np.zeros((nbas,nbas), dtype=dtype)
    #M2 = mapMatrix(E,ibasisremap)
    ##  
    #print("Occupations before M2","\n",dummy)
    #print("N_NMB before M2","\n",N_NMB)
    #dummy = np.matmul(dummy,M2.T)
    #N_NMB = np.matmul(N_aosc,T4)
    #N_NMB = np.matmul(N_NMB,T5)         #Transformation from ao to ryd 
    #N_NMB = np.matmul(N_NMB,M2.T)       #Mapping NMB|NRB
    #print("Occupations after M2","\n",dummy)
    #print("N_NMB after M2","\n",N_NMB)
    #print("nM",nM,"nR",nR)
    #print("M2\n",M2)
    #print("===== Pryry before M2 =====","\n",Pryry)
    #print("===== Sryry before M2 =====","\n",Sryry)
    #Pryry  = np.matmul(M2,np.matmul(Pryry,M2.T))
    #Sryry  = np.matmul(M2,np.matmul(Sryry,M2.T))
    #print("===== Pryry after M2 =====","\n",Pryry)
    #print("===== Sryry after M2 =====","\n",Sryry)
    ###
    #rydmask = [True for i in range(nbas)] #Initialize NRB set as all functions there
    #irydbasisremap=np.zeros(nbas).astype(int)
    ## 
    #wthresh = 1.0e-4  #Here, Ryd functions have lower occupations than wthresh. 
    #nhiryd  = 0
    #nloryd  = 0
    #for i in range(nbas):
    #  if dummy[i] < wthresh:
    #    rydmask[i] = False
    #print("===== Rydmask, false on index of loRyd functions =====","\n",rydmask)
    ##
    #irydbasisremap,nhiryd,nloryd = partBasis(rydmask,nhiryd,nloryd,irydbasisremap)
    #print("================= irydbasisremap indexing vector ===================","\n",irydbasisremap)
    #nhiryd = nbas-nloryd-nM
    #up=nM+nhiryd
    #print("nM:",nM)
    #print("nhiryd:",nhiryd)
    #print("up = nM+nhiryd:",up)
    #print("nloryd:",nloryd)
    #print("nbas:",nbas)
    #print("nM+nhiryd+nloryd:",nM+nhiryd+nloryd)
    #if nM+nhiryd+nloryd != nbas:
    #  print("Found a problem in mapping hi-Ryd and lo-Ryd space")
    #  print("Mismatch in functions balance","nbas:",nbas,"nM+nhiryd+nloryd:",nM+nhiryd+nloryd)
    #print("===================================================")
    ## 
    #E=np.identity(nbas)
    #M3=mapMatrix(E,irydbasisremap)
    #print("Occupations before M3",dummy)
    #print("N_NMB before M3","\n",N_NMB)
    #dummy = np.matmul(dummy,M3.T)
    #N_NMB = np.matmul(N_NMB,M3.T)
    #print("Occupations after M3","\n",dummy)
    #print("N_NMB after M3","\n",N_NMB)
    #print("M3\n",M3)
    #Pryry = np.matmul(np.matmul(M3,Pryry),M3.T)
    #Sryry = np.matmul(np.matmul(M3,Sryry),M3.T)
    #print("===== Pryry after M3 =====","\n",Pryry)
    #print("===== Sryry after M3 =====","\n",Sryry)
    #print("===================================================")
    ##
    #print("===== Ow Weighted Symm Orth for NMB-NMB space =====")
    #Ow = np.identity(nbas)
    #trial = np.zeros((nM,nM))
    #for i in range(nM):
    #  for j in range(nM):
    #    trial[i,j] = S_ij(N_NMB[:,i],N_NMB[:,j],S)
    #print("===== This is a trial overlap matrix in NMB-NMB space =====","\n",trial)
    #print("===== Sryry[0:nM,0:nM] and trial are the same matrix ???? =====","\n",Sryry[0:nM,0:nM])
    #Ow[0:nM,0:nM] = symmOrthMat(Sryry[0:nM,0:nM],dummy[0:nM])
    #print("===== O_w^(NMB) =====","\n",Ow)
    #T6 = Ow.copy()
    #print("===== T6 NMB =====","\n",T6)
    #Pmm = np.zeros((nbas,nbas), dtype=dtype)
    #Smm = np.zeros((nbas,nbas), dtype=dtype)
    #Pmm = np.matmul(T6.T,np.matmul(Pryry,T6))
    #Smm = np.matmul(T6.T,np.matmul(Sryry,T6))
    #print("===== Pmm after T6 NMB-NMB =====","\n",Pmm)
    #print("===== Smm after T6 NMB-NMB =====","\n",Smm)
    #print("===================================================")
    #print("===== Ow Weighted Symm Orth for hiRyd-hiRyd space =")
    #Ow = np.identity(nbas)
    #trial = np.zeros((nhiryd,nhiryd))
    #for i in range(nM,up):
    #  for j in range(nM,up):
    #    trial[i-nM,j-nM] = S_ij(N_NMB[:,i],N_NMB[:,j],S)
    #print("===== This is a trial overlap matrix in hiRyd-hiRyd space =====","\n",trial)
    #print("===== Sryry[nM:up,nM:up] and trial are the same matrix ???? =====","\n",Sryry[nM:up,nM:up])
    #if nhiryd > 0:
    #  Ow[nM:up,nM:up] = symmOrthMat(Sryry[nM:up,nM:up],dummy[nM:up])
    #print("===== O_w^(hiRyd) =====","\n",Ow)
    #T7 = Ow.copy()
    #print("===== T7 =====","\n",T7.T)
    #Phrhr = np.zeros((nbas,nbas), dtype=dtype)
    #Shrhr = np.zeros((nbas,nbas), dtype=dtype)
    #Phrhr=np.matmul(T7.T,np.matmul(Pmm,T7))
    #Shrhr=np.matmul(T7.T,np.matmul(Smm,T7))
    #print("===== Phrhr after T7 hi-Ryd =====","\n",Phrhr)
    #print("===== Shrhr after T7 hi-Ryd =====","\n",Shrhr)
    #print("====================================================================")
    ################### Schmidt orthogonalize the loRyd basis ################
    #Psclory = np.zeros((nbas,nbas), dtype=dtype)
    #Ssclory = np.zeros((nbas,nbas), dtype=dtype)
    #N_slor = N_NMB.copy()
    #N_slor = np.matmul(N_slor,T6)
    #N_slor = np.matmul(N_slor,T7)
    #if nloryd > 0:
    #  print(" ===== Starting the Schmidt-Orth of loRyd space ===== ")
    #  Os = np.identity(nbas)
    #  print("===== set MB+hiR =====","\n",N_slor[:,0:up])
    #  print("===== set loRyd  =====","\n",N_slor[:,up:nbas])
    #  print("Up_idx:",up)
    #  print("nloryd:",nloryd)
    #  print("nbas",nbas)
    #  fullS = GenOverlap(N_slor,S)
    #  print(" ===== Check S before lo-Schmidt ===== ","\n",fullS)
    #  N_pnao = N_slor.copy()
    #  for j in range(nbas):
    #    for i in range(up):
    #      if i<j:
    #        proj = S_ij(N_pnao[:,j],N_slor[:,i],S)
    #        norm = S_ij(N_slor[:,i],N_slor[:,i],S)
    #        if norm < 1.0e-14:
    #          print("Linear dependency found in i,j pair: at lo-Ryd",i,j)
    #          exit()
    #        Os[i,j] = proj/norm
    #        N_slor[:,j] = N_slor[:,j]-N_slor[:,i]*Os[i,j]
    #  for i in range(up,nbas):
    #    norm_c = S_ij(N_slor[:,i],N_slor[:,i],S)
    #    norm_c = np.sqrt(1.0/norm_c)
    #    N_slor[:,i] *= norm_c
    #  print(" ===== Orthogonal N^(Sc) before copy loRyd ===== ","\n",N_slor)
    #  N_slor[:,0:up] = N_pnao[:,0:up].copy()
    #  fullS=GenOverlap(N_slor,S)
    #  print(" ===== Check S after lo-Schmidt ===== ","\n",fullS)
    #  print(" ===== Orthogonal N^(Sc) loRyd) ===== ","\n",N_slor)
    #  T8 = np.matmul(N_slor,np.linalg.inv(N_pnao))
    #  print("N_slor=Os*N_pnao???","\n",np.matmul(T8,N_pnao))
    #  Psclory = np.matmul(np.matmul(N_slor.T,P),N_slor)
    #  Ssclory = np.matmul(np.matmul(N_slor.T,S),N_slor)
    #  print("==== Ppnao after schmidt lo-Ry =====","\n",Psclory)
    #  print("==== Spnao after schmidt lo-Ry =====","\n",Ssclory)
    #print("=========End of loRyd Schmidt orthogonalization=======================")
    #print("======================================================================")
    ##############Lowdin orthogonalization of loRyd space############################
    #print("=============== Ow^(loRyd) reduction to Symm-Orth S^(-1/2) ===========")
    #if nloryd > 0:
    #  Ow = np.identity(nbas)
    #  trial = np.zeros((nloryd,nloryd))
    #  for i in range(up,nbas):
    #    for j in range(up,nbas):
    #      trial[i-up,j-up] = S_ij(N_slor[:,i],N_slor[:,j],S)
    #  print("===== This is a trial overlap matrix in loRyd-loRyd space =====","\n",trial)
    #  print("===== Spnao[up:nbas,up:nbas] and trial are same ???? =====","\n",Ssclory[up:nbas,up:nbas])
    ##### Here the weights of the loRyd space are set to 1.0 ##############################
    ##### Ow^(loRyd) reduces to S^(-1/2) of symmetric orthogonalization of loRyd space ####
    #  dummy[up:nbas] = 1.000000
    #  print("nloryd:",nloryd)
    #  Ow[up:nbas,up:nbas] = symmOrthMat(Ssclory[up:nbas,up:nbas],dummy[up:nbas])
    #  print("===== O_w^(hiRyd) =====","\n",Ow)
    #  T9 = Ow.copy()
    #  print("===== T9 =====","\n",T9.T)
    #  Pww=np.matmul(T9,np.matmul(Psclory,T9.T))
    #  Sww=np.matmul(T9,np.matmul(Ssclory,T9.T))
    #  N_slor  = np.matmul(N_slor,T9.T)
    #  print("===== Ppnao  after T9 loRyd =====","\n",Pww)
    #  print("===== Spnao  after T9 loRyd =====","\n",Sww)
    #  print("===== N_slor after T9 loRyd =====","\n",N_slor)
    #  print("=============================================")
    #print("===== Ow transformations (NMB,hiRyd,loRyd) completed =====")
    #####################################################################################
    ##check = 0.0
    ##for i in range(nbas):
    ##  check += Ppnao[i,i] 
    ##control = abs(ne-check)
    ###control2 = abs(np.sum(atNum[0:natom])-np.sum(W2[:]))
    ##print("Check trace of density after weighted orthogonalization, control:",control)
    ###########Revert mappings############################################################
    ##print("===== Reverse mappings T5 and T4 to recover basis ordering =====")
    ##Pww = np.matmul(np.matmul(M3.T,Pww),M3)
    ##Sww = np.matmul(np.matmul(M3.T,Sww),M3)
    ##N_slor = np.matmul(N_slor,M3)
    ##dummy = np.matmul(dummy,M3)
    #####
    ##Pww = np.matmul(np.matmul(M2.T,Pww),M2)
    ##Sww = np.matmul(np.matmul(M2.T,Sww),M2)
    ##N_aoaw = np.matmul(N_slor,M2)
    ##dummy = np.matmul(dummy,M2)
    ##print("===== Pww after Ow procedure =====","\n",Pww)
    ##print("===== Sww after Ow procedure =====","\n",Sww)
    ##print("===== Occupations after Ow procedure =====","\n",dummy)
    ##print("===== N_aoaw after Ow procedure =====","\n",N_aoaw)
    ##print("===== Mappings finished =====")
    ##### Recovering Natural character of the basis with the eigenfunctions of \tilde(P) ####
    #####
    ##Ww,Navwnao,avPww,avSww,ibasismap,nshell,lmax,ilmap,shellmap,center,func_am = nao_step_2(Pww,Sww)
    ##dummy = Ww.copy()

    ##w1,n1 = GenEig(Pww,Sww,nshell,lmax,ilmap,shellmap,center,func_am)
    ##w2,n2 = GenEig(avPww,avSww,nshell,lmax,ilmap,shellmap,center,func_am)
    ##
    ##T10 = np.matmul(n2,np.linalg.inv(n1))   #Naw,avaw
    ##print("T10.T*T10","\n",np.matmul(T10.T,T10))
    ##
    ##print("T10.T*Pww*T10","\n",np.matmul(np.matmul(T10.T,Pww),T10))
    ##print("avPww","\n",avPww)

    ##T11 = Navwnao.copy()
    ##Pnaonao = np.matmul(Navwnao.T,np.matmul(avPww,Navwnao))
    ##Snaonao = np.matmul(Navwnao.T,np.matmul(avSww,Navwnao))
    ###  
    ##print("Ppnao after all transformations\n",Ppnao)
    ##print("Spnao after all transformations\n",Spnao)
    ###
    ##T = np.identity(nbas)
    ##T = np.matmul(T,T1)
    ##T = np.matmul(T,T2)
    ##T = np.matmul(T3,T)
    ##T = np.matmul(T,T4)
    ##T = np.matmul(T,T5)
    ##T = np.matmul(T,T6)
    ##T = np.matmul(T,T7)
    ##T = np.matmul(T8,T)
    ##T = np.matmul(T,T9)
    ##T = np.matmul(T,T10)
    ###
    ##print("T_(AO)^(NAO)\n",T)
    ###
    ##PNAO  =  np.zeros((nbas,nbas), dtype=dtype)
    ##SNAO  =  np.zeros((nbas,nbas), dtype=dtype)
    ##PNAO  =  np.matmul(T.T,np.matmul(P,T))
    ##SNAO  =  np.matmul(T.T,np.matmul(S,T))
    ##print("dummy","\n",dummy)
    ##print("Test P^(NAO)=T.TP^(AO)T \n",PNAO)
    ##print("Test S^(NAO)=T.TS^(AO)T \n",SNAO)
    ####exit()

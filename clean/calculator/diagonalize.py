import numpy as np 

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

matrix=np.array([[1.1552,  1.3661,  1.1552,  1.3661],
                [1.3661,  1.6156,  1.3661,  1.6156],
                [1.1552,  1.3661,  1.1552,  1.3661],
                [1.3661,  1.6156,  1.3661,  1.6156]])

block=np.array([[1.1552,  1.3661],
                [1.3661,  1.6156]])


#result=2.0*np.einsum("i,j->ij",matrix[:,0],matrix[:,0])

x,y = np.linalg.eig(block)
print("eigenvalues","\n",x)
print("eigenvectors","\n",y)

# nao

#Jun 2024 - j.a.g.cruz@kjemi.uio.no/albertogc1993@gmail.com

The file 3-21G.47 now is reproduced missing the printing format and the optional matrices  
FOCK,KINETIC,NUCLEAR, and DIPOLE matrices                                              

The main code is on the file 

hf.py

Set up an environment verifying psi4 is installed and numpy.

```
  $ conda activate
  $ pip install psi4
  $ pip install numpy
```

Then, you can run the script simply doing 
```
  $ psi4 hf.py
```

The script loads an specified molecule in cartesian format,
computes a RHF optimization and then use the resulting information

The basis are given in the command inside the script, I have tested 
to print the basis for a FILE47. I'm using the following to compare
the only stable for the moment is 3-21G for H2

```
#  set basis 6-311G(d,p)
#  set basis 6-311G(2df,2pd)
#  set basis 3-21G
```

The corresponding file47 created with a gaussian calculation are in the gaussianF47 directory
my goal rigth now is to reproduce this files using the python script as if they were called with
gennbo.

```
 6-311G-d-p.47
 6-311G-2df-2pd.47  
 3-21G.47  
```

TODO
 -- Reproduce the nbo matrices produced after calling gennbo but standalone. Actual status can produce the preNAOs associated to the overlap and density matrices.
 -- Reproduce the transformation matrices for AOs (done), NAOs (in progress), MOs, NBOs, NLMOs.
 -- Unplug the orbital basis from psi4 and interface to HSP. We need to define a molecule object
 -- Collect operators to calculate the shielding tensor. Need to be read from ESS
 -- Reproduce a non-relativistic natural shielding analysis. Methane or water is my actual goal. Optimistic achieve a [PtCl4]2- calculation in the next month
 -- Clean the code

- TODO list

'''
- Add a first mwe of hsp import 
- Add a first orca calc RHF/STO-3G collect P,S,C
- Interface calls to the hf.py script functions
'''

cat << EOF > $1.slm
#!/bin/bash

## Setting up account, job name, number of nodes, cores/node, time, and memory

#SBATCH --account=nn4654k
#SBATCH --job-name=$1
#SBATCH --nodes=$2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=/cluster/home/$USER/logfiles/$1.logfile
#SBATCH --signal=B:USR1@120
###################################################################

## Modules for Gaussian and NBO7
module purge
module load Gaussian/g16_C.01

## Scratch folder
if [[ -d "/cluster/work/users/$USER/$1" ]]
then
    rm -rf /cluster/work/users/$USER/$1
fi

mkdir /cluster/work/users/$USER/$1

# Copying input files (also chk, if it exists) to scratch

cd \$SUBMITDIR
cp $1.* /cluster/work/users/$USER/$1/
cd /cluster/work/users/$USER/$1

# Running the calc with the parallel ib binary from the scratch folder

g16.ib  < $1.gjf > \$SUBMITDIR/$1.log

# Recovering the relevant files and cleaning up

formchk $1.chk
cp $1.47 \$SUBMITDIR

cd \$SUBMITDIR
# rm -rf /cluster/work/users/$USER/$1

EOF

sbatch $1.slm


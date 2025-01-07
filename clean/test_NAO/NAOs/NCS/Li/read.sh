#!/bin/bash
#Purpose is to read the optimized geometry from a gaussian .log file and write an .xyz molden file
#Execution ./read.sh file.log 4
#               gaussian_file number_of_atoms

file=$1
nratoms=$(grep NAtoms $file -m1 | awk '{print $2}')
symbols=$(grep "0 1" ${file%.log}.gjf -A $nratoms | tail -n $nratoms | awk '{print $1}')
#echo $symbols
linenr=$(awk '/Optimized/ {print NR}' $file)
tail -n +${linenr} $file > tmp
secondlinenr=$(awk '/Input orientation/ {print NR}' tmp)
tail -n +${secondlinenr} tmp > tmp1
awk -v var=$nratoms 'NR>5 && FNR<=(5+var)' tmp1 > tmp
awk -F ' ' '{print $4, $5, $6}' tmp > opt_coords.txt
#####Parse opt geometry to file.xyz
echo $nratoms > ${file%.log}.xyz
echo "" >> ${file%.log}.xyz
printf "%s\n" $symbols > opt_sym.txt 
paste opt_sym.txt opt_coords.txt >> ${file%.log}.xyz
rm tmp tmp1 opt_sym.txt opt_coords.txt 

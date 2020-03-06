#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### This script acts like a user on a remote server allocated by SLURM
# Only thing to change is the master directory to save in and virtual environment name
#for pytorch: the script will then
# add a seed directory to this and run the python script train_student.py

# load modules and environments
echo 'entering inner script'
echo 'loading module'
module load anaconda3
echo 'activating virtual env'
source activate torch-env

SDIR="/scratch/gpfs/ruairidh/HKD/seed_${seed}"
echo "saving in: ${SDIR}"

# if directory doesn't exist, make it
if [ -d "${SDIR}" ]
    then
    echo 'SDIR present'
    else
    echo 'making new sdir'
    mkdir ${SDIR}
fi

echo 'entering python script'

# call main python script, with arguments passed in through run_model_HKD_outer.sh
python train_student.py --epochs ${epochs} --teacher ${teacher} --student ${student} \
--learning-rate ${lr} --temperature ${t} --lambda_ ${l} --gamma_ ${g} --iter ${run} \
--dataset cifar10 --cuda 1 --manual_seed ${seed} --trial_id ${jobname} \
--master_outdir ${SDIR} --master_architecture ${arch}

echo 'inner done'

#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G 
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

echo 'entering python script'
echo "loading and saving in ${OUTDIR}"
echo "for student ${STUDENT}"
# call main python script, with arguments passed in through run_model_HKD_outer.sh
python eval_student.py --master_outdir ${OUTDIR} --student_name ${STUDENT}

echo 'inner done'

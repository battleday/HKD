#!/usr/bin/env bash
#SBATCH -p all
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

echo 'entering inner script'
echo 'loading module'
module load anaconda3
echo 'activating virtual env'
source activate torch-env

#should print arguments

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    echo "Key: ${KEY}, Value: ${VALUE}"
done

SDIR="/scratch/gpfs/ruairidh/HKD/seed_${seed}"
echo "saving in: ${SDIR}"

if [ -d "${SDIR}" ]
    then
    echo 'SDIR present'
    else
    echo 'making new sdir'
    mkdir ${SDIR}
fi

echo 'entering python script'

python train_student.py --epochs 200 --teacher ${teacher} --student ${student} \
--learning-rate ${lr} --temperature ${t} --lambda_ ${l} --gamma ${g} --iter ${run} \
--dataset cifar10 --cuda 1 --manual_seed ${seed} --trial_id ${jobname} \
--master_outdir ${SDIR}
/

echo 'inner done'

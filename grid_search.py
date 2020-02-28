import os
learning_rates= [0.1]
temperatures = [5]
lambdas= [0.4]
gammas = [1]
seeds=[0]
runs=[0]
for seed in seeds:
    for run in runs:
        for lr in learning_rates:
            for t in temperatures:
    	        for l in lambdas:
                    for g in gammas:
                        trial_id="seed_${seed}_run_${run}_lr_${lr}_t_${t}_lam_${l}_gam_${g}"
                        os.system("sbatch -N 1 -n 1 -t 5 -o /scratch/gpfs/ruairidh/HKD/human_resnet110_id_${trial_id}.out --gres=gpu:1 --wrap 'python -u train_student.py --epochs 200 --teacher human --student resnet110 --learning-rate ${lr} --temperature ${t} --lambda_ ${l} --gamma_ ${g} --cuda 1 --dataset cifar10  --iter ${run} --manual_seed ${seed} --trial_id ${run} --master_outdir /scratch/gpfs/ruairidh/HKD'")

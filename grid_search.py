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
                        os.system("sbatch -N 1 -n 1 -t 3 -o /tigress/smondal/softlabels_teacheronly_resnet110_number_%s.out --gres=gpu:1 --wrap 'python -u train.py --epochs 200 --teacher resnet110  --learning-rate 0.1 --cuda 1 --dataset cifar10  --trial_id %s'"%(run_no,run_no))

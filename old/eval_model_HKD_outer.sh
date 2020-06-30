#!/usr/bin/env bash

### Master grid search script
# Feel free to generate sbatch command in a different way: I am just used to doing bash 

echo 'outer tuning script'

# parameters to search over, given as strings
time=70
t_ts='0.5 1.0 5.0 10.0 20.0' # 5 10 15'
t_hs='1.0' # 1.0 5.0 10.0 20.0' # 0.5 1.0 5.0 10.0 20.0'
ls='0.0 0.2 0.4 0.6' # 0.2 0.4'
runs='0 1 2 3 4 5 6 7 8 9' # 1 2 3 4 5 6 7 8 9'
distils='KD CE'
teachers='baseline_shake26_0.2 baseline_shake26_0.4 baseline_shake26_0.6 human_shake26_0.2 human_shake26_0.4 human_shake26_0.6'

# grid search
for teacher in $teachers
    do
    echo "teacher: ${teacher}"
    for t_t in $t_ts
        do
        for t_h in $t_hs
            do
            for l in $ls
                do
                for distil in $distils
                    do
                    echo "distil: ${distil}"
            
                    for run in $runs
                        do
                        echo "run: ${run}"
            
                         echo "${t}_${l}_${g}"
                         # unique job name based on grid search parameters. Used as trial ID and in SLURM summary
                         jobname="gen_teacher_${teacher}__distil_${distil}_t_t_${t_t}_t_h_${t_h}_l_${l}_run_${run}"
                         STUDENT="student_resnet8_distil_fn_${distil}_temperature_h_${t_h}_temperature_t_${t_t}_lambda_${l}_gamma_0.0_iter_${run}_best.pth.tar"
                         OUTDIR="/scratch/gpfs/ruairidh/HKD/seed_0/${teacher}/resnet8"
                         # a log file for each job will be dropped in main directory
                         logfile="${jobname}"
                         echo "running job ${jobname}"

                         # run sbatch job, givin sbatch parameters and exporting python parameters to run_model_HKD.sh
                         sbatch --time=${time} --job-name=${jobname} --output=${logfile}  --export=STUDENT=$STUDENT,OUTDIR=$OUTDIR,run=$run,jobname=$jobname eval_model_HKD.sh
                         done
                    done
                done
            done
        done
    done

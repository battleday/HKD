#!/usr/bin/env bash

### Master grid search script
# Feel free to generate sbatch command in a different way: I am just used to doing bash 

echo 'outer tuning script'

# parameters to search over, given as strings
time=300
lrs='0.1'
ts='1 5 10 15'
ls='0.0 0.2 0.4'
gammas='1'
seeds='0'
runs='0 1 2 3 4'
distils='KD CE'
teachers='human'
students='shake26'
arch='shake_shake'
epochs='200'

# grid search
for teacher in $teachers
    do
    echo "teacher: ${teacher}"
    for student in $students
        do
        echo "student: ${student}"
        for seed in $seeds
            do
            echo "seed: ${seed}"
        
            for lr in $lrs
                do
                echo "lr: ${lr}"

                for t in $ts
                    do
                    for l in $ls
                        do
                        for distil in $distils
                            do
                             echo "distil: ${distil}"
            
                             for run in $runs
                                 do
                                 echo "run: ${run}"
            
                                for g in $gammas
                                    do
                                    echo "${t}_${l}_${g}"
                                    # unique job name based on grid search parameters. Used as trial ID and in SLURM summary
                                    jobname="seed_${seed}_teacher_${teacher}_student_${student}_distil_${distil}_lr_${lr}_t_${t}_l_${l}_g_${g}_run_${run}"

                                    # a log file for each job will be dropped in main directory
                                    logfile="${jobname}"
                                    echo "running job ${jobname}"

                                    # run sbatch job, givin sbatch parameters and exporting python parameters to run_model_HKD.sh
                                    sbatch --time=${time} --job-name=${jobname} --output=${logfile}  --export=epochs=$epochs,arch=$arch,seed=$seed,teacher=$teacher,student=$student,distil=$distil,lr=$lr,t=$t,l=$l,g=$g,run=$run,jobname=$jobname run_model_HKD.sh
                    
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done

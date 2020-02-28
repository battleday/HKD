#!/usr/bin/env bash
echo 'outer tuning script'
time=200
lrs='0.1'
ts='0.5 1 5 10 15 20 25'
ls='0.0 0.2 0.4 0.6 0.8 1.0'
gammas='1'
seeds='0'
runs='0'
distils='KD CE'
teachers='human'
students='resnet110'

for teacher in $teachers
    do
    echo "teacher: ${teacher}"
    for student in $students
        do
        echo "student: ${student}"
        for seed in $seeds
            do
            echo "seed: ${seed}"
        
            for distil in $distils
                do
                echo "distil: ${distil}"
                for lr in $lrs
                    do
                    echo "lr: ${lr}"
            
                    for run in $runs
                        do
                        echo "run: ${run}"
            
                        for t in $ts
                            do
                            for l in $ls
                                do
                                for g in $gammas
                                    do
                                    echo "${t}_${l}_${g}"
                                    jobname="seed_${seed}_teacher_${teacher}_student_${student}_distil_${distil}_lr_${lr}_t_${t}_l_${l}_g_${g}_run_${run}"
                                    logfile="${jobname}"
                                    echo "running job ${jobname}"
                                    sbatch --time=${time} --job-name=${jobname} --output=${logfile}  --export=seed=$seed,teacher=$teacher,student=$student,distil=$distil,lr=$lr,t=$t,l=$l,g=$g,run=$run,jobname=$jobname run_model_HKD.sh
                    
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done

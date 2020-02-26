import os
learning_rates= [0.2,0.18,0.15,0.13,0.11,0.105,0.103,0.101,0.1,0.099,0.096,0.093]
temperatures = [5,10,15,20,25]
lambdas= [0.4,0.6,0.8]
gammas = [1]
# for run_no in range(1,4):
# 	for t in temperatures:
# 		for l in lambdas:
# 			for g in gammas:
	
# 				os.system("sbatch -N 1 -n 1 -t 300 -o /tigress/smondal/softlabels_teacherhumandiv_r14TA_bestteacher_resnet39th_tr14_sr8_temp_%s_lambda_%s_gamma_%s_number_%s.out --gres=gpu:1 --wrap 'python -u train.py --epochs 200 --teacher resnet14 --teacher-checkpoint /tigress/smondal/resnet14_3_train10_test50_customdata_softlabels_teacherhumandiv_besteacher_resnet39th_lambda0.6_temp15_best.pth.tar --student resnet8 --cuda 1 --dataset cifar10  --learning-rate 0.096 --trial_id %s --temperature %s --lambda_ %s --gamma_ %s'"%(t,l,g,run_no,run_no,t,l,g))

# 				# os.system("sbatch -N 1 -n 1 -t 300 -o 2ndtime_softabels_teacherbasecce_humandivergence_thrld3_t10_s2_temp_%s_lambda_%s_gamma_%s_number_%s.out --gres=gpu:1 --wrap 'python -u train.py --epochs 200 --teacher plane10 --teacher-checkpoint ./plane10_0_train10_test50_customdata_best.pth.tar --student plane2  --cuda 1 --dataset cifar10  --learning-rate 0.01 --trial_id %s --temperature %s --lambda_ %s --gamma_ %s'"%(t,l,g,run_no,run_no,t,l,g))


# for teacher_no in range(1,101):

	
# 	os.system("sbatch -N 1 -n 1 -t 300 -o /tigress/smondal/hardlabels_new_basekd_teacher_resnet110_%sth_tr110_sr14_temp_25_lambda_0.4_number_%s.out --gres=gpu:1 --wrap 'python -u train.py --epochs 200 --teacher resnet110 --teacher-checkpoint /tigress/smondal/resnet110_%s_train10_test50_customdata_hardlabels_new_teacheronly_resnet110_best.pth.tar --student resnet14 --cuda 1 --dataset cifar10  --learning-rate 0.096 --trial_id %s --temperature 25 --lambda_ 0.4'"%(teacher_no,teacher_no,teacher_no,teacher_no))


for run_no in range(1,51):

	
	os.system("sbatch -N 1 -n 1 -t 300 -o /tigress/smondal/softlabels_teacheronly_resnet110_number_%s.out --gres=gpu:1 --wrap 'python -u train.py --epochs 200 --teacher resnet110  --learning-rate 0.1 --cuda 1 --dataset cifar10  --trial_id %s'"%(run_no,run_no))

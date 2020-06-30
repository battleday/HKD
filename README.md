# HKD
Human Knowledge Distillation
---

## TL;DR
Run the following command to launch a training gridsearch:
`module load anaconda3`
`conda activate torch-env` 
`python slurm_jobber.py -c train_config.json`

Run the following command to launch an analysis gridsearch:
`module load anaconda3`
`conda activate torch-env` 
`python slurm_jobber.py -c analyze_config.json`

Run the following command to launch an evaluate gridsearch:
`module load anaconda3`
`conda activate torch-env` 
`python slurm_jobber.py -c evaluate_config.json`

Run the following commands to allocate yourself to a gpu and get ready to test .py files directly:

`salloc -t 00:05:00 --gres=gpu:1`
`module load anaconda3`
`conda activate torch-env` 
              
N.B. A package file is included for you to create torch-env from; online .txt file in repo.
---

## Introduction
This repo implements Human Knowledge Distillation, an extension of baseline Knowledge Distillation (BKD) and Teacher-Assistant Knowledge Distillation (TAKD) that integrates human uncertainty into the training of student networks. 

KD aims for model compression in image classification networks by using an extra source of supervision in addition to the one-hot ground truth category vectors provided during supervision. This extra supervision comes in the form of soft labels---originally the softened softmax outputs from a trained teacher network with more parameters. KD has two sets of goals:

1. Classification Goals (student network)
* High accuracy on validation set;
* Good out-of-training-sample generalization;
* Robustness to adversarial attacks.

2. Compression Goals (student network)
* Fewer parameters;
* Faster convergence.


The student model is still trained using the cross-entropy loss between its outputs and the ground-truth labels (L1). In addition, the extra supervision is added in the form of a second loss function (L2). The final loss function is a convex combination of these two (L):

 L = (1-&lambda;)L<sub>1</sub> + &lambda;L<sub>2</sub>.
 
 The first loss term is simply the cross-entropy loss between model outputs and the one-hot ground-truth labels:
 
 L<sub>1</sub> = -&Sigma;<sub>i</sub> p<sub>i</sub> log(q<sub>i</sub>),
 
 where p<sub>i</sub> is the neural network output probability for class i, and q<sub>i</sub> is the hard label.
 
 The second loss term, L2, is itself composed of a convex combination of a compression signal from the teacher, L2t, and a data signal from the human labels, L2h:
 
 L<sub>2</sub> = (1-&gamma;)L<sub>2t</sub> + &gamma;L<sub>2h</sub>.
 
 Each of these terms is as follows:
 
 L<sub>2t</sub> = T<sub>t</sub> * T<sub>t</sub> * Loss Function(log(p)<sub>i</sub>/T<sub>t</sub>, log(l)<sub>i</sub>/T<sub>t</sub>),
 
 L<sub>2h</sub> = T<sub>h</sub> * T<sub>h</sub> * Loss Function(log(p)<sub>i</sub>/T<sub>h</sub>, log(h)<sub>i</sub>/T<sub>h</sub>),
 
 where l<sub>i</sub> are the teacher output probabilities and h<sub>i</sub> are the human labels, and T<sub>t</sub> and T<sub>h</sub> are the compression temperature and data temperature, respectively. The human labels and teacher outputs are all be converted into probabilities using a softmax function prior to comparison in the loss function. We use either cross-entropy and relative entropy (KL divergence) as our loss functions.
 
The remaining parameters are related to the optimization process and data storage and loading.
 
 
---
 ## Files
 
 In this repo, the following files are used:
 1. `train_config.json`
 
 Contains gridsearch params for training; only file that needs editing (must change / add paths);
 2. `train_student.py` 
 
The main runfile for training a student. Takes a number of command-line arguments, all specified in parse_args function. From this, it creates all directory structure, initializes the seed for torch, loads student and teacher models and creates the  final save path and logging files.

3. `train_manager.py`
The training algorithm, instatiated as a class. When called, the `train_manager.py` `TrainManager` `train` method will train the student and keep track of the training and validationi losses, and model state at the highest-scoring validation accuracy. After training, it will call the `save` method to dump the losses and best model state to 'outfile'.

4. `find_best_teacher.py`
 
Helper module to load teacher.

5. `analyze_config.json`
Contains gridsearch params to analyze all models in a particular folder. This will give their training and validation accuracies and losses on cifar10 and cifar10h. Currently does not analyze for cinic and imagenet far.

6. `csv_dump_model_scores.py`
 
This takes a directory of trained models, scrapes the model results, averages them across runs, and outputs a csv file in the same directory.
 
7. `evaluate_config.json`
Contains gridsearch params to evaluate all models in a particular folder on cinic and imagenet far, and dump generalization results as npy file in the samed folder.

8. `eval_student.py`
Runfile for evaluation. Currently a bug where after successfully evaluating the first model in the folder, halts on test_loader error for second model.

9. `eval_manager.py`
Algorithm for evaluation. Working.

10. Various dataloaders
 
 The data loader module, originally from here: https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation/blob/master/data_loader.py, and adapted in separate files to load cinic and imagenet-far
 
12. `slurm_jobber.py`
Python script to generate sbatch command. Add relevant configuration file with -c flag (see examples at top; https://github.com/jcpeterson/easy_slurm).

13. `utils.py`, `return_args.py`
Auxillary files for slurm_jobber.

14. `model_factory.py`
 
Originally from here: https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation/blob/master/model_factory.py.
 
15. `torch-env.txt`
Dump of anaconda environment I use to run the above.

---
## Directories

1. Data

Contains data files kept locally (test50k_labels.npy, cifar10h-probs.npy,		train10k_images.npy, test50k_images.npy,		train10k_labels.npy, etc). Email to receive these.

2. Architectures

Contains architectures from TAKD repo, and provides the backend to `model_factory.py`.

3. Old
Old scripts.

4. Results
Results from the pilot



---

This repo is based on the following two pytorch repos for CIFAR10 image classification models:

Teacher-Assistant Knowledge Distillation
https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation

Pytorch image classification
https://github.com/hysts/pytorch_image_classification

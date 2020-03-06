# HKD
Human Knowledge Distillation
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
 
 L<sub>2</sub> = (1-&gamma;)L<sub>2S</sub> + &gamma;L<sub>2D</sub>.
 
 Each of these terms is as follows:
 
 L<sub>2t</sub> = T<sub>t</sub> * T<sub>t</sub> * Loss Function(p<sub>i</sub>/T<sub>t</sub>, l<sub>i</sub>/T<sub>t</sub>),
 
 L<sub>2h</sub> = T<sub>h</sub> * T<sub>h</sub> * Loss Function(p<sub>i</sub>/T<sub>h</sub>, h<sub>i</sub>/T<sub>h</sub>),
 
 where l<sub>i</sub> are the teacher output probabilities and h<sub>i</sub> are the human labels, and T<sub>t</sub> and T<sub>h</sub> are the compression temperature and data temperature, respectively. Typically these inputs will all be converted into probabilities using a softmax function prior to comparison in the loss function. We use either cross-entropy and relative entropy (KL divergence) as our loss functions.
 
The remaining parameters are related to the optimization process and data storage and loading.
 
 
---
 ## Files
 
 In this repo, the following files are used:
 1. train_manager. 
 
 This module contains all the training code. It takes a student and teacher outputs, learning and optimization parameters, and directory pointers, and then trains the student and saves accordingly. The main object class is `TrainManager`, which takes as input a `student` pytorch model, a numpy array of 10000 * 10 probabilities derived from a teacher network for all images in the validation subset of CIFAR10, `teacherProbs`, `train_loader` and `test_loader`, derived from the data loader, and `train_config`. This last file should contain the following keys:
 
 'epochs'---number of epochs to train for;
 'learning_rate'---of student;
 'momentum';
 'weight_decay';
 'device';
 'trial_id'---currently used to give unique string specifier;
 'batch_size';
 'distil_fn';
 'lambda_';
 'gamma_';
 'temperature_h';
 'temperature_t';
 'outfile'.
 
 When called, the `train_manager.py` `TrainManager` `train` method will train the student and keep track of the training and validationi losses, and model state at the highest-scoring validation accuracy. After training, it will call the `save` method to dump the losses and best model state to 'outfile'.
                                
 2. train_student. The main runfile for training a student. Takes a number of command-line arguments, all specified at top of file. From this, it creates all directory structure, initializes the seed for torch, loads student model and teacherProbs, and creates the final save path and logging files.
---
This repo is based on the following two pytorch repos for CIFAR10 image classification models:

Teacher-Assistant Knowledge Distillation
https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation

Pytorch image classification
https://github.com/hysts/pytorch_image_classification

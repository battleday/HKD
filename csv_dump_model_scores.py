import os, sys
import numpy as np
import pickle
from find_best_teacher_test import *
script, master_dir = sys.argv


"""Helper script, to be able to call functions in find_best_teacher module using simple
command-line arguments"""
print('dumping in dir: {}'.format(master_dir))

with open('{}/model_summary.p'.format(master_dir), 'rb') as pfile:
    averages = pickle.load(pfile)

csv_dump(averages, master_dir)

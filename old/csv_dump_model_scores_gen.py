import os, sys
import numpy as np
import pickle
from find_best_teacher_test import *
script, master_dir = sys.argv


"""Helper script, to be able to call functions in find_best_teacher module using simple
command-line arguments"""
print('scanning in dir: {}'.format(master_dir))
averages = compile_model_summary_gen(master_dir)

with open('{}/model_summary_gen.p'.format(master_dir), 'wb') as pfile:
    pickle.dump(averages, pfile)

"""Helper script, to be able to call functions in find_best_teacher module using simple
command-line arguments"""
print('dumping in dir: {}'.format(master_dir))

with open('{}/model_summary_gen.p'.format(master_dir), 'rb') as pfile:
    averages = pickle.load(pfile)

csv_dump_gen(averages, master_dir)

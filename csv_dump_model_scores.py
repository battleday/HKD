import os, sys
import numpy as np
import pickle
import argparse
from find_best_teacher import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Basic evaluation of model results')
    parser.add_argument('-rd', '--result_dir', required=True, type=str,  help='dataset directory')
    args = parser.parse_args()
    return args

args = parse_arguments()
print(args)

master_dir = args.result_dir
"""Helper script, to be able to call functions in find_best_teacher module using simple
command-line arguments"""
print('scanning in dir: {}'.format(master_dir))
averages = compile_model_summary(master_dir)

with open('{}/model_summary.p'.format(master_dir), 'wb') as pfile:
    pickle.dump(averages, pfile)

"""Helper script, to be able to call functions in find_best_teacher module using simple
command-line arguments"""
print('dumping in dir: {}'.format(master_dir))

with open('{}/model_summary.p'.format(master_dir), 'rb') as pfile:
    averages = pickle.load(pfile)

csv_dump(averages, master_dir)

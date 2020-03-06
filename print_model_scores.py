import os, sys
import numpy as np

from find_best_teacher import *
script, master_dir = sys.argv


"""Helper script, to be able to call functions in find_best_teacher module using simple
command-line arguments"""

compile_model_summary(master_dir)

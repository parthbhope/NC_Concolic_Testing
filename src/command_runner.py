import os

command = 'python deepconcolic.py --model ../saved_models/custom_new.h5 --dataset custom --criterion nc --outputs outs/ --max-iterations 100'
os.system(command)
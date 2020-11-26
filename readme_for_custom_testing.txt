1. To run test on customised MNIST-complicated model 
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --dataset custom-2 --outputs outs/ --max-iterations 50

2. To run on customised MNIST2 model 
python deepconcolic.py --model ../saved_models/mnist2.h5 --dataset custom-2 --outputs outs/ --max-iterations 50

3. To run test on conv_1.h5 model (combination of conv2d over a simple ANN)
python deepconcolic.py --model ../saved_models/conv_1.h5  --dataset custom --outputs outs/ --max-iterations 100

Note : Upon running any of the above commands the user will be promted to select one input dataset from 1n/2n/3n/4n/5n/6n.csv 

set --max-iterations to any higher value for more testing iterations 


Google colab link : https://colab.research.google.com/drive/15S0jfhHHFQY8uHiNR-2bozQb_FrUflOi?usp=sharing

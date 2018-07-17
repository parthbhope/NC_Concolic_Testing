import argparse
import sys
import os
from datetime import datetime

import keras
from keras.models import *
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.layers import *
from keras import *
from utils import *
from nc_lp import *
from lp_encoding import *

def run_nc_linf(test_object, outs):
  print('\n== nc, linf ==\n')
  if not os.path.exists(outs):
    os.system('mkdir -p {0}'.format(outs))
  if not outs.endswith('/'):
    outs+='/'
  nc_results=outs+'nc_report-{0}.txt'.format(str(datetime.now()).replace(' ', '-'))
  nc_results=nc_results.replace(':', '-')

  layer_functions=get_layer_functions(test_object.dnn)
  print('\n== Got layer functions: {0} ==\n'.format(len(layer_functions)))
  cover_layers=get_cover_layers(test_object.dnn, 'NC')
  print('\n== Got cover layers: {0} ==\n'.format(len(cover_layers)))

  batch=10000
  if len(test_object.raw_data.data)<batch: batch=len(test_object.raw_data.data)
  activations = eval_batch(layer_functions, test_object.raw_data.data[0:batch])
  #print(activations[0].shape)
  #print(len(activations))

  calculate_pfactors(activations, cover_layers)

  #### configuration phase done

  test_cases=[]
  adversarials=[]

  xdata=test_object.raw_data.data
  iseed=np.random.randint(0, len(xdata))
  im=xdata[0]

  test_cases.append(im)
  update_nc_map_via_inst(cover_layers, eval(layer_functions, im))
  covered, not_covered=nc_report(cover_layers)
  #print (covered)
  print('\n== neuron coverage: {0}==\n'.format(covered*1.0/(covered+not_covered)))
  #print (np.argmax(test_object.dnn.predict(np.array([im]))))
  #return
  #y = test_object.dnn.predict_classes(np.array([im]))[0]
  #y=(np.argmax(test_object.dnn.predict(np.array([im]))))
  save_an_image(im, 'seed-image', outs)
  #return
  f = open(nc_results, "a")
  f.write('NC-cover: {0} #test cases: {1} #adversarial examples: {2}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials)))
  f.close()


  base_constraints=create_base_constraints(test_object.dnn)

  while True:
    index_nc_layer, nc_pos, nc_value=get_nc_next(cover_layers)
    #print (nc_layer.layer_index, nc_pos, nc_value/nc_layer.pfactor)
    nc_layer=cover_layers[index_nc_layer]
    print (np.array(nc_layer.activations).shape)
    shape=np.array(nc_layer.activations).shape
    pos=np.unravel_index(nc_pos, shape)
    im=test_cases[pos[0]]
    act_inst=eval(layer_functions, im)


    s=pos[0]*int(shape[1]*shape[2])
    if nc_layer.is_conv:
      s*=int(shape[3])*int(shape[4])
    print ('\n::', nc_pos, pos, nc_pos-s)
    print (nc_layer.layer, nc_layer.layer_index)
    print ('the max v', nc_value)

    mkey=nc_layer.layer_index
    if act_in_the_layer(nc_layer.layer) != 'relu':
      mkey+=1
    feasible, d, new_im=negate(test_object.dnn, act_inst, [im], nc_layer, nc_pos-s, base_constraints[mkey])

    cover_layers[index_nc_layer].disable_by_pos(pos)
    if feasible:
      print ('\nis feasible!!!\n')
      test_cases.append(new_im)
      update_nc_map_via_inst(cover_layers, eval(layer_functions, new_im))
      #y1 = test_object.dnn.predict_classes(np.array([im]))[0]
      #y2= test_object.dnn.predict_classes(np.array([new_im]))[0]
      y1 =(np.argmax(test_object.dnn.predict(np.array([im])))) 
      y2= (np.argmax(test_object.dnn.predict(np.array([im]))))
      if y1 != y2: adversarials.append([im, new_im])
      old_acts=eval(layer_functions, im)
      new_acts=eval(layer_functions, new_im)
      if nc_layer.is_conv:
        print ('\n should be < 0', old_acts[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]], '\n')
        print ('\n should be > 0', new_acts[nc_layer.layer_index][pos[1]][pos[2]][pos[3]][pos[4]], '\n')
    else:
      print ('\nis NOT feasible!!!\n')
    covered, not_covered=nc_report(cover_layers)
    f = open(nc_results, "a")
    f.write('NC-cover: {0} #test cases: {1} #adversarial examples: {2}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials)))
    f.close()
    #break

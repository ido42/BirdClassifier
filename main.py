import cv2
import numpy as np
from conv2D import *
from pooling import *
import sys
import pickle
from matplotlib import pyplot as plt
from Classes.ANN import *
from logistic_regression import *
#from Train import *
from image_load import *

"""
birdsEncoded, birdsTrain, birdsTest, birdsTrainFile, birdsTestFile=image_load()
dict_keys=list(birdsTest.keys())
inp,output=shuffled_matrices(birdsTrain,birdsEncoded,3)
print("matrices are ready")

log=logistic_regression(np.shape(inp),0.2,3)
log.gradient_descent(inp,output)
print(("trained"))
rand_int=random.randint(0,2)
new_input=birdsTest[dict_keys[rand_int]]
print(dict_keys[rand_int])
encoded_result=log.classify(new_input.flatten())
print(encoded_result)"""

"""while True:
    cv2.imshow("m",birdsTrain['BARN OWL'][0])
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
Trainer = Train(3, 5, 25, 5, 0.1, [5, 5, 5, 2], 1, 'max')

for bird in range(len(train_species)):
    if train_species[bird] == [0, 1]:
        img = cv2.imread(imgTrainF+"\\"+train_birds[bird])
    else:
        img = cv2.imread(imgTrainBO+"\\"+train_birds[bird])
    Trainer.train_with_one_img(img, train_species[bird] )"""

""" 
print("matrices are ready")
print(np.shape(inp))
print(np.shape(output))
input_size=np.shape(inp)
shf_in, shf_out = shuffle_matrix(inp,output)
num_classes=3
print("shuffled")
with open('huge_shf_matrix_in.pickle', 'wb') as dump_var1:
    pickle.dump(shf_in, dump_var1)
with open('huge_shf_matrix_out.pickle', 'wb') as dump_var2:
    pickle.dump(shf_out, dump_var2)"""

"""
pickle_in = open('huge_shf_matrix_in.pickle', 'rb')
inp_mat = pickle.load(pickle_in)
pickle_out = open('huge_shf_matrix_out.pickle', 'rb')
out_mat = pickle.load(pickle_out)
learning_rate=0.1
smp_in=inp_mat[0:100]# take sample
smp_out=out_mat[0:100]
layers_neurons=[smp_in.shape[1],100,100,100,10,3]
network=ANN(learning_rate, layers_neurons)
network.forward_pass(smp_in[0])
for i in range(smp_in.shape[0]):
    while any((abs(smp_out[i]-network.softmax_out.transpose())>np.ones((1,np.size(smp_out[i])))*0.05)[0]):
        network.forward_pass(smp_in[i])
        network.back_prop_m(smp_out[i])
        e=smp_out[i] - network.softmax_out.transpose()
print(("trained"))
network.forward_pass(inp_mat[10])
bird_num=int(np.where(network.softmax_out==np.max(network.softmax_out))[0])
bird_class_encoded=np.zeros((1,3))
bird_class_encoded[:,bird_num]=1
print(bird_class_encoded)
print(out_mat[10])"""

learning_rate=0.1
num_classes=3
reg_lambda=0.2
input_size=inp_mat[0:10].shape
log=logistic_regression(input_size,learning_rate,num_classes,reg_lambda)
for i in range(10):
    smp_in=inp_mat[i/10:(i+1)/10]# take sample
    smp_out=out_mat[i/10:(i+1)/10]
    log.gradient_descent(smp_in, smp_out )
print("trained")

the_bird=log.classify(inp_mat[100])
print (the_bird)
print(out_mat[100])

# tests = 222
# count = 0
# for j in range(tests):
#     sys.stdout.write("\r----- {:3.4}% done -----".format(100*(j+1)/tests))
#     sys.stdout.flush()
#
# #   parameters
#     x = np.array(range(1000))
#     x = (x - x.mean())/np.sqrt(x.var())
#     learning_rate=0.01
#     layers_neurons=[1000,5,10]
#     network=ANN(learning_rate, layers_neurons)
#     result=np.array([0,0,1,0,0,0,0,0,0,0])
#     soft_list=[]
#
#     for i in range(1000):
#         network.forward_pass(x)
#         # print(network.softmax_out)
#         res = True in [ele < 10e-15 or np.isnan(ele) for ele in network.softmax_out] # if a value is dangerously small or NaN
#         if res:
#             # print('Epoch', i + 1, 'overflows.')
#             # if 1-network.softmax_out[2] < 10e-90:
#                 # print('Successfully converged.')
#             break
#         network.back_prop(result)
#         soft_list.append(network.softmax_out)
#
#     if 1 - network.softmax_out[2] < 0.05: # if the output is sufficiently close to the result
#         count += 1
#         # print('Successfully converged.')
#     # else:
#         # print('Failed to converge.')
# print('\n{:3.4}% accuracy.'.format(100*count/tests))
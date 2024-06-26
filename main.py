import pickle
from logistic_regression import *
from Train import *
import sys
from time import time
from image_load import *
from Classes.confusion_matrix import *
#birdsEncoded, birdsTrain, birdsTest, birdsTrainFile, birdsTestFile=image_load()


# pickle_log = open('trained_log_reg.pickle', 'rb')
# log_reg = pickle.load(pickle_log)
# pickle_in = open('pooled_in_56_10birds.pickle', 'rb')
# inp_mat = pickle.load(pickle_in)
# pickle_out = open('pooled_out_56_10birds.pickle', 'rb')
# out_mat = pickle.load(pickle_out)
# learning_rate=0.2
# num_classes=10
# reg_lambda=0.01
# input_size=inp_mat[0:55].shape
#
# conf=confusion_mat(10)
# p=[]
# inp_mat, out_mat = shuffle_matrix(inp_mat, out_mat)
# for i in range(inp_mat.shape[0]):
#     pred,f=log_reg.classify(inp_mat[i])
#     true=out_mat[i]
#     p.append(pred)
#     con,fail=conf.update(true,pred,f)
# print(con)
# print(conf.fail)
"""
for e in range(10):
    inp_mat,out_mat=shuffle_matrix(inp_mat, out_mat)
    s=inp_mat.shape[0]//100
    for b in range(100):
        log.gradient_descent(inp_mat[b*s:(b+1)*s], out_mat[b*s:(b+1)*s])
        print("epoch"+str(e)+"batch"+str(b))
with open('trained_log_reg.pickle', 'wb') as dump_var1:
    pickle.dump(log, dump_var1)
conf=confusion_mat(10)
p=[]

dict_keys=list(birdsTest.keys())
rand_int=random.randint(0,2)
new_input=birdsTest[dict_keys[rand_int]]
print(np.shape(new_input[0]))

chosen=new_input[rand_choose]
chosen=chosen.reshape(1,50176)
print(chosen)
print(chosen.shape)

rand_choose=random.randint(0,len(inp_mat))

print(ann.softmax_out)
ann.forward_pass(inp_mat[rand_choose].reshape(50176))
#bird_num = int(np.where(ann.softmax_out == np.max(ann.softmax_out))[0])
#bird_class_encoded = np.zeros((1, 3))
#bird_class_encoded[:, bird_num] = 1
print(ann.softmax_out)
print(out_mat[rand_choose])
"""

num_cl=10
l_r=0.2
reg_lambda=0.05
b_size=inp_mat.shape[0]//10
log_reg=logistic_regression(np.shape(inp_mat[0:b_size]),l_r,num_cl,reg_lambda)
for epoch in range(2):
    inp_mat,out_mat=shuffle_matrix(inp_mat,out_mat)
    for b in range(10):
        log_reg.gradient_descent(inp_mat[b*b_size:(b+1)*b_size],out_mat[b*b_size:(b+1)*b_size])
        print("epoch" + str(epoch)+"batch"+str(b))
    conf=confusion_mat(10)
    for er in range(inp_mat.shape[0]):
        pred,f=log_reg.classify(inp_mat[er])
        true=out_mat[er]
        confusion_mat.update(true,pred,f)
    print(conf.c_mat)
    print(conf.fail)
print(("trained"))
rand_int=random.randint(0,np.shape(inp_mat)[0])
new_input=inp_mat[rand_int]
encoded_result=log_reg.classify(new_input)
print(encoded_result)
print(out_mat[rand_int])


# for epoch in range(15):
#     inp_mat,out_mat=shuffle_matrix(inp_mat,out_mat)
#     network.dropout(0.5)
#     for i in range(inp_mat.shape[0]): # feed every image in train
#         c = 0
#         while any((abs(out_mat[i]-network.softmax_out.transpose())>np.ones((1,np.size(out_mat[i])))*0.02)[0]):
#             network.forward_pass(inp_mat[i])
#             network.back_prop_m(out_mat[i])
#             c+=1
#     print("epoch"+str(epoch))
# print(("trained"))
# print(network.softmax_out)
# rand_int=random.randint(0,out_mat.shape[0]-1)
# network.forward_pass(inp_mat[rand_int])
# print(network.softmax_out)
# print(out_mat[rand_int])
# rand_int=random.randint(0,out_mat.shape[0]-1)
# network.forward_pass(inp_mat[rand_int])
# print(network.softmax_out)
# print(out_mat[rand_int])
# with open('trained_ann.pickle', 'wb') as dump_var1:
#     pickle.dump(network, dump_var1)


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

rng = np.random.default_rng()
labels, picturesTrain, picturesTest, _, _ = image_load()
subsample = poolingLayer(56)
ann = ANN(0.01, (3136,32,12,3))

# t1 = time()
for epoch in range(5):
    t2 = time()
    for bird in rng.permutation(list(labels)):
        t3 = time()
        count = 0
        for picture in picturesTrain[bird][epoch]:
            sys.stdout.write("\r----- "+bird+" {:3.4}% done -----".format(100 * count / len(picturesTrain[bird][epoch])))
            sys.stdout.flush()

            t4 = time()
            picSub = subsample.pool(picture).reshape(56*56,1)
            ann.forward_pass(picSub)
            # ann.dropout(0.5)
            ann.back_prop_m(labels[bird].reshape(3,1))
            d4 = time() - t4
            count += 1
            # epoch
        # d3 = time() - t3
        # epoch
    print("\nEpoch:", epoch, "Error:", (ann.loss**2).mean()/np.sum([len(picturesTrain[i][0]) for i in list(labels)]), "\n", time() - t2)
    ann.loss = []
    count = 0
    for bird in rng.permutation(list(labels)):
        for picture in picturesTest[bird][epoch]:
            t8 = time()
            picSub = subsample.pool(picture).reshape(56 * 56, 1)
            ann.forward_pass(picSub)
            if np.where(ann.out == ann.out.max())[0][0] == np.where(labels[bird] == 1)[0][0]:
                count += 1
            d8 = time() - t8
    print("\n",count / np.sum([len(picturesTest[i][0]) for i in list(labels)]) * 100, '%')
    # epoch
# d1 = time() - t1
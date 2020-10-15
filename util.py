from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from scipy.spatial.distance import pdist, cdist, squareform
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.utils import np_utils
import pdb
from scipy.stats import entropy
import keras
import keras.backend as K
from keras.models import Model
from sklearn.cluster import KMeans


# Set random seed
np.random.seed(123)
NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'celeb': 20}

def lid(logits, k=20):
    """
    Calculate LID for a minibatch of training samples based on the outputs of the network.

    :param logits:
    :param k:
    :return:
    """
    epsilon = 1e-12
    batch_size = tf.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
        tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k / v_log
    return lids


def mle_single(data, x, k):
    """
    lid of a single query point x.
    numpy implementation.

    :param data:
    :param x:
    :param k:
    :return:
    """
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]


def mle_batch(data, batch, k):
    """
    lid of a batch of query points X.
    numpy implementation.

    :param data:
    :param batch:
    :param k:
    :return:
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    #print(current_class)
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


def get_lids_random_batch(model, X, k=20, batch_size=128):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model: if None: lid of raw inputs, otherwise LID of deep representations
    :param X: normal images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    if model is None:
        lids = []
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        for i_batch in range(n_batches):
            start = i_batch * batch_size
            end = np.minimum(len(X), (i_batch + 1) * batch_size)
            X_batch = X[start:end].reshape((end - start, -1))

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch = mle_batch(X_batch, X_batch, k=k)
            lids.extend(lid_batch)

        lids = np.asarray(lids, dtype=np.float32)
        return lids

    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
             for out in [model.get_layer("lid").output]]
    lid_dim = len(funcs)

    #     print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i] = mle_batch(X_act, X_act, k=k)

        return lid_batch

    lids = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in range(n_batches):
        lid_batch = estimate(i_batch)
        lids.extend(lid_batch)

    lids = np.asarray(lids, dtype=np.float32)

    return lids


def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset:
    :param noise_ratio:
    :return:
    """
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)


def uniform_noise_model_P(num_classes, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (num_classes - 1).
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = noise / (num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(P, (1 - noise) * np.ones(num_classes))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def get_deep_representations(model, X, batch_size=128):
    """
    Get the deep representations before logits.
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    output_dim = model.layers[-3].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-3].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output

# make prediction by quality_model, compare the predicted label and y_train
def select_clean_uncertain(X_train, y_train_noisy, y_clean, quality_model, classifier):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    maxProbTP = 0
    maxProbFP = 0
    predict_prob = quality_model.predict(X_train, batch_size=128, verbose=0, steps=None)
    y_predict_label = predict_prob.argmax(axis=-1)
    clean_list_quality = []
    dirty_list_quality = []
    for i in np.arange(len(y_predict_label)):
      if (np.argmax(y_train_noisy[i]) == y_predict_label[i]): #or ((np.argmax(y_train_noisy[i]) != y_predict_label[i]) and np.max(predict_prob[i,:]) < 0.8) :
        clean_list_quality.append(i)
        if (np.argmax(y_train_noisy[i]) != np.argmax(y_clean[i])): #Real Noisy
          FN = FN + 1
        else:
          TN = TN + 1
 
      else:
        dirty_list_quality.append(i)
        
        if (np.argmax(y_train_noisy[i]) != np.argmax(y_clean[i])): #Real Noisy
          TP = TP +1
          maxProbTP = maxProbTP + np.max(predict_prob[i,:])
        else:
          FP = FP +1
          maxProbFP = maxProbFP + np.max(predict_prob[i,:])
       

    # y_predict_label_classifier = classifier.predict(X_train[dirty_list_quality], batch_size=128, verbose=0, steps=None)
    # y_predict_label_classifier = y_predict_label_classifier.argmax(axis=-1)
    # clean_list_classifier = []
    # dirty_list_classifier = []
    # y_train_classifier = y_train[dirty_list_quality]
    # for i in np.arange(len(dirty_list_quality)):
    #     if (np.argmax(y_train_classifier[i]) == y_predict_label_classifier[i]):
	#         clean_list_classifier.append(i)
    #     else:
    #         dirty_list_classifier.append(i)
    #
    # clean_list = np.append(clean_list_classifier, clean_list_quality)
    # uncertain_list = dirty_list_classifier

    #return clean_list, uncertain_list
    return clean_list_quality, dirty_list_quality, predict_prob, FN, TN, TP, FP, maxProbTP, maxProbFP


def select_clean_noisy(X_train, y_train_noisy, y_clean, y_predict, y_predict_label_second, predict_prob, model):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    maxProbTP = 0
    maxProbFP = 0

    clean_pred_list = []
    clean_list_model = []
    dirty_list_model = []
    for i in np.arange(len(y_predict[:,0])):
      if (np.argmax(y_train_noisy[i,:]) == np.argmax(y_predict[i,:])):# or y_predict_label_second[i]== np.argmax(y_train_noisy[i])): #filter 1
        clean_list_model.append(i)
        if (np.argmax(y_train_noisy[i,:]) != np.argmax(y_clean[i,:])): #Real Noisy
          FN = FN + 1
        else:
          TN = TN + 1
 
      else:
        
        dirty_list_model.append(i)
        
        if (np.argmax(y_train_noisy[i,:]) != np.argmax(y_clean[i,:])): #Real Noisy
          TP = TP +1
          maxProbTP = maxProbTP + np.max(predict_prob[i,:])
        else:
          FP = FP +1
          maxProbFP = maxProbFP + np.max(predict_prob[i,:])

    return clean_list_model, clean_pred_list, dirty_list_model, FN, TN, TP, FP, maxProbTP, maxProbFP

    
def al_number_generator(dynamic, num_al, budget, used_budget, predict_prob, loss_old):
    if dynamic == 0:
        return num_al, 0 , 0
    else:
        if used_budget < budget :
            #flag = 0
            loss_new = sum(entropy(np.transpose(predict_prob)))/len(predict_prob[:,0])
            num_al = int(np.ceil(num_al * (1 - ((loss_old - loss_new)/loss_new)))) #self.AL-int(np.sign(loss_old - loss_new
                                    #))*int(np.ceil(np.absolute(0.3*((loss_old - loss_new)/loss_new))))#
            
            if num_al <0:
              pdb.set_trace()
            used_budget = used_budget + num_al
            if used_budget > budget :
              num_al = num_al - (used_budget - budget)
              used_budget = budget
              #flag = 1  
            
        elif used_budget > budget :
            num_al = num_al - (used_budget - budget)
            used_budget = budget
            loss_new = loss_old
        else:
            num_al = 0
            loss_new = loss_old
            used_budget = budget
            #flag = 1

        return num_al, loss_new, used_budget#, flag


def al_number_gen_ws(dynamic, num_als, num_alw, budget_s, used_budget_s, budget_w, used_budget_w, predict_prob, loss):

    if dynamic == 0:
        num_als = num_als
        num_al_w = num_alw
        loss = 0 
        
    ########## dynamic case and loss ######################
    return num_als, num_alw

def active_selection(predict_prob, al_method, num_al):
    if num_al > 0:
      if (al_method == "unc"):
      # Choose the points with the lowest best class probability
        inf_ind = np.argsort(np.max(predict_prob, axis = 1))[0:num_al]
      elif (al_method == "bvssb"):
        # Choose the points with the lowest Best vs. Second best probability
        sorted_prob = np.sort(predict_prob)
        bestVSsecond = sorted_prob[:, -1] - sorted_prob[:, -2]
        inf_ind = np.argsort(bestVSsecond)[0:num_al]
      elif (al_method == "ent"):
        # Choose the points with the lowest best class probability
        entr = entropy(np.transpose(predict_prob))
        inf_ind = np.argsort(entr)[-num_al:]

      #oracle_list = ((np.array(dirty_list_quality,dtype='int64'))[inf_ind]).tolist()
    else:
      #oracle_list = []
      inf_ind = []  


    return inf_ind
    
def BNN_active_selection(predict_prob, dirty_list_quality, al_method, num_al):
    if num_al > 0:
      if (al_method == "unc"):
      # Choose the points with the lowest best class probability
        inf_ind = np.argsort(np.max(predict_prob[dirty_list_quality,:], axis = 1))[0:num_al]
        inf = np.max(predict_prob[inf_ind,:], axis = 1)
      elif (al_method == "bvssb"):
        # Choose the points with the lowest Best vs. Second best probability
        sorted_prob = np.sort(predict_prob[dirty_list_quality])
        bestVSsecond = sorted_prob[:, -1] - sorted_prob[:, -2]
        inf_ind = np.argsort(bestVSsecond)[0:num_al]
        inf = bestVSsecond[inf_ind]
      elif (al_method == "ent"):
        # Choose the points with the lowest best class probability
        entr = entropy(np.transpose(predict_prob[dirty_list_quality]))
        inf_ind = np.argsort(entr)[-num_al:]
        inf = entr(inf_ind)
      oracle_list = ((np.array(dirty_list_quality,dtype='int64'))[inf_ind]).tolist()
    else:
      oracle_list = []
      inf_ind = []  
      inf = []

    return inf_ind, inf

# make prediction by classifier
# def select_clean_quality(X_train, y_train, quality_model, dataset):
#     y_predict_label = quality_model.predict(X_train, batch_size=128, verbose=0, steps=None)
#     y_predict_label = y_predict_label.argmax(axis=-1)
#     clean_list = []
#     for i in np.arange(len(y_predict_label)):
#         if (np.argmax(y_train[i]) == y_predict_label[i]):
# 	    clean_list.append(i)
#     X_clean_train = X_train[clean_list]
#     y_clean_train = np_utils.to_categorical(y_train[clean_list], NUM_CLASSES[dataset])
#     print("X_clean_train:", X_clean_train.shape)
#     print("y_clean_train:", y_clean_train.shape)
#     return X_clean_train, y_clean_train, clean_list;

# make prediction by quality_model, compare the predicted label and y_train
def select_clean_knn(X_train, y_train, quality_model,dataset):
    X_train_linear = np.reshape(X_train, (X_train.shape[0], -1))
    y_predict_label = quality_model.predict(X_train_linear, k=10)
    #y_predict_label = y_predict_label.argmax(axis=-1)
    clean_list = []
    for i in np.arange(len(y_predict_label)):
        if (y_train[i].argmax(axis=-1) == y_predict_label[i].argmax(axis=-1)):
            clean_list.append(i)
    X_clean_train = X_train[clean_list]
    #y_clean_train = np_utils.to_categorical(y_train[clean_list], NUM_CLASSES[dataset])
    print("X_clean_train:", X_clean_train.shape)
    print("y_clean_train:", y_train[clean_list].shape)
    return X_clean_train, y_train[clean_list];

# combine two h
def combine_result(h, h_training_epoch):
    h.history['accuracy'] += h_training_epoch.history['accuracy']
    h.history['loss'] += h_training_epoch.history['loss']
    h.history['val_accuracy'] += h_training_epoch.history['val_accuracy']
    h.history['val_loss'] += h_training_epoch.history['val_loss']
    return h;

def inject_noise(dataset, y, noise_level):
    if noise_level > 100 or noise_level < 0:
        raise ValueError('Noise level can not be bigger than 100 or smaller than 0')
    if dataset == "cifar-10" or dataset == "celeb" or dataset == "mnist":
        noisy_idx = np.random.choice(len(y), int(len(y)*noise_level/100.0), replace = False)
        for i in noisy_idx:
            y[i] = np_utils.to_categorical(other_class(NUM_CLASSES[dataset], y[i].argmax(axis=-1)), NUM_CLASSES[dataset])

    return y, noisy_idx
    
def BNN_label_predict(X_train, model, n_classes):
    
    predict_prob = model.predict(X_train, batch_size=128)
    y_predict_label = np.argmax(predict_prob, axis = -1)
    y_predict_label_second = np.argsort(predict_prob, axis = -1)[:,-2]
    y_predict = np.zeros((len(y_predict_label), n_classes))
    for i in range (len(y_predict_label)):
      y_predict[i,:] = np_utils.to_categorical(y_predict_label[i], n_classes)
      
    return y_predict, predict_prob, y_predict_label_second
    
def y_weak_generator (y_true, n_classes,n_w):
    y_weak = np.zeros((len(y_true[:,0]), n_w))
    for i in range (0,len(y_true[:,0])):
        other_class = np.arange(n_classes)
        current_class = y_true[i].argmax(axis=-1)
        other_class = np.delete(other_class,current_class)
        weak_ind = np.random.choice(n_classes-1, n_w - 1)
        y_weak[i,:] = np.append(other_class[weak_ind], y_true[i].argmax(axis=-1))

    return y_weak
    

def kmeans(model, dirty_list_model, x_train_iteration, n_classes, state, inf_ind):
    
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('lid').output)
    feats = intermediate_layer_model.predict(x_train_iteration[dirty_list_model])
    kmeans = KMeans(n_clusters= n_classes, random_state=0).fit(feats)
    kmeans_list = []
    shorten_k_list = []
    
    if state == "first":
        for i in range (0,n_classes):
            kmeans_list.append(np.random.choice(np.where(kmeans.labels_== i)[0],30))
            flat_list = [item for sublist in kmeans_list for item in sublist]
    else:
        for i in range (0,n_classes):
            #kmeans_list.append((dirty_list_model[inf_ind[np.where(kmeans.labels_== i)[0]][0:30]]))
            cluster_ind = np.where(kmeans.labels_== i)[0]
            kmeans_list = []
            for j in range (0, len(inf_ind)):
                if inf_ind[j] in cluster_ind:
                     kmeans_list.append(inf_ind[j])
            shorten_k_list.append(kmeans_list[0:30])
            flat_list = [item for sublist in shorten_k_list for item in sublist]

    
    return flat_list
    
def stat_func (strong_list, weak_list, clean_list, noisy_list, kmeans_list, y_noisy, y_true):
    FN_s = 0
    FN_d = 0
    FN_c = 0
    if strong_list == []:
      oracle_list = weak_list
    elif weak_list == []:
      oracle_list = strong_list
    else:
      oracle_list = np.append(strong_list, weak_list)   
    
    for i in range (0, len (noisy_list)):
        if noisy_list[i] in oracle_list: ##### Selected Noisy for cleaning
            if np.argmax(y_noisy[noisy_list[i],:]) == np.argmax(y_true[noisy_list[i],:]): ## Was clean but seelected as noisy
                FN_s += 1
        else: ### ALL discarded ones
            if np.argmax(y_noisy[noisy_list[i],:]) == np.argmax(y_true[noisy_list[i],:]): ## Was clean but discarded
                FN_d += 1
                
        if i in kmeans_list: ##### Selected Noisy and clustered for cleaning
            if np.argmax(y_noisy[noisy_list[i],:]) == np.argmax(y_true[noisy_list[i],:]): ## Was clean but seelected as noisy in clustering
                FN_c += 1  
             
    return FN_s, FN_d, FN_c 

 
def label_decider (dirty_list_model, y_true, y_pred, pred_prob, n_classes, inf_ind, inf, loss, batch_budget, spent_s, spent_w, limit_w, c, alpha, beta, thr):

    L = loss
    #print("loss:", loss)
    strong_list = []
    weak_list = []
    first_yes = 0
    weak_questions = []
    unique_pic = 0
    true_label_rank = []
    
    for i in range (0, len(inf_ind)):
        pred_prob_sort = np.argsort(pred_prob[dirty_list_model[inf_ind[i]],:])
        true_label_rank.append(n_classes-1- np.where(np.argmax(y_true[dirty_list_model[inf_ind[i]],:]) == pred_prob_sort)[0])
    
        I = inf[i]
        if spent_s == 0:
            cost_s = 1
        else:
            cost_s = spent_s
        
        if spent_w == 0:
            cost_w = 1
        else:
            cost_w = spent_w   
            
        Q_s = loss/((pow(cost_s,beta))*(pow(I,alpha)))

        if Q_s>thr:
            if spent_s + spent_w + c <= batch_budget:
                strong_list.append(dirty_list_model[inf_ind[i]])
                spent_s = spent_s + c
            elif spent_s + spent_w + 1 <= batch_budget:
                #pdb.set_trace()
                yes_flag = y_weak_yn_generator (y_pred[dirty_list_model[inf_ind[i]],:], y_true[dirty_list_model[inf_ind[i]],:]) ####### First Ask the Predicted Label
                y_ask = [np.argmax(y_pred[dirty_list_model[inf_ind[i]],:])]
                
                unique_pic += 1
                spent_w = spent_w + 1
                if (yes_flag):
                    weak_list.append(dirty_list_model[inf_ind[i]])
                    first_yes +=1
                else:
                    pred_prob_sort = np.argsort(pred_prob[dirty_list_model[inf_ind[i]],:]) ###### smart
                    
                    for k in range (0,limit_w):
                      if spent_s + spent_w + 1 <=batch_budget:
                        pred_prob_sort = np.delete(pred_prob_sort,np.where(y_ask[-1] == pred_prob_sort)[0]) ####### smart
                        y_ask.append(pred_prob_sort[-1]) ######## smart
      
                        k = k+1
                        #pdb.set_trace()
                        yes_flag = y_weak_yn_generator(np_utils.to_categorical(y_ask[-1],n_classes), y_true[dirty_list_model[inf_ind[i]],:])
                        spent_w = spent_w + 1
        
                        if (yes_flag):
                            weak_list.append(dirty_list_model[inf_ind[i]])
                            weak_questions.append(k)
                            break
                
        elif spent_s + spent_w + 1 <= batch_budget:
            #pdb.set_trace()
            yes_flag = y_weak_yn_generator (y_pred[dirty_list_model[inf_ind[i]],:], y_true[dirty_list_model[inf_ind[i]],:]) ####### First Ask the Predicted Label
            y_ask = [np.argmax(y_pred[dirty_list_model[inf_ind[i]],:])]
            
            unique_pic += 1
            spent_w = spent_w + 1
            if (yes_flag):
                weak_list.append(dirty_list_model[inf_ind[i]])
                first_yes +=1
            else:
                pred_prob_sort = np.argsort(pred_prob[dirty_list_model[inf_ind[i]],:]) ####### smart
                for k in range (0,limit_w):
                  if spent_s + spent_w + 1 <=batch_budget:
                    pred_prob_sort = np.delete(pred_prob_sort,np.where(y_ask[-1] == pred_prob_sort)[0]) ####### smart
                    y_ask.append(pred_prob_sort[-1]) ######## smart
      
                    k = k+1
                    #pdb.set_trace()
                    yes_flag = y_weak_yn_generator(np_utils.to_categorical(y_ask[-1],n_classes), y_true[dirty_list_model[inf_ind[i]],:])
                    spent_w = spent_w + 1
        
                    if (yes_flag):
                        weak_list.append(dirty_list_model[inf_ind[i]])
                        weak_questions.append(k)
                        break
     
    #print("first yes:", first_yes) 
    #print("weak questions:", weak_questions)                    
    #print("mean weak questions:", np.mean(weak_questions))
    #print("unique_pic:", np.mean(unique_pic))
    #print("Average True Label Rank:", np.mean(true_label_rank))
    return strong_list, weak_list, spent_s, spent_w, first_yes, weak_questions, np.mean(weak_questions), unique_pic, true_label_rank
                

def y_weak_yn_generator (y_given, y_true):
    if np.argmax(y_given) == np.argmax(y_true):
        yes_flag = 1
    else:
        yes_flag = 0

    return yes_flag

def model_reliability (model, val_data, val_y):
    rel_flag = 0  
    loss, val_acc = model.evaluate(val_data, val_y, verbose = 0)
     
    return loss

def cost_control(i,inf_ind): #change
    if i in inf_ind:
      flag = 1
    else:
      flag = 0

    return flag

    
def all_noisy_selection (noisy_idx, num_al):
    inf_ind = np.random.choice(noisy_idx, num_al)
    oracle_list = inf_ind.tolist()

    return oracle_list, inf_ind
    
def dropout_inference (model, X, batch_size, T, pred=None, dropout = 1):
    if pred == None:
        pred = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
        
    nbatch = int(np.ceil(X.shape[0]/batch_size))
    
    Yt_hat = np.array(pred([X[0:batch_size], dropout]))
    
    for i in range(1,nbatch):
        Yt_tmp = np.array( pred([X[ i*batch_size:np.min(((i+1)*batch_size,X.shape[0])) ], dropout]) )
        Yt_hat = np.concatenate((Yt_hat,Yt_tmp),axis=1)
        
    for j in range(1,T):
        print(' Bayesian inference: j =',j+1,'/',T,end='\r')
        Yt_hat2 = np.array( pred([X[ 0:batch_size ], dropout]) )
        
        for i in range(1,nbatch):
            Yt_tmp = np.array( pred([X[ i*batch_size:np.min(((i+1)*batch_size,X.shape[0])) ], dropout]) )
            Yt_hat2 = np.concatenate((Yt_hat2,Yt_tmp),axis=1)
          
        Yt_hat = np.concatenate((Yt_hat,Yt_hat2),axis=0)
        
    print(' ')
    return Yt_hat

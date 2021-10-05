import os
import numpy as np
import pandas as pd
import time
from collections import deque
import warnings
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
import random
import pickle
import math
import datetime

from risk_losses import geoRisk
from tfr_losses_local import *


def RunGuidedRecParametrized(list_of_args):
    if not os.path.exists("./Output/"):
        os.makedirs("./Output/")
    if not os.path.isfile("./Output/" + "names.txt"):
        with open("./Output/" + "names.txt", "w") as fu:
            fu.close()
    with open("./Output/" + "names.txt", "a") as fu:
        fu.write(str(list_of_args))
        fu.close()
    #################################
    drop_rate = list_of_args[0]
    num_baseline_dropouts = list_of_args[1]
    local_losfun = list_of_args[2]
    do_multi_dropout_with_logloss = list_of_args[3]
    LOSSFUN = list_of_args[4]
    add_l2_reg_on_risk = list_of_args[5]
    eval_ideal_risk = list_of_args[6]
    do_diff_to_ideal_risk = list_of_args[7]
    add_loss_on_risk = list_of_args[8]
    alpha_risk = list_of_args[9]
    name_exec = list_of_args[10]

    #################################
    WGT_DECAY_LOGLOSS = 0.00001
    # LR_LOGLOSS = 0.0001
    LR_LOGLOSS = list_of_args[11]
    id = list_of_args[12]
    dataset = list_of_args[13]

    WGT_DECAY_SURROGATE = 0.0000007
    LR_SURROGATE = 0.0012

    WGT_DECAY_NDCG = 0.00001
    LR_NDCG = 0.00065
    #################################
    # dataset = "ml1m"

    BATCH_SIZE = 250
    NEGSAMPLES = 1
    USER_NUM = 943
    ITEM_NUM = 1682
    DIM = 50
    # EPOCH_MAX = 450
    EPOCH_MAX = 1
    # DEVICE = "/gpu:0"
    DEVICE = "/cpu:0"
    PERC = 0.9
    SURROGATE = 1
    SEED = 45
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    tf.random.set_random_seed(SEED)
    random.seed(SEED)
    print('Graph', "-", SURROGATE, "-", SEED)

    if not os.path.exists("./Output/" + name_exec + "/"):
        os.makedirs("./Output/" + name_exec + "/")

    ##########
    parameters = [BATCH_SIZE, NEGSAMPLES, DIM, EPOCH_MAX, SEED, LOSSFUN,
                  num_baseline_dropouts, local_losfun, add_l2_reg_on_risk, add_loss_on_risk, alpha_risk,
                  do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR_LOGLOSS, drop_rate, do_multi_dropout_with_logloss,
                  name_exec, id]

    parameters_names = "BATCH_SIZE, NEGSAMPLES, DIM, EPOCH_MAX, SEED, LOSSFUN,\
                              num_baseline_dropouts, local_losfun, add_l2_reg_on_risk, add_loss_on_risk, alpha_risk,\
                              do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR_LOGLOSS, drop_rate, do_multi_dropout_with_logloss, name_exec, id"

    with open("./Output/" + name_exec + "/" + "configs.txt", "w") as fo:
        for name, value in zip(parameters_names.replace(" ", "").split(","), parameters):
            fo.write(name + ":" + str(value) + "\n")
    ##########

    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn import preprocessing

    tf.reset_default_graph()

    def load_data(filename):
        try:
            with open(filename, "rb") as f:
                x = pickle.load(f)
        except:
            x = []
        return x

    def save_data(data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def inferenceDense(phase, ufsize, ifsize, user_batch, item_batch, time_batch, idx_user, idx_item, ureg, ireg,
                       user_num,
                       item_num, dim=5, UReg=0.0, IReg=0.0, device="/cpu:0"):
        with tf.device(device):
            user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
            item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")

            ul = tf.get_variable('ul', [ufsize, 20], initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED),
                                 dtype=tf.float64)
            ulb = tf.get_variable('ulb', [1, 20], initializer=tf.constant_initializer(0), dtype=tf.float64)

            il = tf.get_variable('il', [ifsize, 20], initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED),
                                 dtype=tf.float64)
            ilb = tf.get_variable('ilb', [1, 20], initializer=tf.constant_initializer(0), dtype=tf.float64)

            outputul1mf = tf.add(tf.matmul(user_batch, ul), ulb)
            outputil1mf = tf.add(tf.matmul(item_batch, il), ilb)

            ul1mf = tf.nn.crelu(outputul1mf)
            il1mf = tf.nn.crelu(outputil1mf)

            ###
            # tf.nn.dropout
            drops_u = []
            for i in range(num_baseline_dropouts):
                drops_u.append(tf.layers.dropout(ul1mf, rate=drop_rate, training=phase))
            drops_i = []
            for i in range(num_baseline_dropouts):
                drops_i.append(tf.layers.dropout(il1mf, rate=drop_rate, training=phase))

            infers_drop = []
            for i in range(num_baseline_dropouts):
                InferInputMF_drop = tf.multiply(drops_u[i], drops_i[i])
                infers_drop.append(tf.reduce_sum(InferInputMF_drop, 1, name="inference"))
            ###

            InferInputMF = tf.multiply(ul1mf, il1mf)

            infer = tf.nn.sigmoid(tf.reduce_sum(InferInputMF, 1, name="inference"))
            inferSig = tf.reduce_sum(InferInputMF, 1, name="inference")
            varlist = [ul, ulb, il, ilb]
            regularizer = tf.reduce_sum(
                tf.add(UReg * tf.nn.l2_loss(ul1mf), IReg * tf.nn.l2_loss(il1mf), name="regularizer"))
        return infer, infer, inferSig, infers_drop, varlist, regularizer

    # def optNDCG(NDCGscore, varlist, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    #     with tf.device(device):
    #         costNDCG = -NDCGscore
    #         train_NDCG = tf.contrib.opt.AdamWOptimizer(0.00008, learning_rate=0.0013).minimize(costNDCG,
    #                                                                                            var_list=varlist)
    #     return train_NDCG
    #
    # def optLoss(NDCGscore, NDCGTrue, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    #     with tf.device(device):
    #         costLOSS = tf.nn.l2_loss(tf.subtract(NDCGscore, NDCGTrue))
    #         train_LOSS = tf.contrib.opt.AdamWOptimizer(0.000001, learning_rate=0.002).minimize(costLOSS)
    #     return train_LOSS, costLOSS

    def optNDCG(NDCGscore, varlist, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
        with tf.device(device):
            costNDCG = -NDCGscore
            train_NDCG = tf.contrib.opt.AdamWOptimizer(WGT_DECAY_NDCG, learning_rate=LR_NDCG).minimize(costNDCG,
                                                                                                       var_list=varlist)
        return train_NDCG

    def optLoss(NDCGscore, NDCGTrue, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
        with tf.device(device):
            costLOSS = tf.nn.l2_loss(tf.subtract(NDCGscore, NDCGTrue))
            train_LOSS = tf.contrib.opt.AdamWOptimizer(WGT_DECAY_SURROGATE, learning_rate=LR_SURROGATE).minimize(
                costLOSS)
        return train_LOSS, costLOSS

    def optLoss2(NDCGscore, NDCGTrue, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
        with tf.device(device):
            costLOSS = tf.nn.l2_loss(tf.subtract(NDCGscore, NDCGTrue))
            grads = tf.reduce_sum(tf.concat([tf.reshape(g, [-1]) for g in tf.gradients(costLOSS, tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='Surrogate'), unconnected_gradients='zero')], axis=0))
            grads = tf.stop_gradient(grads)
            grads_square = tf.square(grads)
            tempgrads1 = tf.gradients(costLOSS, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Surrogate'),
                                      unconnected_gradients='zero')
            tempgrads2 = tf.gradients(tempgrads1[0],
                                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Surrogate'),
                                      unconnected_gradients='zero')
            grads_grads = tf.reduce_sum(tf.concat([tf.reshape(g, [-1]) for g in tempgrads2], axis=0))
            grads_grads = tf.stop_gradient(grads_grads)
            nominator = tf.abs(grads_grads)
            denominator = tf.pow(1.0 + grads_square, 1.5)
            curvature = tf.divide(nominator, denominator)
            AllLoss = curvature + costLOSS
            train_LOSS = tf.contrib.opt.AdamWOptimizer(0.00001, learning_rate=0.000015).minimize(AllLoss)
        return train_LOSS, costLOSS

    def optimization(infer, infers_drop, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
        global_step = tf.train.get_global_step()
        assert global_step is not None
        with tf.device(device):
            # cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,
            #                                                logits=infer)  # tf.nn.l2_loss(tf.subtract(infer, rate_batch))
            # train_op = tf.contrib.opt.AdamWOptimizer(WGT_DECAY_LOGLOSS, learning_rate=LR_LOGLOSS).minimize(cost,
            #                                                                                                global_step=global_step)
            # # GradientDescentOptimizer

            rate_batch = tf.dtypes.cast(rate_batch, tf.float32)
            regularizer = tf.dtypes.cast(regularizer, tf.float32)
            # cost = regularizer
            cost = tf.dtypes.cast(0, tf.float32)
            if local_losfun == "":
                if do_multi_dropout_with_logloss:
                    for infer_temp in infers_drop:
                        cast_infer = tf.dtypes.cast(infer_temp, tf.float32)
                        cost_drop = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,
                                                                            logits=cast_infer)
                        cost = tf.add(cost, cost_drop)
                else:
                    for infer_temp in infers_drop:
                        cast_infer = tf.dtypes.cast(infer_temp, tf.float32)
                        cost_drop, _ = globals()[LOSSFUN](tf.reshape(rate_batch, [25, 10]),
                                                          tf.reshape(cast_infer, [25, 10]))
                        cost = tf.add(cost, tf.reduce_sum(cost_drop))
            else:
                mat = []
                for infer_temp in infers_drop:
                    cast_infer = tf.dtypes.cast(infer_temp, tf.float32)
                    cost_drop, _ = globals()[local_losfun](tf.reshape(rate_batch, [25, 10]),
                                                           tf.reshape(cast_infer, [25, 10]))

                    if local_losfun == "GumbelApproxNDCGLossLocal":
                        cost_drop = -cost_drop
                    elif local_losfun == "NeuralSortCrossEntropyLossLocal":
                        cost_drop = cost_drop / tf.reduce_max(cost_drop)

                    mat.append(cost_drop)

                    if add_l2_reg_on_risk:
                        # cost_reg = tf.nn.l2_loss(tf.subtract(cast_infer, rate_batch))
                        cost_reg = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,
                                                                            logits=cast_infer)
                        cost = tf.add(cost, cost_reg)

                    if add_loss_on_risk:
                        cost = tf.add(cost, tf.reduce_sum(cost_drop))

                if eval_ideal_risk:
                    cc, _ = globals()[local_losfun](tf.reshape(rate_batch, [25, 10]), tf.reshape(rate_batch, [25, 10]))
                    cc = cc / tf.reduce_max(cc)
                else:
                    cc = tf.ones(cost_drop.shape, tf.float32)

                mat.append(cc)

                mat = tf.squeeze(mat)
                mat = tf.transpose(mat)

                if do_diff_to_ideal_risk:
                    # cost = tf.add(geoRisk(mat, alpha_risk) - geoRisk(mat, alpha_risk, i=-1), cost)
                    cost = tf.add(geoRisk(mat, alpha_risk, i=-1) - geoRisk(mat, alpha_risk), cost)
                else:
                    # cost = tf.add(geoRisk(mat, alpha_risk), cost)
                    cost = tf.add(-geoRisk(mat, alpha_risk), cost)

            # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            train_op = tf.contrib.opt.AdamWOptimizer(WGT_DECAY_LOGLOSS, learning_rate=LR_LOGLOSS).minimize(cost,
                                                                                                           global_step=global_step)

        return cost, train_op

    def clip(x):
        return np.clip(x, 1.0, 5.0)

    def make_scalar_summary(name, val):
        return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

    def dcg_score(y_true, y_score, k=10, gains="exponential"):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        if gains == "exponential":
            gains = 2 ** y_true - 1
        elif gains == "linear":
            gains = y_true
        else:
            raise ValueError("Invalid gains option.")

        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def ndcg_score(y_true, y_score, k=10, gains="exponential"):
        best = dcg_score(y_true, y_true, k, gains)
        actual = dcg_score(y_true, y_score, k, gains)
        return actual / best

    def get_UserData100k():
        col_names = ["user", "age", "gender", "occupation", "PostCode"]
        df = pd.read_csv('../Data/ml100k/u.user', sep='|', header=None, names=col_names, engine='python')
        del df["PostCode"]
        df["user"] -= 1
        df = pd.get_dummies(df, columns=["age", "gender", "occupation"])
        del df["user"]
        return df.values

    def get_ItemData100k():
        col_names = ["movieid", "movietitle", "releasedate", "videoreleasedate", "IMDbURL"
            , "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime", "Documentary"
            , "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi", "Thriller"
            , "War", "Western"]
        df = pd.read_csv('../Data/ml100k/u.item', sep='|', header=None, names=col_names, engine='python')
        df['releasedate'] = pd.to_datetime(df['releasedate'])
        df['year'], df['month'] = zip(*df['releasedate'].map(lambda x: [x.year, x.month]))
        df['year'] -= df['year'].min()
        df['year'] /= df['year'].max()
        df['year'] = df['year'].fillna(0.0)

        del df["month"]
        del df["movietitle"]
        del df["releasedate"]
        del df["videoreleasedate"]
        del df["IMDbURL"]

        df["movieid"] -= 1
        del df["movieid"]
        return df.values

    def getHR(ranklist, gtItem):
        if (gtItem in ranklist):
            return 1
        return 0

    def getNDCG(ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0

    def getNDCG_SIM(ranklist, gtItem, Sim):
        NewTestItem = -1
        ItemFeatRow = ITEMDATA[np.asarray([gtItem], dtype=np.int32), :]
        RankedFeatRows = ITEMDATA[np.asarray(ranklist, dtype=np.int32), :]

        simvals = cosine_similarity(RankedFeatRows, ItemFeatRow)
        simvals = (simvals + 1) / 2

        simvals[simvals < Sim] = 0.0

        for i in range(len(ranklist)):
            if (simvals[i] > 0.0 and ranklist[i] != gtItem):
                NewTestItem = ranklist[i]
                break
            if (simvals[i] > 0.0 and ranklist[i] == gtItem):
                NewTestItem = gtItem
                break
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == NewTestItem:
                return math.log(2) / math.log(i + 2)
        return 0

    def my_elementwise_func(x):
        return tf.matmul(x, tf.transpose(x))

    def GuidedRec(train, ItemData=False, UserData=False, Graph=True, lr=0.00003, ureg=0.0, ireg=0.0):
        AdjacencyUsers = [[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)]
        DegreeUsers = [[0 for x in range(1)] for y in range(USER_NUM)]
        DegreeUsersVec = [0 for y in range(USER_NUM)]
        VarUsersVec = [list() for y in range(USER_NUM)]
        VarUsersVals = [0 for y in range(USER_NUM)]

        AdjacencyItems = [[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)]
        DegreeItems = [[0 for x in range(1)] for y in range(ITEM_NUM)]
        DegreeItemsVec = [0 for y in range(ITEM_NUM)]
        VarItemsVec = [list() for y in range(ITEM_NUM)]
        VarItemsVals = [0 for y in range(ITEM_NUM)]
        if (Graph):
            for index, row in train.iterrows():
                userid = int(row['user'])
                itemid = int(row['item'])
                AdjacencyUsers[userid][itemid] = row['rate'] / 5.0
                AdjacencyItems[itemid][userid] = row['rate'] / 5.0
                DegreeUsersVec[userid] += 1
                DegreeItemsVec[itemid] += 1
                DegreeUsers[userid][0] += 1
                DegreeItems[itemid][0] += 1
                VarUsersVec[userid].append(row['rate'])
                VarItemsVec[itemid].append(row['rate'])

            for i in range(USER_NUM):
                VarUsersVals[i] = np.var(VarUsersVec[i])
            for i in range(ITEM_NUM):
                VarItemsVals[i] = np.var(VarItemsVec[i])

            DUserMax = np.amax(DegreeUsers)
            DItemMax = np.amax(DegreeItems)
            DegreeUsers = np.true_divide(DegreeUsers, DUserMax)
            DegreeItems = np.true_divide(DegreeItems, DItemMax)

            AdjacencyUsers = np.asarray(AdjacencyUsers)
            AdjacencyItems = np.asarray(AdjacencyItems)

        if (Graph):
            UserFeatures = np.concatenate((np.identity(USER_NUM), AdjacencyUsers, DegreeUsers), axis=1)
            ItemFeatures = np.concatenate((np.identity(ITEM_NUM), AdjacencyItems, DegreeItems), axis=1)
        else:
            UserFeatures = np.identity(USER_NUM)
            ItemFeatures = np.identity(ITEM_NUM)

        if (UserData):
            UsrDat = get_UserData100k()
            UserFeatures = np.concatenate((UserFeatures, UsrDat), axis=1)

        if (ItemData):
            ItmDat = get_ItemData100k()
            ItemFeatures = np.concatenate((ItemFeatures, ItmDat), axis=1)

        # samples_per_batch = len(train) // int(BATCH_SIZE / (NEGSAMPLES + 1))
        samples_per_batch = INTERACTIONS_TRAIN_NUM // int(BATCH_SIZE / (NEGSAMPLES + 1))

        user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
        item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
        y_batch = tf.placeholder(tf.float64, shape=[None, 1], name="y")
        m_batch = tf.placeholder(tf.float64, shape=[None, 1], name="m")
        d_batch = tf.placeholder(tf.float64, shape=[None, 1], name="d")
        dw_batch = tf.placeholder(tf.float64, shape=[None, 1], name="dw")
        dy_batch = tf.placeholder(tf.float64, shape=[None, 1], name="dy")
        w_batch = tf.placeholder(tf.float64, shape=[None, 1], name="w")
        ndcgTargets = tf.placeholder(tf.float64, shape=[25, 1], name="ndcgTargets")

        time_batch = tf.concat([y_batch, m_batch, d_batch, dw_batch, dy_batch, w_batch], 1)
        rate_batch = tf.placeholder(tf.float64, shape=[None])
        phase = tf.placeholder(tf.bool, name='phase')
        modelphase = tf.placeholder(tf.bool, name='modelphase')

        w_user = tf.constant(UserFeatures, name="userids", shape=[USER_NUM, UserFeatures.shape[1]], dtype=tf.float64)
        w_item = tf.constant(ItemFeatures, name="itemids", shape=[ITEM_NUM, ItemFeatures.shape[1]], dtype=tf.float64)

        infer, infern, inferSig, infers_drop, varlist, regularizer = inferenceDense(phase, UserFeatures.shape[1],
                                                                                    ItemFeatures.shape[1],
                                                                                    user_batch, item_batch, time_batch,
                                                                                    w_user, w_item,
                                                                                    ureg, ireg, user_num=USER_NUM,
                                                                                    item_num=ITEM_NUM,
                                                                                    dim=DIM,
                                                                                    device=DEVICE)

        infernStopped = tf.stop_gradient(infern)

        Infere2d = tf.reshape(infer, [25, 10])
        rate2d = tf.reshape(rate_batch, [25, 10])
        NDCG2d = tf.concat([Infere2d, rate2d], 1)

        Infere3d = tf.reshape(infer, [25, 10, 1])
        rate3d = tf.reshape(rate_batch, [25, 10, 1])
        NDCG3d = tf.concat([Infere3d, rate3d], 2)

        Infere2dn = tf.reshape(infernStopped, [25, 10])
        NDCG2dn = tf.concat([Infere2dn, rate2d], 1)

        Infere3dn = tf.reshape(infernStopped, [25, 10, 1])
        NDCG3dn = tf.concat([Infere3dn, rate3d], 2)

        ####################################################
        with tf.variable_scope("Surrogate"):
            ################# Copy of the surrogate model without access to the recsys parameters
            ##MLP
            l11 = tf.layers.dense(inputs=NDCG2dn, units=20, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l11")
            l12 = tf.layers.dense(inputs=l11, units=20, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l12")
            ##NFM
            b21 = tf.reshape(NDCG3dn, [-1, 2])
            l21 = tf.layers.dense(inputs=b21, units=20, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l21")
            b22 = tf.reshape(l21, [-1, 10, 20])
            b22 = tf.map_fn(my_elementwise_func, b22)
            b23 = tf.reduce_mean(b22, axis=1)
            l22 = tf.layers.dense(inputs=b23, units=20, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l22")
            l31 = tf.multiply(l22, l12)
            ##Comb
            l4 = tf.layers.dense(inputs=l31, units=8, activation=tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l4")
            NDCGScore_without_recsys = tf.layers.dense(inputs=l4, units=1, activation=tf.nn.tanh,
                                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                                       seed=SEED),
                                                       name="NDCGScore")

            ################# Copy of the surrogate model with access to the recsys parameters
            ## MLP
            l11n = tf.layers.dense(inputs=NDCG2d, units=20, reuse=True, trainable=False, activation=tf.nn.tanh,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l11")
            l12n = tf.layers.dense(inputs=l11n, units=20, reuse=True, trainable=False, activation=tf.nn.tanh,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l12")
            ## NFM
            b21n = tf.reshape(NDCG3d, [-1, 2])
            l21n = tf.layers.dense(inputs=b21n, units=20, reuse=True, trainable=False, activation=tf.nn.tanh,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l21")
            b22n = tf.reshape(l21n, [-1, 10, 20])
            b22n = tf.map_fn(my_elementwise_func, b22n)
            b23n = tf.reduce_mean(b22n, axis=1)
            l22n = tf.layers.dense(inputs=b23n, units=20, reuse=True, trainable=False, activation=tf.nn.tanh,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l22")
            l31n = tf.multiply(l22n, l12n)
            ##Comb
            l4n = tf.layers.dense(inputs=l31n, units=8, reuse=True, trainable=False, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), name="l4")
            NDCGScore_with_recsys = tf.layers.dense(inputs=l4n, units=1, reuse=True, trainable=False,
                                                    activation=tf.nn.tanh,
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                                    seed=SEED),
                                                    name="NDCGScore")

        ####################################################
        global_step = tf.contrib.framework.get_or_create_global_step()
        ## Logloss
        cost, train_op = optimization(inferSig, infers_drop, regularizer, rate_batch, learning_rate=lr, reg=0.09,
                                      device=DEVICE)
        ## Maximizing NDCG and only using the recsys model paramteres
        train_NDCG = optNDCG(NDCGScore_with_recsys, varlist, device=DEVICE)
        ## Minimizing the SE loss of the surrogate model without affecting the recsys model paramteres
        train_Loss, loss_cost = optLoss(NDCGScore_without_recsys, ndcgTargets, device=DEVICE)

        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        finalerror = -1
        degreelist = list()
        predlist = list()
        degreelist0 = list()
        predlist0 = list()
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            print("epochs, train_err,train_err1,train_err2,HitRatio5,HitRatio10,HitRatio20,NDCG5,NDCG10,NDCG20")
            now = datetime.datetime.now()
            textTrain_file = open(
                "./Output/" + name_exec + "/" + now.strftime('%Y%m%d%H%M%S') + '_Graph' + "_" + str(
                    SURROGATE) + "_" + str(SEED) + ".txt",
                "w", newline='')
            errors = deque(maxlen=samples_per_batch)
            logcost = deque(maxlen=samples_per_batch)
            losscost = deque(maxlen=samples_per_batch)
            truecost = deque(maxlen=samples_per_batch)
            start = time.time()

            bestndcg10 = 0
            print(samples_per_batch)
            for i in range(EPOCH_MAX * samples_per_batch):

                users, items, rates = GetTrainSample(grouped_instances, BATCH_SIZE, 10)
                ################################

                if (SURROGATE):
                    if ((
                            i // samples_per_batch) < 0):  ### change the zero value to the desired number of epochs if you want to warm up the surrogate NDCG model for some epochs first
                        _, pred_batch, cst = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                                                                                          item_batch: items,
                                                                                          rate_batch: rates,
                                                                                          phase: True})
                        inf2d, rt2d = sess.run([Infere2d, rate2d], feed_dict={user_batch: users,
                                                                              item_batch: items,
                                                                              rate_batch: rates,
                                                                              phase: True,
                                                                              modelphase: True})
                        totndcg = np.asarray([ndcg_score(x, y, k=10) for x, y in zip(rt2d.tolist(), inf2d.tolist())])
                        _, lscost = sess.run([train_Loss, loss_cost], feed_dict={user_batch: users,
                                                                                 item_batch: items,
                                                                                 rate_batch: rates,
                                                                                 ndcgTargets: totndcg.reshape((-1, 1)),
                                                                                 phase: True,
                                                                                 modelphase: True})

                        losscost.append(lscost)
                        truecost.append(totndcg)
                    else:
                        _, pred_batch, logcst = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                                                                                             item_batch: items,
                                                                                             rate_batch: rates,
                                                                                             phase: True})
                        logcost.append(logcst)
                        inf2d, rt2d = sess.run([Infere2d, rate2d], feed_dict={user_batch: users,
                                                                              item_batch: items,
                                                                              rate_batch: rates,
                                                                              phase: True,
                                                                              modelphase: True})

                        totndcg = np.asarray([ndcg_score(x, y, k=10) for x, y in zip(rt2d.tolist(), inf2d.tolist())])
                        _, lscost = sess.run([train_Loss, loss_cost], feed_dict={user_batch: users,
                                                                                 item_batch: items,
                                                                                 rate_batch: rates,
                                                                                 ndcgTargets: totndcg.reshape((-1, 1)),
                                                                                 phase: True,
                                                                                 modelphase: True})

                        losscost.append(lscost)
                        truecost.append(totndcg)
                        #############################train_NDCG,
                        _, pred_batch, cst = sess.run([train_NDCG, infer, NDCGScore_with_recsys],
                                                      feed_dict={user_batch: users,
                                                                 item_batch: items,
                                                                 rate_batch: rates,
                                                                 phase: True,
                                                                 modelphase: False})
                        errors.append(cst)

                else:
                    ################################
                    _, pred_batch, cst = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                                                                                      item_batch: items,
                                                                                      rate_batch: rates,
                                                                                      phase: True})
                    losscost.append(0)
                    errors.append(0)
                    truecost.append(0)
                    logcost.append(0)
                    pred_batch = clip(pred_batch)
                ################################

                if i % samples_per_batch == 0:

                    train_err = np.mean(errors)
                    train_err1 = np.mean(truecost)
                    train_err2 = np.mean(losscost)
                    train_err3 = np.mean(logcost)

                    totalhits = 0
                    ############
                    correcthits10 = 0
                    correcthits20 = 0
                    correcthits5 = 0
                    ndcg10 = 0
                    ndcg20 = 0
                    ndcg5 = 0
                    ndcg10_50 = 0
                    ndcg10_90 = 0
                    ndcg10_80 = 0
                    ndcg10_95 = 0
                    ###########
                    for userid in grouped_instances:
                        # items = dictUsers[userid][3]
                        # TestItem = dictUsers[userid][2][0]
                        role = "validation"
                        TestItem = grouped_instances[userid][role + "_item"]
                        # TestUser = userid
                        users = np.repeat(userid, 100)
                        # items = dictUsers[userid][3]
                        items = grouped_instances[userid][role + "_with_negative_sample"]
                        pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                                item_batch: items,
                                                                phase: False})

                        sorteditems = [x for _, x in
                                       sorted(zip(pred_batch, items), key=lambda pair: pair[0], reverse=True)]
                        #######
                        topitems10 = sorteditems[:5]
                        correcthits10 = correcthits10 + getHR(sorteditems[:10], TestItem)
                        correcthits20 = correcthits20 + getHR(sorteditems[:20], TestItem)
                        correcthits5 = correcthits5 + getHR(sorteditems[:5], TestItem)

                        ndcg10 = ndcg10 + getNDCG(sorteditems[:10], TestItem)
                        ndcg20 = ndcg20 + getNDCG(sorteditems[:20], TestItem)
                        ndcg5 = ndcg5 + getNDCG(sorteditems[:5], TestItem)

                        ndcg10_50 = ndcg10_50  # +getNDCG_SIM(topitems10,TestItem,0.5)
                        ndcg10_80 = ndcg10_80  # +getNDCG_SIM(topitems10,TestItem,0.8)
                        ndcg10_90 = ndcg10_90  # +getNDCG_SIM(topitems10,TestItem,0.9)
                        ndcg10_95 = ndcg10_95  # +getNDCG_SIM(topitems10,TestItem,0.95)

                        totalhits = totalhits + 1
                        ############
                    HitRatio10 = correcthits10 / totalhits
                    HitRatio20 = correcthits20 / totalhits
                    HitRatio5 = correcthits5 / totalhits
                    NDCG10 = ndcg10 / totalhits
                    NDCG20 = ndcg20 / totalhits
                    NDCG5 = ndcg5 / totalhits

                    NDCG_50 = ndcg10_50 / totalhits
                    NDCG_80 = ndcg10_80 / totalhits
                    NDCG_90 = ndcg10_90 / totalhits
                    NDCG_95 = ndcg10_95 / totalhits
                    print(
                        "{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch,
                                                                                         train_err,
                                                                                         train_err1, train_err2,
                                                                                         train_err3,
                                                                                         HitRatio5, HitRatio10,
                                                                                         HitRatio20,
                                                                                         NDCG5, NDCG10, NDCG20))
                    textTrain_file.write(
                        "{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch,
                                                                                         train_err,
                                                                                         train_err1, train_err2,
                                                                                         train_err3,
                                                                                         HitRatio5, HitRatio10,
                                                                                         HitRatio20,
                                                                                         NDCG5, NDCG10, NDCG20) + '\n')
                    textTrain_file.flush()

                    if NDCG10 > bestndcg10:
                        bestndcg10 = NDCG10

                        train_err = np.mean(errors)
                        train_err1 = np.mean(truecost)
                        train_err2 = np.mean(losscost)
                        train_err3 = np.mean(logcost)

                        totalhits = 0
                        ############
                        correcthits10 = 0
                        correcthits20 = 0
                        correcthits5 = 0
                        ndcg10 = 0
                        ndcg20 = 0
                        ndcg5 = 0
                        ndcg10_50 = 0
                        ndcg10_90 = 0
                        ndcg10_80 = 0
                        ndcg10_95 = 0
                        ###########
                        matrix_measures = []
                        pred_batches = []
                        for userid in grouped_instances:
                            # items = dictUsers[userid][3]
                            # TestItem = dictUsers[userid][2][0]
                            role = "test"
                            TestItem = grouped_instances[userid][role + "_item"]
                            # TestUser = userid
                            users = np.repeat(userid, 100)
                            # items = dictUsers[userid][3]
                            items = grouped_instances[userid][role + "_with_negative_sample"]
                            pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                                    item_batch: items,
                                                                    phase: False})
                            pred_batches.append(pred_batch)

                            sorteditems = [x for _, x in
                                           sorted(zip(pred_batch, items), key=lambda pair: pair[0], reverse=True)]
                            #######
                            topitems10 = sorteditems[:5]
                            correcthits10 = correcthits10 + getHR(sorteditems[:10], TestItem)
                            correcthits20 = correcthits20 + getHR(sorteditems[:20], TestItem)
                            correcthits5 = correcthits5 + getHR(sorteditems[:5], TestItem)

                            ndcg10 = ndcg10 + getNDCG(sorteditems[:10], TestItem)
                            ndcg20 = ndcg20 + getNDCG(sorteditems[:20], TestItem)
                            ndcg5 = ndcg5 + getNDCG(sorteditems[:5], TestItem)

                            ndcg10_50 = ndcg10_50  # +getNDCG_SIM(topitems10,TestItem,0.5)
                            ndcg10_80 = ndcg10_80  # +getNDCG_SIM(topitems10,TestItem,0.8)
                            ndcg10_90 = ndcg10_90  # +getNDCG_SIM(topitems10,TestItem,0.9)
                            ndcg10_95 = ndcg10_95  # +getNDCG_SIM(topitems10,TestItem,0.95)

                            matrix_measures.append([getHR(sorteditems[:5], TestItem),
                                                    getHR(sorteditems[:10], TestItem),
                                                    getHR(sorteditems[:20], TestItem),
                                                    getNDCG(sorteditems[:5], TestItem),
                                                    getNDCG(sorteditems[:10], TestItem),
                                                    getNDCG(sorteditems[:20], TestItem)
                                                    ])

                            totalhits = totalhits + 1
                            ############
                        HitRatio10 = correcthits10 / totalhits
                        HitRatio20 = correcthits20 / totalhits
                        HitRatio5 = correcthits5 / totalhits
                        NDCG10 = ndcg10 / totalhits
                        NDCG20 = ndcg20 / totalhits
                        NDCG5 = ndcg5 / totalhits

                        NDCG_50 = ndcg10_50 / totalhits
                        NDCG_80 = ndcg10_80 / totalhits
                        NDCG_90 = ndcg10_90 / totalhits
                        NDCG_95 = ndcg10_95 / totalhits
                        print(
                            "NEWBEST:{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(
                                i // samples_per_batch,
                                train_err,
                                train_err1, train_err2,
                                train_err3,
                                HitRatio5, HitRatio10,
                                HitRatio20,
                                NDCG5, NDCG10, NDCG20))
                        with open("./Output/" + name_exec + "/" + "measures.csv", "w+") as fo:
                            for line in matrix_measures:
                                fo.write(str(line) + "\n")
                        with open("./Output/" + name_exec + "/" + "predictions.txt", "w+") as pre_out:
                            for ii_pred in pred_batches:
                                for ii_pred2 in ii_pred:
                                    pre_out.write(str(ii_pred2) + "\n")

            textTrain_file.close()
        # degreelist, predlist = zip(*sorted(zip(degreelist, predlist)))

        return

    def GetTrainSample(grouped_instances, BatchSize=1000, topn=10):
        trainusers = list()
        trainitems = list()
        traintargets = list()
        numusers = int(BatchSize / (topn))

        for i in range(numusers):
            batchusers = random.randint(0, USER_NUM - 1)  ####
            # while len(DictUsers[batchusers][0]) == 0:
            while len(grouped_instances[batchusers]["train_items_with_interaction"]) == 0:
                # batchusers = random.choice(list(DictUsers.keys()))  ####
                batchusers = random.choice(list(grouped_instances.keys()))
            trainusers.extend(np.repeat(batchusers, topn))
            ##Pos

            # trainitems.extend(np.random.choice(DictUsers[batchusers][0], int(5), replace=True))
            trainitems.extend(
                np.random.choice(grouped_instances[batchusers]["train_items_with_interaction"], int(5), replace=True))
            traintargets.extend(np.ones(int(5)))
            ##Neg
            # trainitems.extend(np.random.choice(DictUsers[batchusers][1], int(5), replace=True))
            trainitems.extend(
                np.random.choice(grouped_instances[batchusers]["items_without_interaction"], int(5), replace=True))
            traintargets.extend(np.zeros(int(5)))

        trainusers = np.asarray(trainusers)
        trainitems = np.asarray(trainitems)
        traintargets = np.asarray(traintargets)
        return trainusers, trainitems, traintargets

    # ITEMDATA = get_ItemData100k()
    ITEMDATA = None

    # dictUsers = load_data("../Data/ml100k/UserDict.dat")
    # df_train = load_data("../Data/ml100k/RankData.dat")

    overall_infos = load_data(f"./../prepared_data/{dataset}/dataset_infos.dat")
    USER_NUM = overall_infos["USER_NUM"]
    ITEM_NUM = overall_infos["ITEM_NUM"]
    INTERACTIONS_TRAIN_NUM = overall_infos["INTERACTIONS_TRAIN_NUM"]
    grouped_instances = load_data(f"./../prepared_data/{dataset}/dict_data_preparation.dat")

    warnings.filterwarnings('ignore')
    GuidedRec(grouped_instances, ItemData=False, UserData=False, Graph=False, lr=0.001, ureg=0.0, ireg=0.0)
    tf.reset_default_graph()

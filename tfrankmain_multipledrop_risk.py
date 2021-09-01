import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

logging.getLogger('tensorflow').disabled = True
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)

tf.disable_v2_behavior()

import numpy as np
import pandas as pd
import time
from collections import deque
import warnings
from six import next
from tensorflow.core.framework import summary_pb2
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
import random
import pickle
import math
import tensorflow_addons as tfa
from sklearn import preprocessing
import tensorflow_ranking as tfr
import datetime
from tensorflow_ranking.python.losses_impl import neural_sort
tf.reset_default_graph()

def NeuralSortCrossEntropyLossLocal(labels, logits, temperature=1.0):
    def is_label_valid(labels):
        """Returns a boolean `Tensor` for label validity."""
        labels = tf.convert_to_tensor(value=labels)
        return tf.greater_equal(labels, 0.)

    temperature = temperature
    is_valid = is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(is_valid, labels, -1e3 * tf.ones_like(labels))

    # shape = [batch_size, list_size, list_size].
    true_perm = neural_sort(labels, temperature=temperature)
    smooth_perm = neural_sort(logits, temperature=temperature)
    losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=true_perm, logits=tf.math.log(1e-20 + smooth_perm), axis=2)
    # shape = [batch_size, list_size].
    losses = tf.reduce_mean(input_tensor=losses, axis=-1, keepdims=True)

    return losses, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])

BATCH_SIZE = 250
NEGSAMPLES = 1
USER_NUM = 943
ITEM_NUM = 1682
DIM = 50
EPOCH_MAX = 500
# DEVICE = "/gpu:0"
DEVICE = "/cpu:0"
PERC = 0.9
# SURROGATE = int(sys.argv[1])
SURROGATE = 0
# SEED = int(sys.argv[2])
SEED = 45
# LOSSFUN = sys.argv[3]
LOSSFUN = "neural_sort_cross_entropy_loss"

print('GraphTFRank', "-", SURROGATE, "-", SEED, "-", LOSSFUN)

np.random.seed(SEED)

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


def inferenceDense(phase, ufsize, ifsize, user_batch, item_batch, time_batch, idx_user, idx_item, ureg, ireg, user_num,
                   item_num, dim=5, UReg=0.0, IReg=0.0, device="/cpu:0"):
    with tf.device(device):
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")

        ul1mf = tf.layers.dense(inputs=user_batch, units=20, name='ul1mf', activation=tf.nn.crelu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        drops_u = []
        for i in range(3):
            drops_u.append(tf.layers.dropout(ul1mf, rate=0.1, training=phase))
        il1mf = tf.layers.dense(inputs=item_batch, units=20, name='il1mf', activation=tf.nn.crelu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        drops_i = []
        for i in range(3):
            drops_i.append(tf.layers.dropout(il1mf, rate=0.1, training=phase))

        # InferInputMF = tf.multiply(ul1mf, il1mf)
        # infer = tf.reduce_sum(InferInputMF, 1, name="inference")
        infers = []
        for i in range(3):
            InferInputMF = tf.multiply(drops_u[i], drops_i[i])
            infers.append(tf.reduce_sum(InferInputMF, 1, name="inference"))

        regularizer = tf.reduce_sum(
            tf.add(UReg * tf.nn.l2_loss(ul1mf), IReg * tf.nn.l2_loss(il1mf), name="regularizer"))
    # return infer, infer, infer, infer, regularizer
    return infers, regularizer


def optNDCG(NDCGscore, varlist, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        costNDCG = -NDCGscore
        train_NDCG = tf.train.AdamOptimizer(0.00001).minimize(costNDCG)
    return train_NDCG


def optLoss(NDCGscore, NDCGTrue, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        costLOSS = tf.nn.l2_loss(tf.subtract(NDCGscore, NDCGTrue))
        train_LOSS = tf.train.AdamOptimizer(0.000015).minimize(costLOSS)
    return train_LOSS, costLOSS


def optimization(infer, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=rate_batch,
                                                       logits=infer)  # tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return cost, train_op


def optimizationRank(infers, regularizer, rate_batch, learning_rate=0.00003, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        losfun = tfr.losses.make_loss_fn(LOSSFUN)
        rate_batch = tf.dtypes.cast(rate_batch, tf.float32)
        regularizer = tf.dtypes.cast(regularizer, tf.float32)
        cost = regularizer

        mat = []
        for infer in infers:
            cast_infer = tf.dtypes.cast(infer, tf.float32)
            # cost_drop = losfun(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer, [25, 10]), None)
            cost_drop, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer, [25, 10]))
            cost_drop = cost_drop / tf.reduce_max(cost_drop)
            mat.append(cost_drop)
        cc, _ = NeuralSortCrossEntropyLossLocal(tf.reshape(rate_batch, [25, 10]), tf.reshape(rate_batch, [25, 10]))
        cc = cc / tf.reduce_max(cc)###TODO trocar por tudo 1
        mat.append(cc)

        mat = tf.squeeze(mat)
        mat = tf.transpose(mat)

        def zRisk(mat, alpha, i=0):
            alpha_tensor = tf.dtypes.cast(alpha, tf.float32)
            si = tf.reduce_sum(mat[:, i])
            tj = tf.reduce_sum(mat, axis=1)
            n = tf.reduce_sum(tj)
            xij_eij = mat[:, i] - si * (tj / n)
            subden = si * (tj / n)
            den = tf.math.sqrt(subden + 1e-10)
            u = tf.dtypes.cast((den == 0), tf.float32) * tf.dtypes.cast(9e10, tf.float32)
            den = u + den
            div = xij_eij / den
            less0 = (mat[:, i] - si * (tj / n)) / (den) < 0
            less0 = alpha_tensor * tf.dtypes.cast(less0, tf.float32)
            z_risk = div * less0 + div
            z_risk = tf.reduce_sum(z_risk)
            return z_risk

        def geoRisk(mat, alpha, i=0):
            mat = mat * tf.dtypes.cast((mat > 0), tf.float32)
            si = tf.reduce_sum(mat[:, i])
            z_risk = zRisk(mat, alpha, i=i)
            num_queries = tf.cast(mat.shape[0], tf.float32)
            value = z_risk / num_queries
            m = tf.distributions.Normal(0.0, 1.0)
            ncd = m.cdf(value)
            # return tf.math.sqrt((si / num_queries) * ncd + 1e-10)
            return (si / num_queries) * ncd

        cost = tf.add(geoRisk(mat, 2), cost)

        # cost = losfun(tf.reshape(rate_batch, [25, 10]), tf.reshape(infer, [25, 10]), None)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def get_UserData100k():
    col_names = ["user", "age", "gender", "occupation", "PostCode"]
    df = pd.read_csv('src/Data/u.user', sep='|', header=None, names=col_names, engine='python')
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
    df = pd.read_csv('src/Data/u.item', sep='|', header=None, names=col_names, engine='python')
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
    # print(ranklist)
    NewTestItem = -1
    ItemFeatRow = ITEMDATA[np.asarray([gtItem], dtype=np.int32), :]
    RankedFeatRows = ITEMDATA[np.asarray(ranklist, dtype=np.int32), :]

    simvals = cosine_similarity(RankedFeatRows, ItemFeatRow)
    simvals = (simvals + 1) / 2
    # print(simvals)
    simvals[simvals < Sim] = 0.0

    # print(simvals.shape)
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


def Main(train, ItemData=False, UserData=False, Graph=True, lr=0.00003, ureg=0.0, ireg=0.0):
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
        print(np.max(DegreeUsersVec))
        print(np.min(DegreeUsersVec))
        print(np.max(DegreeItemsVec))
        print(np.min(DegreeItemsVec))

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

    print(UserFeatures.shape)
    print(ItemFeatures.shape)
    if (UserData):
        UsrDat = get_UserData100k()
        UserFeatures = np.concatenate((UserFeatures, UsrDat), axis=1)

    if (ItemData):
        ItmDat = get_ItemData100k()
        ItemFeatures = np.concatenate((ItemFeatures, ItmDat), axis=1)

    print(UserFeatures.shape)
    print(ItemFeatures.shape)

    print("Finish")
    samples_per_batch = len(train) // int(BATCH_SIZE / (NEGSAMPLES + 1))

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

    infers, regularizer = inferenceDense(phase, UserFeatures.shape[1], ItemFeatures.shape[1],
                                         user_batch, item_batch, time_batch, w_user, w_item,
                                         ureg, ireg, user_num=USER_NUM, item_num=ITEM_NUM,
                                         dim=DIM,
                                         device=DEVICE)

    cost, train_op = optimizationRank(infers, regularizer, rate_batch, learning_rate=lr, reg=0.09, device=DEVICE)

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
        # summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("epoch, train_err,train_err1,train_err2,HitRatio5,HitRatio10,HitRatio20,NDCG5,NDCG10,NDCG20")
        now = datetime.datetime.now()
        textTrain_file = open(
            "./Output/" + now.strftime('%Y%m%d%H%M%S') + '_GraphTFRank' + "_" + str(SURROGATE) + "_" + str(
                SEED) + "_" + LOSSFUN + ".txt", "w", newline='')
        errors = deque(maxlen=samples_per_batch)
        losscost = deque(maxlen=samples_per_batch)
        truecost = deque(maxlen=samples_per_batch)
        start = time.time()
        print(samples_per_batch)
        for i in range(EPOCH_MAX * samples_per_batch):
            # users, items, rates,y,m,d,dw,dy,w = next(iter_train)
            users, items, rates = GetTrainSample(dictUsers, BATCH_SIZE, 10)
            ################################

            if True:
                ################################
                runner_nodes = [train_op]
                for infer in infers:
                    runner_nodes.append(infer)
                runner_nodes.append(cost)

                # _, pred_batch, cst = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                rtn = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   phase: True})
                pred_batch = rtn[1]
                cst = rtn[-1]
                losscost.append(0)
                errors.append(0)
                truecost.append(0)
                pred_batch = clip(pred_batch)
            ################################

            # print(i,' ',samples_per_batch)
            if i % samples_per_batch == 0:
                # print('-----')

                train_err = np.mean(errors)
                train_err1 = np.mean(truecost)
                train_err2 = np.mean(losscost)
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
                for userid in dictUsers:
                    items = dictUsers[userid][3]
                    TestItem = dictUsers[userid][2][0]
                    TestUser = userid
                    users = np.repeat(userid, 100)
                    items = dictUsers[userid][3]
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,
                                                            phase: False})

                    sorteditems = [x for _, x in sorted(zip(pred_batch, items), key=lambda pair: pair[0], reverse=True)]
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
                print("{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,
                                                                                  train_err1, train_err2, HitRatio5,
                                                                                  HitRatio10, HitRatio20, NDCG5, NDCG10,
                                                                                  NDCG20))
                textTrain_file.write(
                    "{:3d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(i // samples_per_batch, train_err,
                                                                                train_err1, train_err2, HitRatio5,
                                                                                HitRatio10, HitRatio20, NDCG5, NDCG10,
                                                                                NDCG20) + '\n')
                textTrain_file.flush()

        textTrain_file.close()
    # TODO
    # degreelist, predlist = zip(*sorted(zip(degreelist, predlist)))
    return


def GenNDCGSamples():
    # print(ndcg_score([1, 0, 1,0], [0.9, 0.6, 0.7,0.65],k=5))

    yhats = np.random.rand(100, 10)
    trues = np.zeros((100, 10))

    for row in trues:
        row[np.random.choice(range(10), 5, replace=False)] = 1

    ndcgs = 0
    totndcg = np.asarray([ndcg_score(x, y) for x, y in zip(trues.tolist(), yhats.tolist())])
    comp = np.asarray([[[l, m] for l, m in zip(x, y)] for x, y in zip(trues.tolist(), yhats.tolist())])

    return trues, yhats, comp, totndcg.reshape((-1, 1))


def GetTrainSampleold(DictUsers, BatchSize=1000, topn=10):
    trainusers = list()
    trainitems = list()
    traintargets = list()
    numusers = int(BatchSize / (topn))
    # print(numusers)
    for i in range(numusers):
        batchusers = random.randint(0, USER_NUM - 1)
        while len(DictUsers[batchusers][0]) == 0:
            batchusers = random.choice(list(DictUsers.keys()))
        trainusers.extend(np.repeat(batchusers, topn))
        ##Pos

        trainitems.extend(np.random.choice(DictUsers[batchusers][0], int(topn / 2), replace=True))
        traintargets.extend(np.ones(int(topn / 2)))
        ##Neg
        trainitems.extend(np.random.choice(DictUsers[batchusers][1], int(topn / 2), replace=True))
        traintargets.extend(np.zeros(int(topn / 2)))

    trainusers = np.asarray(trainusers)
    trainitems = np.asarray(trainitems)
    traintargets = np.asarray(traintargets)
    return trainusers, trainitems, traintargets


def GetTrainSample(DictUsers, BatchSize=1000, topn=10):
    trainusers = list()
    trainitems = list()
    traintargets = list()
    numusers = int(BatchSize / (topn))
    # print(numusers)
    for i in range(numusers):
        batchusers = random.randint(0, USER_NUM - 1)
        while len(DictUsers[batchusers][0]) == 0:
            batchusers = random.choice(list(DictUsers.keys()))
        trainusers.extend(np.repeat(batchusers, topn))
        ##Pos

        trainitems.extend(np.random.choice(DictUsers[batchusers][0], int(5), replace=True))
        traintargets.extend(np.ones(int(5)))
        ##Neg
        trainitems.extend(np.random.choice(DictUsers[batchusers][1], int(5), replace=True))
        traintargets.extend(np.zeros(int(5)))

    trainusers = np.asarray(trainusers)
    trainitems = np.asarray(trainitems)
    traintargets = np.asarray(traintargets)
    return trainusers, trainitems, traintargets


ITEMDATA = get_ItemData100k()

dictUsers = load_data("src/Data/UserDict.dat")
df_train = load_data("src/Data/RankData.dat")
print(len(df_train))
print(df_train.shape)
warnings.filterwarnings('ignore')
Main(df_train, ItemData=False, UserData=False, Graph=False, lr=0.001, ureg=0.0, ireg=0.0)
tf.reset_default_graph()

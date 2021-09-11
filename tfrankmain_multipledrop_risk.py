import os
import logging
import warnings

from process_data import load_data

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True

import time
import random
import math
import datetime
import numpy as np
import pandas as pd
from collections import deque
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow.compat.v1 as tf
from tensorflow.core.framework import summary_pb2
import tensorflow_ranking as tfr
from tfr_losses_local import *
from risk_losses import geoRisk

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
tf.reset_default_graph()


def LocalEval(list_of_args):
    num_baseline_dropouts = list_of_args[0]
    local_losfun = list_of_args[1]
    add_l2_reg_on_risk = list_of_args[2]
    add_loss_on_risk = list_of_args[3]
    alpha_risk = list_of_args[4]
    do_diff_to_ideal_risk = list_of_args[5]
    eval_ideal_risk = list_of_args[6]
    dataset = list_of_args[7]
    LR = list_of_args[8]
    LOSSFUN = list_of_args[9]
    drop_rate = list_of_args[10]

    id = list_of_args[11]

    # TODO arrumar parametros deles
    BATCH_SIZE = 250
    NEGSAMPLES = 1
    DIM = 50  # TODO aqui está 50 mas lá está prefixado o 20
    EPOCH_MAX = 450
    # DEVICE = "/gpu:0"
    DEVICE = "/cpu:0"

    SEED = 45
    # LOSSFUN = "neural_sort_cross_entropy_loss"
    # LOSSFUN = ""
    ####
    # num_baseline_dropouts = 3
    # local_losfun = "PairwiseLogisticLossLocal"
    # add_l2_reg_on_risk = True
    # add_loss_on_risk = True
    # alpha_risk = 2
    # do_diff_to_ideal_risk = True
    # eval_ideal_risk = True
    # dataset = "ml100k"
    ####
    np.random.seed(SEED)

    # ITEMDATA = get_ItemData100k()
    ITEMDATA = None

    # dictUsers = load_data(data_path + "UserDict.dat")
    grouped_instances = load_data(f"prepared_data/{dataset}/dict_data_preparation.dat")
    # df_train = load_data(data_path + "RankData.dat")
    df_train = None
    overall_infos = load_data(f"prepared_data/{dataset}/dataset_infos.dat")
    USER_NUM = overall_infos["USER_NUM"]
    ITEM_NUM = overall_infos["ITEM_NUM"]
    INTERACTIONS_TRAIN_NUM = overall_infos["INTERACTIONS_TRAIN_NUM"]
    # USER_NUM = 943
    # ITEM_NUM = 1682

    parameters = [BATCH_SIZE, NEGSAMPLES, DIM, EPOCH_MAX, SEED, LOSSFUN,
                  num_baseline_dropouts, local_losfun, add_l2_reg_on_risk, add_loss_on_risk, alpha_risk,
                  do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR, drop_rate, id]

    parameters_names = "BATCH_SIZE, NEGSAMPLES, DIM, EPOCH_MAX, SEED, LOSSFUN,\
                          num_baseline_dropouts, local_losfun, add_l2_reg_on_risk, add_loss_on_risk, alpha_risk,\
                          do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR, drop_rate, id"

    output_path = "./Output/{:s}/".format(str(id))
    # data_path = "src/Data/"
    data_path = None
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + "configs.txt", "w") as fo:
        for name, value in zip(parameters_names.replace(" ", "").split(","), parameters):
            fo.write(name + ":" + str(value) + "\n")

    print("Starting")

    if os.name == 'nt':
        os.system("copy \"{}\" \"{}\"".format("tfrankmain_multipledrop_risk.py",
                                              output_path + "tfrankmain_multipledrop_risk.py"))
        os.system("copy \"{}\" \"{}\"".format("tfrankmain_multipledrop_risk_temp.py",
                                              output_path + "tfrankmain_multipledrop_risk_temp.py"))
    else:
        os.system("cp {} {}".format("tfrankmain_multipledrop_risk.py", output_path + "tfrankmain_multipledrop_risk.py"))
        os.system("cp {} {}".format("tfrankmain_multipledrop_risk_temp.py",
                                    output_path + "tfrankmain_multipledrop_risk_temp.py"))

    # Main(df_train, ItemData=False, UserData=False, Graph=False, lr=0.001, ureg=0.0, ireg=0.0)

    def inferenceDense(phase, ufsize, ifsize, user_batch, item_batch, time_batch, idx_user, idx_item, ureg, ireg,
                       user_num,
                       item_num, dim=5, UReg=0.0, IReg=0.0, device="/cpu:0"):
        with tf.device(device):
            user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
            item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")

            ul1mf = tf.layers.dense(inputs=user_batch, units=20, name='ul1mf', activation=tf.nn.crelu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            drops_u = []
            for i in range(num_baseline_dropouts):
                drops_u.append(tf.layers.dropout(ul1mf, rate=drop_rate, training=phase))
            il1mf = tf.layers.dense(inputs=item_batch, units=20, name='il1mf', activation=tf.nn.crelu,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            drops_i = []
            for i in range(num_baseline_dropouts):
                drops_i.append(tf.layers.dropout(il1mf, rate=drop_rate, training=phase))

            infers = []
            for i in range(num_baseline_dropouts):
                InferInputMF = tf.multiply(drops_u[i], drops_i[i])
                infers.append(tf.reduce_sum(InferInputMF, 1, name="inference"))

            regularizer = tf.reduce_sum(
                tf.add(UReg * tf.nn.l2_loss(ul1mf), IReg * tf.nn.l2_loss(il1mf), name="regularizer"))

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

            rate_batch = tf.dtypes.cast(rate_batch, tf.float32)
            regularizer = tf.dtypes.cast(regularizer, tf.float32)
            cost = regularizer

            if local_losfun == "":
                losfun = tfr.losses.make_loss_fn(LOSSFUN)

                for infer in infers:
                    cast_infer = tf.dtypes.cast(infer, tf.float32)
                    cost_drop = losfun(tf.reshape(rate_batch, [25, 10]), tf.reshape(cast_infer, [25, 10]), None)
                    cost = tf.add(cost, cost_drop)
            else:
                mat = []
                for infer in infers:
                    cast_infer = tf.dtypes.cast(infer, tf.float32)
                    cost_drop, _ = globals()[local_losfun](tf.reshape(rate_batch, [25, 10]),
                                                           tf.reshape(cast_infer, [25, 10]))

                    if local_losfun == "GumbelApproxNDCGLossLocal":
                        cost_drop = -cost_drop
                    elif local_losfun == "NeuralSortCrossEntropyLossLocal":
                        cost_drop = cost_drop / tf.reduce_max(cost_drop)

                    # cost_drop = cost_drop / tf.reduce_max(cost_drop)
                    mat.append(cost_drop)

                    if add_l2_reg_on_risk:
                        cost_reg = tf.nn.l2_loss(tf.subtract(cast_infer, rate_batch))
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
        df = pd.read_csv(data_path + 'u.user', sep='|', header=None, names=col_names, engine='python')
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
        df = pd.read_csv(data_path + 'u.item', sep='|', header=None, names=col_names, engine='python')
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

    def getNDCG_SIM(ranklist, gtItem, Sim):  # TODO ndcg com similarity??, seira algo tipo novelty or other
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
                AdjacencyUsers[userid][itemid] = row['rate'] / 5.0  # TODO se usar essa parte tem que corrigir
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
        # samples_per_batch = sum([len(dictUsers[x][0]) for x in dictUsers.keys()]) // int(BATCH_SIZE / (NEGSAMPLES + 1))

        # samples_per_batch = sum(
        #     [len(grouped_instances[x]["train_items_with_interaction"]) for x in grouped_instances.keys()]) // int(
        #     BATCH_SIZE / (NEGSAMPLES + 1))

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

            print("epoch,train_err,train_err1,train_err2,HitRatio5,HitRatio10,HitRatio20,NDCG5,NDCG10,NDCG20,time")
            now = datetime.datetime.now()

            if local_losfun == "":
                textTrain_file = open(
                    output_path + now.strftime('%Y%m%d%H%M%S') + '_GraphTFRank-train' + "_" + "_"
                    + str(SEED) + "_" + LOSSFUN + ".txt", "w", newline='')

                textTest_file = open(
                    output_path + now.strftime('%Y%m%d%H%M%S') + '_GraphTFRank-test' + "_" + "_"
                    + str(SEED) + "_" + LOSSFUN + ".txt", "w", newline='')
            else:
                textTrain_file = open(
                    output_path + now.strftime('%Y%m%d%H%M%S') + '_GraphTFRank-train' + "_" + "_"
                    + str(SEED) + "_" + local_losfun + ".txt", "w", newline='')
                textTest_file = open(
                    output_path + now.strftime('%Y%m%d%H%M%S') + '_GraphTFRank-test' + "_" + "_"
                    + str(SEED) + "_" + local_losfun + ".txt", "w", newline='')

            errors = deque(maxlen=samples_per_batch)
            losscost = deque(maxlen=samples_per_batch)
            truecost = deque(maxlen=samples_per_batch)
            best_measure = 0

            for i in range(EPOCH_MAX * samples_per_batch):
                start = time.time()

                users, items, rates = GetTrainSample(grouped_instances, BATCH_SIZE, 10)

                runner_nodes = [train_op]
                for infer in infers:
                    runner_nodes.append(infer)
                runner_nodes.append(cost)

                # _, pred_batch, cst = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                # rtn = sess.run([train_op, infer, cost], feed_dict={user_batch: users,
                rtn = sess.run(runner_nodes, feed_dict={user_batch: users,
                                                        item_batch: items,
                                                        rate_batch: rates,
                                                        phase: True})
                pred_batch = rtn[1]
                cst = rtn[-1]
                losscost.append(cst)
                errors.append(0)
                truecost.append(0)
                pred_batch = clip(pred_batch)
                end = time.time()
                elapsed_epoch_time = end - start
                if i % samples_per_batch == 0:
                    train_err = np.mean(errors)
                    train_err1 = np.mean(truecost)
                    train_err2 = np.mean(losscost)

                    measures_means, measures_names = call_evaluation_measures(sess, infer, user_batch, item_batch,
                                                                              phase,
                                                                              role="validation")

                    log_str = ""
                    for m in measures_means:
                        log_str += "{:f},".format(m)

                    log_line = "{:3d},{:f},{:f},{:f},{:s}{:f}".format(i // samples_per_batch,
                                                                      train_err,
                                                                      train_err1, train_err2,
                                                                      log_str,
                                                                      elapsed_epoch_time).replace(" ", "")

                    print(log_line)
                    textTrain_file.write(log_line + '\n')
                    textTrain_file.flush()

                    NDCG10 = measures_means[4]
                    if NDCG10 > best_measure:
                        best_measure = NDCG10
                        measures_means, measures_names = call_evaluation_measures(sess, infer, user_batch, item_batch,
                                                                                  phase, role="test", wout=True)

                        log_str = ""
                        for m in measures_means:
                            log_str += "{:f},".format(m)

                        log_line = "{:3d},{:f},{:f},{:f},{:s}{:f}".format(i // samples_per_batch,
                                                                          train_err,
                                                                          train_err1, train_err2,
                                                                          log_str,
                                                                          elapsed_epoch_time).replace(" ", "")

                        print("NEW BEST:" + log_line)
                        textTest_file.write(
                            "New best at iteration " + str(i // samples_per_batch) + ": " + str(NDCG10) + '\n')
                        textTest_file.write(log_line + '\n')
                        textTest_file.flush()

            textTrain_file.close()
            textTest_file.close()

        # TODO
        # degreelist, predlist = zip(*sorted(zip(degreelist, predlist)))

        return

    def GenNDCGSamples():
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

    def call_evaluation_measures(sess, infer, user_batch, item_batch, phase, role="validation", wout=False):
        measures_matrix = []
        metrics = [getHR, getNDCG]
        list_of_k = [5, 10, 20]

        measures_evalauted_names = []
        for metric in metrics:
            for k in list_of_k:
                measures_evalauted_names.append(str(metric) + str(k))

        for userid in grouped_instances:
            ListOfItemsToRank = grouped_instances[userid][role + "_item"]
            users = np.repeat(userid, 100)
            items = grouped_instances[userid][role + "_with_negative_sample"]

            pred_batch = sess.run(infer, feed_dict={user_batch: users, item_batch: items, phase: False})

            sorteditems = [x for _, x in sorted(zip(pred_batch, items), key=lambda pair: pair[0], reverse=True)]

            line_of_measures = []

            for metric in metrics:
                for k in list_of_k:
                    # metric_for_user_at_k = globals()[metric](sorteditems[:k], ListOfItemsToRank)
                    metric_for_user_at_k = metric(sorteditems[:k], ListOfItemsToRank)
                    line_of_measures.append(metric_for_user_at_k)

            measures_matrix.append(line_of_measures)

        if wout:
            pd.DataFrame(measures_matrix, columns=measures_evalauted_names).to_csv(output_path + "measures_test.txt",
                                                                                   index=False)

        mean_measures = np.mean(measures_matrix, axis=0)
        # return HitRatio5, HitRatio10, HitRatio20, NDCG5, NDCG10, NDCG20
        return mean_measures, measures_evalauted_names

    Main(df_train, ItemData=False, UserData=False, Graph=False, lr=LR, ureg=0.0, ireg=0.0)
    tf.reset_default_graph()

    with open(output_path + "configs-fim.txt", "w") as fo:
        for name, value in zip(parameters_names.replace(" ", "").split(","), parameters):
            fo.write(name + ":" + str(value) + "\n")

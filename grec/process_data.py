import numpy as np
import pandas as pd
import pickle
import os
import time


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


def data_preparation(ratings_file, ouput_path):
    all_ratings = pd.read_csv(ratings_file, sep="\t", names=["user", "item", "rating", "timestamp"])

    all_ratings = all_ratings.sort_values(["user", "timestamp"], ascending=True)
    test_and_vali = all_ratings.groupby("user").tail(2)

    train = all_ratings[~all_ratings.index.isin(test_and_vali.index)]
    test = test_and_vali.groupby("user").tail(1)
    validation = test_and_vali[~test_and_vali.index.isin(test.index)]

    user_ids = all_ratings["user"].unique()  # already ordered in pandas
    item_ids = np.sort(all_ratings["item"].unique())
    USER_NUM = len(user_ids)
    ITEM_NUM = len(item_ids)

    range_of_items_ids = np.arange(ITEM_NUM)
    id_index_items_map = {}
    for i in range(ITEM_NUM):
        id_index_items_map.setdefault(item_ids[i], i)

    def getIndexById(id):
        return id_index_items_map[id]

    dict_data_preparation = {}

    for i in range(USER_NUM):
        train_interactions = train.query("user == " + str(user_ids[i])).to_numpy()

        train_items_with_interaction = list(map(getIndexById, train_interactions[:, 1]))
        items_without_interaction = np.delete(range_of_items_ids, train_items_with_interaction)

        validation_interaction = validation.query("user == " + str(user_ids[i])).to_numpy()
        validation_item = getIndexById(validation_interaction[0, 1])
        rate_validation_item = validation_interaction[0, 2]
        validation_with_negative_sample = np.append(np.random.choice(items_without_interaction, 99), validation_item)

        test_interaction = test.query("user == " + str(user_ids[i])).to_numpy()
        test_item = getIndexById(test_interaction[0, 1])
        rate_test_item = test_interaction[0, 2]
        test_with_negative_sample = np.append(np.random.choice(items_without_interaction, 99), test_item)

        dict_data_preparation.setdefault(i, {
            "original_user_id": user_ids[i],

            "train_items_with_interaction": train_items_with_interaction,
            "rates_train_items_with_interaction": train_interactions[:, 2],
            "items_without_interaction": items_without_interaction,

            "validation_item": validation_item,
            "rate_validation_item": rate_validation_item,
            "validation_with_negative_sample": validation_with_negative_sample,

            "test_item": test_item,
            "rate_test_item": rate_test_item,
            "test_with_negative_sample": test_with_negative_sample,
        })

    overall_infos = {
        "USER_NUM": USER_NUM,
        "ITEM_NUM": ITEM_NUM,
        "INTERACTIONS_TRAIN_NUM": train.shape[0]
    }

    save_data(dict_data_preparation, ouput_path + "dict_data_preparation.dat")
    save_data(id_index_items_map, ouput_path + "id_index_items_map.dat")
    save_data(overall_infos, ouput_path + "dataset_infos.dat")


if __name__ == '__main__':
    print("Starting")
    for dataset in ['ml100k', 'ml1m']:
        outdir = f'prepared_data/{dataset}/'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        s = time.time()
        data_preparation(f'src/Data/{dataset}/u.data', outdir)
        e = time.time()
        print("elapsed time: %.3fs" % (e - s))

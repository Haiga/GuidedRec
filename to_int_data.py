import numpy as np
import pandas as pd
import pickle
import os
import time




def data_preparation(ratings_file, output_file, sep="\t"):
    cont_user = 1
    cont_item = 1

    users_dict = {}
    items_dict = {}
    with open(ratings_file) as fi:
        with open(output_file, "w+") as fo:
            for line in fi:
                line_splitted = line.strip().split(sep)
                user_id = line_splitted[0]
                item_id = line_splitted[1]
                rating = line_splitted[2]
                timestamp = line_splitted[3]

                user_new_id = ""
                item_new_id = ""

                if user_id in users_dict:
                    user_new_id = users_dict[user_id]
                else:
                    users_dict.setdefault(user_id, cont_user)
                    user_new_id = cont_user
                    cont_user += 1

                if item_id in items_dict:
                    item_new_id = items_dict[item_id]
                else:
                    items_dict.setdefault(item_id, cont_item)
                    item_new_id = cont_item
                    cont_item += 1

                fo.write(f"{user_new_id}{sep}{item_new_id}{sep}{rating}{sep}{timestamp}\n")



if __name__ == '__main__':
    print("Starting")
    # for dataset in ['ml100k', 'ml1m']:
    for dataset in ['amazon2']:

        s = time.time()
        # data_preparation(f'src/Data/{dataset}/u.data', outdir)
        data_preparation(f'src/Data/{dataset}/ratings.csv', f'src/Data/{dataset}/u.data', ",")
        e = time.time()
        print("elapsed time: %.3fs" % (e - s))

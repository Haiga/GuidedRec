import os
import pandas as pd
import numpy as np

indexes = []
metrics = []

for i in os.listdir("Output/"):
    if i.isdigit():
        i = int(i)

    if os.path.isfile(f"Output/{i}/measures_test.txt"):
        df = pd.read_csv(f"Output/{i}/measures_test.txt", sep=",")
        df.columns = ["HR5", "HR10", "HR20", "NDCG5", "NDCG10", "NDCG20"]
        indexes.append(i)
        m = np.mean(df["NDCG10"].values)
        metrics.append(m)

order_indexes = np.argsort(metrics)[::-1]

to_read_order = np.array(indexes)[order_indexes]
metrics_order = np.array(metrics)[order_indexes]

with open("Output/results.csv", "w") as fo:
    parameters_names = "BATCH_SIZE, NEGSAMPLES, DIM, EPOCH_MAX, SEED, LOSSFUN,\
                              num_baseline_dropouts, local_losfun, add_l2_reg_on_risk, add_loss_on_risk, alpha_risk,\
                              do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR, id"
    parameters_names = parameters_names.replace(" ", "").replace(",", "\t")
    fo.write(f"NDCG10\t{parameters_names}\n")
    size = len(to_read_order)

    for i in range(size):
        fo.write(f"{metrics_order[i]}\t")

        for line in open(f"Output/{to_read_order[i]}/configs.txt"):
            line_corrected = line.replace("\n", "").split(":")
            if len(line_corrected) < 2:
                line_corrected = ""
            else:
                line_corrected = line_corrected[1]
            fo.write(f"{line_corrected}\t")
        fo.write("\n")

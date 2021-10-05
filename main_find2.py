# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

# from tfrankmain_multipledrop_risk import LocalEval
from tfrankmain_multipledrop_risk_temp import LocalEval

if __name__ == '__main__':
    # # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        id = 66489321
        all_lists = []
        # for LOSSFUN in ["neural_sort_cross_entropy_loss", "gumbel_approx_ndcg_loss", "pairwise_logistic_loss", "list_mle_loss"]:
        for LOSSFUN in ["list_mle_loss"]:
        # for LOSSFUN in [""]:
        # for LOSSFUN in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal","PairwiseLogisticLossLocal"]:
            for num_baseline_dropouts in [1, 3, 5, 10]:#music at 115 (rodou 16) falta os de list_mle_loss
                for local_losfun in [""]:
                    for add_l2_reg_on_risk in [True]:
                        for add_loss_on_risk in [True]:
                            for alpha_risk in [2]:
                                for do_diff_to_ideal_risk in [True]:
                                    for eval_ideal_risk in [False]:
                                        for dataset in ["music"]:
                                            for EMB in [45]:
                                                # for dataset in ["ml100k"]:
                                            # for dataset in ["ml100k"]:
                                                for LR in [0.0005]:
                                                    drop_rate = 0.1
                                                    if num_baseline_dropouts == 1:
                                                        drop_rate = 0

                                                        list_of_args = [num_baseline_dropouts, local_losfun, add_l2_reg_on_risk,
                                                                        add_loss_on_risk,
                                                                        alpha_risk,
                                                                        do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR, LOSSFUN, drop_rate, EMB, id]
                                                        all_lists.append(list_of_args)
                                                        id += 1

                                                    list_of_args = [num_baseline_dropouts, local_losfun, add_l2_reg_on_risk,
                                                                    add_loss_on_risk,
                                                                    alpha_risk,
                                                                    do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR, LOSSFUN, drop_rate, EMB, id]
                                                    all_lists.append(list_of_args)
                                                    id += 1
                                                    # LocalEval(all_lists)
                                                    # if id == 1: return all_lists
        return all_lists


    all_lists = run()
    with multiprocessing.Pool(processes=2) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(LocalEval, all_lists)
        for r in results:
            print(r)


    # def run2():
    #     id = 66489421
    #     all_lists = []
    #     for LOSSFUN in [""]:
    #         # for LOSSFUN in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal","PairwiseLogisticLossLocal"]:
    #         for num_baseline_dropouts in [2, 5]:
    #             for local_losfun in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal"]:
    #                 for add_l2_reg_on_risk in [True]:
    #                     for add_loss_on_risk in [True]:
    #                         for alpha_risk in [2, 5]:
    #                             for do_diff_to_ideal_risk in [True]:
    #                                 for eval_ideal_risk in [False]:
    #                                     for dataset in ["music"]:
    #                                         # for dataset in ["ml100k"]:
    #                                         # for dataset in ["ml100k"]:
    #                                         for LR in [0.001, 0.0005]:
    #                                             drop_rate = 0.1
    #
    #                                             list_of_args = [num_baseline_dropouts, local_losfun, add_l2_reg_on_risk,
    #                                                             add_loss_on_risk,
    #                                                             alpha_risk,
    #                                                             do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR,
    #                                                             LOSSFUN, drop_rate, 45, id]
    #                                             all_lists.append(list_of_args)
    #                                             id += 1
    #                                             # LocalEval(all_lists)
    #                                             # if id == 1: return all_lists
    #     return all_lists
    #
    #
    # all_lists = run2()
    # with multiprocessing.Pool(processes=5) as pool:
    #     # a = executor.submit(LocalEval, list_of_args)
    #     results = pool.map(LocalEval, all_lists)
    #     for r in results:
    #         print(r)



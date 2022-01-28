# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

# from tfrankmain_multipledrop_risk import LocalEval
from tfrankmain_multipledrop_risk_temp import LocalEval

if __name__ == '__main__':


    def run2():
        id = 74000
        all_lists = []
        for LOSSFUN in [""]:
            # for LOSSFUN in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal","PairwiseLogisticLossLocal"]:
            for num_baseline_dropouts in [2, 5]:
                for local_losfun in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal", "PairwiseLogisticLossLocal"]:
                    for add_l2_reg_on_risk in [True]:
                        for add_loss_on_risk in [True]:
                            for alpha_risk in [2, 5]:
                                for do_diff_to_ideal_risk in [True]:
                                    for eval_ideal_risk in [False]:
                                        for dataset in ["aiv"]:
                                            # for dataset in ["ml100k"]:
                                            # for dataset in ["ml100k"]:
                                            for LR in [0.0004, 0.0005]:
                                                drop_rate = 0.05

                                                list_of_args = [num_baseline_dropouts, local_losfun, add_l2_reg_on_risk,
                                                                add_loss_on_risk,
                                                                alpha_risk,
                                                                do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR,
                                                                LOSSFUN, drop_rate, 45, id]
                                                all_lists.append(list_of_args)
                                                id += 1
                                                # LocalEval(all_lists)
                                                # if id == 1: return all_lists
        return all_lists


    all_lists = run2()
    with multiprocessing.Pool(processes=3) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(LocalEval, all_lists)
        for r in results:
            print(r)



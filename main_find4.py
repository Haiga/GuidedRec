# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from tfrankmain_multipledrop_risk_tune import LocalEval

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        id = 995666
        all_lists = []
        LOSSFUN = ""
        for num_baseline_dropouts in [10]:
            for local_losfun in ["GumbelApproxNDCGLossLocal"]:
                for add_l2_reg_on_risk in [True]:
                    for add_loss_on_risk in [False]:
                        for alpha_risk in [2, 5]:
                            for do_diff_to_ideal_risk in [True]:
                                for eval_ideal_risk in [True]:
                                    # for dataset in ["ml100k", "ml1m"]:
                                    # for dataset in ["ml100k"]:
                                    for dataset in ["ml1m"]:
                                        # for LR in [0.01, 0.0001]:
                                        for LR in [0.0005, 0.0004]:
                                            for EMB_SIZE in [30, 45, 60]:
                                                for drop_rate in [0.2, 0.01]:
                                                    for REG in [0.1, 1]:
                                                        list_of_args = [num_baseline_dropouts, local_losfun, add_l2_reg_on_risk,
                                                                        add_loss_on_risk,
                                                                        alpha_risk,
                                                                        do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR,
                                                                        LOSSFUN, drop_rate, EMB_SIZE, REG, id]
                                                        all_lists.append(list_of_args)
                                                        id += 1
                                            # if id == 1: return all_lists
        return all_lists


    all_lists = run()
    with multiprocessing.Pool(processes=2) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(LocalEval, all_lists)
        for r in results:
            print(r)
        # submits.append(a)

        # if id == 4: return

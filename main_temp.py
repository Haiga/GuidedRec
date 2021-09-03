# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

# from tfrankmain_multipledrop_risk import LocalEval
from tfrankmain_multipledrop_risk_temp import LocalEval

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        id = 100
        all_lists = []
        for LOSSFUN in []:
            for num_baseline_dropouts in [1, 3, 5, 10]:
                for local_losfun in [""]:
                    for add_l2_reg_on_risk in [True]:
                        for add_loss_on_risk in [True]:
                            for alpha_risk in [2]:
                                for do_diff_to_ideal_risk in [True]:
                                    for eval_ideal_risk in [False]:
                                        for dataset in ["ml100k", "ml1m"]:
                                            # for dataset in ["ml100k"]:
                                            # for dataset in ["ml1m"]:
                                            for LR in [0.001, 0.0001]:
                                                list_of_args = [num_baseline_dropouts, local_losfun, add_l2_reg_on_risk,
                                                                add_loss_on_risk,
                                                                alpha_risk,
                                                                do_diff_to_ideal_risk, eval_ideal_risk, dataset, LR, LOSSFUN, id]
                                                all_lists.append(list_of_args)
                                                id += 1
                                                # if id == 4: return all_lists
        return all_lists


    all_lists = run()
    with multiprocessing.Pool(processes=2) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(LocalEval, all_lists)
        for r in results:
            print(r)
        # submits.append(a)

        # if id == 4: return

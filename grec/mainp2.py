# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from guided_rec_parametrized import RunGuidedRecParametrized

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        base_args = []
        id = 400
        for num_baseline_dropouts in [2, 5]:
            for alpha_risk in [2, 5]:
                for local_losfun in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal"]:
                    for LR_LOGLOSS in [0.01, 0.005]:
                        list_of_args = [0.1,
                                        num_baseline_dropouts,
                                        local_losfun,
                                        True,
                                        "",
                                        True,
                                        True,
                                        True,
                                        False,
                                        alpha_risk,
                                        str(id),
                                        LR_LOGLOSS, id]
                        id += 1
                        base_args.append(list_of_args)

        return base_args


    all_lists = run()
    with multiprocessing.Pool(processes=2) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(RunGuidedRecParametrized, all_lists)
        for r in results:
            print(r)
        # submits.append(a)

        # if id == 4: return

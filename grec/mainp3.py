# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from guided_rec_parametrized import RunGuidedRecParametrized

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        base_args = []
        id = 200
        # for num_baseline_dropouts in [1, 3, 5, 10]:
        #     if num_baseline_dropouts == 1:
        #         list_of_args = [0,
        #                         num_baseline_dropouts,
        #                         "",
        #                         True,
        #                         "None",
        #                         True,
        #                         True,
        #                         True,
        #                         False,
        #                         2,
        #                         str(id),
        #                         0.001, id]
        #         id += 1
        #         base_args.append(list_of_args)
        #
        #     list_of_args = [0.1,
        #                     num_baseline_dropouts,
        #                     "",
        #                     True,
        #                     "None",
        #                     True,
        #                     True,
        #                     True,
        #                     False,
        #                     2,
        #                     str(id),
        #                     0.001, id]
        #     id += 1
        #     base_args.append(list_of_args)

        for num_baseline_dropouts in [2, 3, 5]:
            for alpha_risk in [2, 3, 5]:
                for local_losfun in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal"]:
                    for LR_LOGLOSS in [0.001, 0.0001]:
                        list_of_args = [0.05,
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
    for l in all_lists:
        RunGuidedRecParametrized(l)
    # with multiprocessing.Pool(processes=2) as pool:
    #     # a = executor.submit(LocalEval, list_of_args)
    #     results = pool.map(RunGuidedRecParametrized, all_lists)
    #     for r in results:
    #         print(r)
    #     # submits.append(a)
    #
    #     # if id == 4: return

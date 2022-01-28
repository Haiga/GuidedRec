# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from guided_rec_parametrized_aiv import RunGuidedRecParametrized

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run2():
        base_args = []
        id = 78010
        for num_baseline_dropouts in [1, 3, 5, 10]:
            if num_baseline_dropouts == 1:
                list_of_args = [0,
                                num_baseline_dropouts,
                                "",
                                True,
                                "None",
                                True,
                                True,
                                True,
                                False,
                                2,
                                str(id),
                                0.0005, id]
                id += 1
                base_args.append(list_of_args)
            else:
                list_of_args = [0.1,
                                num_baseline_dropouts,
                                "",
                                True,
                                "None",
                                True,
                                True,
                                True,
                                False,
                                2,
                                str(id),
                                0.0005, id]
                id += 1
            base_args.append(list_of_args)

        return base_args


    all_lists = run2()
    with multiprocessing.Pool(processes=1) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(RunGuidedRecParametrized, all_lists)
        for r in results:
            print(r)





    def run():
        base_args = []
        id = 78070

        for num_baseline_dropouts in [2, 5]:
            for alpha_risk in [2, 5]:
                for LR_LOGLOSS in [0.0005, 0.0001, 0.00006]:
                    for local_losfun in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal"]:

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
    with multiprocessing.Pool(processes=1) as pool:
        # a = executor.submit(LocalEval, list_of_args)
        results = pool.map(RunGuidedRecParametrized, all_lists)
        for r in results:
            print(r)
        # submits.append(a)

        # if id == 4: return

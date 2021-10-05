# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from guided_rec_parametrized_ml1m import RunGuidedRecParametrized

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        base_args = []
        id = 333336333
        for dataset in ["music", "ml1m", "ml100k"]:
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
                                    0.0005, id, dataset]
                    id += 1
                    base_args.append(list_of_args)

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
                                0.0005, id, dataset]
                id += 1
                base_args.append(list_of_args)

            for num_baseline_dropouts in [2]:
                for alpha_risk in [5]:
                    for local_losfun in ["NeuralSortCrossEntropyLossLocal", "GumbelApproxNDCGLossLocal"]:
                        for LR_LOGLOSS in [0.0005]:
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
                                            LR_LOGLOSS, id, dataset]
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

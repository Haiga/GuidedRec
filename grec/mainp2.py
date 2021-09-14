# from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from guided_rec_parametrized import RunGuidedRecParametrized

if __name__ == '__main__':
    # executor = ThreadPoolExecutor(max_workers=2)
    def run():
        base_args = []
        id = 200
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
                                0.0001, id]
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
                            0.0001, id]
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

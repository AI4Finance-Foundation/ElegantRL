# Evaluation

To evaluate your trained model, here are some steps to follow:

**Step 1**. Prepare your *pre-trained model checkpoint* and *test instances data file*. Put them in your preferred place. e.g., we will test the `AttentionModel` on TSP50:

```
.
├── rl4co/
│   └── ...
├── checkpoints/
│   └── am-tsp50.ckpt
└── data/
    └── tsp/
        └── tsp50_test_seed1234.npz
```

You can generate the test instances data file by running the following command:

```bash
python -c "from rl4co.data.generate_data import generate_default_datasets; generate_default_datasets('data')"
```

**Step 2**. Run the `eval.py` with your customized setting. e.g., let's use the `sampling` method with a `top_p=0.95` sampling strategy:

```bash
python rl4co/tasks/eval.py --problem tsp --data-path data/tsp/tsp50_test_seed1234.npz --model AttentionModel --ckpt-path checkpoints/am-tsp50.ckpt --method sampling --top-p 0.95
```

Arguments guideline:

- `--problem`: the problem name, e.g., `tsp`, `cvrp`, `pdp`, etc. This should be consistent with the `env.name`. Default is `tsp`.
- `--generator-params`: the generator parameters for the test instances. You could specify the `num_loc` etc. Default is `{'num_loc': 50}`.
- `--data-path`: the path to the test instances data file. Default is `data/tsp/tsp50_test_seed1234.npz`.
- `--model`: the model **class name**, e.g., `AttentionModel`, `POMO`, `SymNCO`, etc. It will be dynamically imported and instantiated. Default is `AttentionModel`.
- `--ckpt-path`: the path to the pre-trained model checkpoint. Default is `checkpoints/am-tsp50.ckpt`.
- `--device`: the device to run the evaluation, e.g., `cuda:0`, `cpu`, etc. Default is `cuda:0`.
- `--method`: the evaluation method, e.g., `greedy`, `sampling`, `multistart_greedy`, `augment_dihedral_8`, `augment`, `multistart_greedy_augment_dihedral_8`, and `multistart_greedy_augment`. Default is `greedy`.
- `--save-results`: whether to save the evaluation results as a `.pkl` file. Deafult is `True`. The results include `actions`, `rewards`, `inference_time`, and `avg_reward`.
- `--save-path`: the path to save the evaluation results. Default is `results/`.
- `--num-instances`: the number of test instances to evaluate. Default is `1000`.

If you use the `sampling` method, you may need to specify the following parameters:

- `--samples`: the number of samples for the sampling method. Default is `1280`.
- `--temperature`: the temperature for the sampling method. Default is `1.0`.
- `--top-p`: the top-p for the sampling method. Default is `0.0`, i.e. not activated.
- `--top-k`: the top-k for the sampling method. Deafult is `0`, i.e. not activated.
- `--select-best`: whether to select the best action from the sampling results. If `False`, the results will include all sampled rewards, i.e., `[num_instances * num_samples]`.

If you use the `augment` method, you may need to specify the following parameters:

- `--num-augments`: the number of augmented instances for the augment method. Default is `8`.
- `--force-dihedral-8`: whether to force the augmented instances to be dihedral 8. Default is `True`.

**Step 3**. If you want to launch several evaluations with various parameters, you may refer to the following examples:

- Evaluate POMO on TSP50 with a sampling of different Top-p and temperature:

    ```bash
        #!/bin/bash

        top_p_list=(0.5 0.6 0.7 0.8 0.9 0.95 0.98 0.99 0.995 1.0)
        temp_list=(0.1 0.3 0.5 0.7 0.8 0.9 1.0 1.1 1.2 1.5 1.8 2.0 2.2 2.5 2.8 3.0)

        device=cuda:0

        problem=tsp
        model=POMO
        ckpt_path=checkpoints/pomo-tsp50.ckpt
        data_path=data/tsp/tsp50_test_seed1234.npz

        num_instances=1000
        save_path=results/tsp50-pomo-topp-1k

        for top_p in ${top_p_list[@]}; do
            for temp in ${temp_list[@]}; do
                python rl4co/tasks/eval.py --problem ${problem} --model ${model} --ckpt_path ${ckpt_path} --data_path ${data_path} --save_path ${save_path} --method sampling --temperature=${temp} --top_p=${top_p} --top_k=0 --device ${device}
            done
        done
    ```

- Evaluate POMO on CVRP50 with a sampling of different Top-k and temperature:

    ```bash
        #!/bin/bash

        top_k_list=(5 10 15 20 25)
        temp_list=(0.1 0.3 0.5 0.7 0.8 0.9 1.0 1.1 1.2 1.5 1.8 2.0 2.2 2.5 2.8 3.0)

        device=cuda:1

        problem=cvrp
        model=POMO
        ckpt_path=checkpoints/pomo-cvrp50.ckpt
        data_path=data/vrp/vrp50_test_seed1234.npz

        num_instances=1000
        save_path=results/cvrp50-pomo-topk-1k

        for top_k in ${top_k_list[@]}; do
            for temp in ${temp_list[@]}; do
                python rl4co/tasks/eval.py --problem ${problem} --model ${model} --ckpt_path ${ckpt_path} --data_path ${data_path} --save_path ${save_path} --method sampling --temperature=${temp} --top_p=0.0 --top_k=${top_k} --device ${device}
            done
        done
    ```

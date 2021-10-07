# Meta-Learning-based Deep Reinforcement Learning for Multiobjective Optimization Problems

## Dependencies

* Python>=3.6
* NumPy
* [PyTorch](http://pytorch.org/)>=1.4

## Meta-Learning

For training meta-model on MOTSP-20 instances:
```bash
python run.py --graph_size 20 --CUDA_VISIBLE_ID "0" --is_train --meta_iterations 10000
```

For training meta-model on MOTSP-50 instances:

```bash
python run.py --graph_size 50 --CUDA_VISIBLE_ID "0" --is_train --meta_iterations 5000
```

You can initialize or resume a run using a pretrained meta-model by using the `--load_path` option, e.g.:

```bash
python run.py --graph_size 50 --is_load --load_path "meta-model-MOTSP50.pt" --CUDA_VISIBLE_ID "0" --is_train --meta_iterations 10000 --start_meta_iteration 5000
```

## Fine-tuning

For fine-tuning the trained meta-model on MOTSP-50 instances with 10-step per subproblem:
```bash
python run.py --graph_size 50 --is_load --load_path "meta-model-MOTSP50.pt" --CUDA_VISIBLE_ID "0" --is_test --update_step_test 10
```

For fine-tuning the trained meta-model on MOTSP-30 instances with 100-step per subproblem:

```bash
python run.py --graph_size 30 --is_load --load_path "meta-model-MOTSP50.pt" --CUDA_VISIBLE_ID "0" --is_test --update_step_test 100
```

For fine-tuning the random-model on MOTSP-50 instances with 10-step per subproblem:

```bash
python run.py --graph_size 50 --CUDA_VISIBLE_ID "0" --is_test --update_step_test 10
```

## Transfer-Learning

For training all the submodels with transfer-learning by loading the well trained 1st-submodel on MOTSP-50 instances with 10-step per subproblem:

```bash
python run.py --graph_size 50 --is_load --load_path "model-0.pt" --CUDA_VISIBLE_ID "0" --is_transfer --is_test --update_step_test 10
```

## Acknowledgements

Thanks to [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for getting me started with the code for the Attention Model.


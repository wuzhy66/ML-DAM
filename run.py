#!/usr/bin/env python
import re
import os
import numpy as np
import json
import pprint as pp
import random
import torch
from torch import nn
import torch.optim as optim
import time
from nets.attention_model import set_decode_type
from tqdm import tqdm

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import CriticBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem
from torch.utils.data import DataLoader
from utils import move_to
from copy import deepcopy
from train import clip_grad_norms
from collections import OrderedDict

opts = get_options()
os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_ID
load_path0 = opts.load_path if opts.is_load else None


opts.baseline = 'critic'
opts.run_name = 'step{}_task{}'.format(opts.update_step, opts.task_num)
opts.save_dir = os.path.join(
    opts.output_dir,
    "{}_{}".format(opts.problem, opts.graph_size),
    "meta_MORL_{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
)
opts.npy_save_path = os.path.join("npy", "{}_tsp{}_ts{}_st{}_fst{}.npy".format(
    opts.meta_algorithm,
    opts.graph_size, opts.update_step, opts.task_num,
    opts.update_step_test))

if opts.is_load is False and opts.is_train is False:
    opts.npy_save_path = os.path.join("npy", "no_meta_tsp{}_teststep{}.npy".format(opts.graph_size,
                                                                                   opts.update_step_test))
if opts.is_transfer:
    opts.npy_save_path = os.path.join("npy", "transfer", "tsp{}_step{}.npy".format(
        opts.graph_size, opts.update_step_test))
pp.pprint(vars(opts))

if opts.is_train:
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

torch.manual_seed(opts.seed)
opts.device = torch.device("cuda")
problem = load_problem(opts.problem)

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, model, baseline, opts, optimizer=None):

        super(Meta, self).__init__()
        self.meta_algorithm = opts.meta_algorithm
        self.sub_train_lr = opts.sub_train_lr
        self.meta_lr = opts.meta_lr
        self.finetunning_lr = opts.finetunning_lr
        self.update_step = opts.update_step
        self.update_step_test = opts.update_step_test
        self.task_num = opts.task_num

        self.actor_model = model
        self.baseline = baseline
        self.sub_actor_model = deepcopy(model)
        self.sub_baseline = deepcopy(baseline)

        self.meta_optim = optim.Adam(
            [{'params': model.parameters(), 'lr': self.meta_lr}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': self.meta_lr}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        ) if optimizer is None else optimizer

        set_decode_type(self.sub_actor_model, "sampling")

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def get_module_layer(self, model, name):
        """Get specific layer according to name"""
        items = name.split('.')[:-1]

        layer = model
        for item in items:
            layer = getattr(layer, item)
        return layer

    def copy_layer(self, source, target):
        """Copy specific layer's parameters"""
        for key, value in source._parameters.items():
            if value is not None:
                target._parameters[key] = value.clone()

    def copy_module(self, source, target):
        """Copy module's parameters"""
        all_params = source.named_parameters()
        for name, _ in all_params:
            self.copy_layer(self.get_module_layer(source, name), self.get_module_layer(target, name))

    def forward(self, query_batch=None, new_update_lr=None, meta_lr=None):

        update_lr = self.sub_train_lr if new_update_lr is None else new_update_lr
        support_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.batch_size * self.update_step, distribution=opts.data_distribution)
        support_dataloader = DataLoader(support_dataset, batch_size=opts.batch_size, num_workers=0)

        if self.meta_algorithm == "reptile":
            self.meta_lr = meta_lr if meta_lr is not None else 0.1
            actor_new_weights = []
            critic_new_weights = []
            actor_weights_original = deepcopy(self.actor_model.state_dict())
            critic_weights_original = deepcopy(self.baseline.critic.state_dict())

            for task_id in range(self.task_num):
                random_num = random.random()
                weight = torch.tensor([random_num, 1.0 - random_num])

                # print("task_id:{}, weight:{}".format(task_id, weight))
                self.sub_actor_model.load_state_dict(actor_weights_original)
                self.sub_baseline.critic.load_state_dict(critic_weights_original)

                optimizer = optim.Adam(
                    [{'params': self.sub_actor_model.parameters(), 'lr': update_lr}]
                    + (
                        [{'params': self.sub_baseline.get_learnable_parameters(), 'lr': update_lr}]
                        if len(self.sub_baseline.get_learnable_parameters()) > 0
                        else []
                    )
                )

                for update_id, support_batch in enumerate(support_dataloader):
                    s_x = move_to(support_batch, opts.device)
                    cost, log_likelihood = self.sub_actor_model(s_x)
                    bl_val, bl_loss = self.sub_baseline.eval(s_x, cost, weight)

                    c = weight[0] * cost[0] + weight[1] * cost[1]
                    reinforce_loss = ((c - bl_val) * log_likelihood).mean()
                    sub_loss = reinforce_loss + bl_loss

                    optimizer.zero_grad()
                    sub_loss.backward()
                    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
                    optimizer.step()
                    # print("update_id:{}, cost_obj1:{}, cost_obj2:{}, sub_loss:{}".format(update_id, cost[0].mean(
                    # ), cost[1].mean(), sub_loss))

                    torch.cuda.empty_cache()

                actor_new_weights.append(deepcopy(self.sub_actor_model.state_dict()))
                critic_new_weights.append(deepcopy(self.sub_baseline.critic.state_dict()))
                torch.cuda.empty_cache()

            actor_ws = len(actor_new_weights)
            actor_fweights = {name: actor_new_weights[0][name] / float(actor_ws) for name in actor_new_weights[0]}
            for i in range(1, actor_ws):
                # cur_weights = deepcopy(model.state_dict())
                for name in actor_new_weights[i]:
                    actor_fweights[name] += actor_new_weights[i][name] / float(actor_ws)

            critic_ws = len(critic_new_weights)
            critic_fweights = {name: critic_new_weights[0][name] / float(critic_ws) for name in critic_new_weights[0]}
            for i in range(1, critic_ws):
                # cur_weights = deepcopy(model.state_dict())
                for name in critic_new_weights[i]:
                    critic_fweights[name] += critic_new_weights[i][name] / float(critic_ws)

            self.actor_model.load_state_dict({name:
                                                  actor_weights_original[name] + (
                                                          actor_fweights[name] - actor_weights_original[
                                                      name]) * self.meta_lr for
                                              name in actor_weights_original})
            self.baseline.critic.load_state_dict({name:
                                                      critic_weights_original[name] + (
                                                              critic_fweights[name] - critic_weights_original[
                                                          name]) * self.meta_lr for
                                                  name in critic_weights_original})

            return

        elif self.meta_algorithm == "maml":
            q_x = move_to(query_batch, opts.device)
            meta_loss = torch.tensor(0.0)
            meta_loss = meta_loss.to(opts.device)
            for task_id in range(self.task_num):

                random_num = random.random()
                weight = torch.tensor([random_num, 1.0 - random_num])

                # print("task_id:{}, weight:{}".format(task_id, weight))
                self.copy_module(self.actor_model, self.sub_actor_model)
                self.copy_module(self.baseline.critic, self.sub_baseline.critic)

                for update_id, support_batch in enumerate(support_dataloader):
                    s_x = move_to(support_batch, opts.device)
                    cost, log_likelihood = self.sub_actor_model(s_x)
                    bl_val, bl_loss = self.sub_baseline.eval(s_x, cost, weight)

                    c = weight[0] * cost[0] + weight[1] * cost[1]
                    reinforce_loss = ((c - bl_val) * log_likelihood).mean()
                    sub_loss = reinforce_loss + bl_loss

                    actor_grad = torch.autograd.grad(sub_loss, list(self.sub_actor_model.parameters()),
                                                     retain_graph=opts.no_first_order,
                                                     create_graph=opts.no_first_order)
                    baseline_grad = torch.autograd.grad(sub_loss, list(self.sub_baseline.critic.parameters()),
                                                        retain_graph=opts.no_first_order,
                                                        create_graph=opts.no_first_order)

                    # update \theta'
                    for j, param in enumerate(self.sub_actor_model.parameters()):
                        param -= update_lr * actor_grad[j]
                    for j, param in enumerate(self.sub_baseline.critic.parameters()):
                        param -= update_lr * baseline_grad[j]

                    # print("update_id:{}, cost_obj1:{}, cost_obj2:{}, sub_loss:{}".format(update_id, cost[0].mean(
                    # ).item(), cost[1].mean().item(), sub_loss.item()))
                    torch.cuda.empty_cache()
                # test
                cost_t, log_likelihood_t = self.sub_actor_model(q_x)
                bl_val_t, bl_loss_t = self.sub_baseline.eval(q_x, cost_t, weight)
                c_t = weight[0] * cost_t[0] + weight[1] * cost_t[1]
                reinforce_loss_t = ((c_t - bl_val_t) * log_likelihood_t).mean()
                sub_loss_t = reinforce_loss_t + bl_loss_t
                meta_loss += sub_loss_t

                # print("-----------------------------------------------")
                # del sub_loss_t, reinforce_loss_t, c_t, bl_val_t, bl_loss_t, cost_t, log_likelihood_t
                torch.cuda.empty_cache()

            meta_loss = meta_loss / self.task_num
            print("meta_loss:", meta_loss.item())
            # optimize meta-policy parameters
            self.meta_optim.zero_grad()
            meta_loss.backward()
            grad_norms = clip_grad_norms(self.meta_optim.param_groups, opts.max_grad_norm)
            self.meta_optim.step()
            torch.cuda.empty_cache()

    def finetunning(self, weight=None):
        if weight is None:
            weight = torch.tensor([0.0, 1.0])
        print("--------------------------------------------")
        print("weight:{}, {}".format(weight[0].item(), weight[1].item()))
        actor_model = deepcopy(self.actor_model)
        baseline = deepcopy(self.baseline)

        optimizer = optim.Adam(
            [{'params': actor_model.parameters(), 'lr': self.finetunning_lr}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': self.finetunning_lr}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
        finetunning_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.batch_size * self.update_step_test,
            distribution=opts.data_distribution)
        finetunning_dataloader = DataLoader(finetunning_dataset, batch_size=opts.batch_size, num_workers=0)

        for update_id, batch in enumerate(finetunning_dataloader):  # for update_id in range(self.update_step_test):
            x = move_to(batch, opts.device)
            cost, log_likelihood = actor_model(x)
            bl_val, bl_loss = baseline.eval(x, cost, weight)
            c = weight[0] * cost[0] + weight[1] * cost[1]
            reinforce_loss = ((c - bl_val) * log_likelihood).mean()
            loss = reinforce_loss + bl_loss
            if self.update_step_test < 500:
                print(
                    "update_id:{}, cost_obj1:{}, cost_obj2:{}, loss:{}, lr:{}".format(update_id,
                                                                                      cost[0].mean().item(),
                                                                                      cost[1].mean().item(),
                                                                                      loss.item(),
                                                                                      optimizer.param_groups[0][
                                                                                          'lr']))

            optimizer.zero_grad()
            loss.backward()
            # grad_norms = clip_grad_norms(self.meta_optim.param_groups, opts.max_grad_norm)
            optimizer.step()
            # if optimizer.param_groups[0]['lr'] >= 1e-4:
            #     lr_scheduler.step(update_id)
            torch.cuda.empty_cache()

        return actor_model, baseline

    def validate(self, dataset, actor_model=None):
        if actor_model is None:
            print("validating meta_model...")
            actor_model = deepcopy(self.actor_model)
        else:
            print("validating sub_model...")

        set_decode_type(actor_model, "greedy")
        lcost_obj1 = []
        lcost_obj2 = []
        for batch in DataLoader(dataset, batch_size=opts.eval_batch_size):
            x = move_to(batch, opts.device)
            cost, log_likelihood = actor_model(x)
            lcost_obj1.append(cost[0])
            lcost_obj2.append(cost[1])
        cost_obj = [torch.cat(lcost_obj1, 0), torch.cat(lcost_obj2, 0)]

        torch.cuda.empty_cache()
        return [cost_obj[0].mean(), cost_obj[1].mean()]

# Initialize model
model_class = {
    'attention': AttentionModel,
}.get(opts.model, None)
assert model_class is not None, "Unknown model: {}".format(model_class)
model = model_class(
    opts.embedding_dim,
    opts.hidden_dim,
    problem,
    n_encode_layers=opts.n_encode_layers,
    mask_inner=True,
    mask_logits=True,
    normalization=opts.normalization,
    tanh_clipping=opts.tanh_clipping,
    checkpoint_encoder=opts.checkpoint_encoder,
    shrink_size=opts.shrink_size
).to(opts.device)

if opts.use_cuda and torch.cuda.device_count() > 1 and len(opts.CUDA_VISIBLE_ID) > 1:
    model = torch.nn.DataParallel(model)

# Initialize baseline
assert problem.NAME == 'tsp' and opts.baseline == 'critic', "Critic only supported for TSP"
baseline = CriticBaseline(
    (
        CriticNetwork(
            4,
            opts.embedding_dim,
            opts.hidden_dim,
            opts.n_encode_layers,
            opts.normalization
        )
    ).to(opts.device)
)

# Initialize optimizer
optimizer = optim.Adam(
    [{'params': model.parameters(), 'lr': opts.meta_lr}]
    + (
        [{'params': baseline.get_learnable_parameters(), 'lr': opts.meta_lr}]
        if len(baseline.get_learnable_parameters()) > 0
        else []
    )
)
load_path = load_path0
if load_path is not None:
    print('  [*] Loading data from {}'.format(load_path))
    load_data = torch_load_cpu(load_path)
    model_ = get_inner_model(model)
    if opts.is_load_multi:
        state_dict = load_data.get('model', {})
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model_.load_state_dict({**model_.state_dict(), **new_state_dict})
    else:
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

val_dataset = problem.make_dataset(
    size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

meta_learner = Meta(model, baseline, opts)  # , optimizer)
model.train()
set_decode_type(model, "sampling")

print("meta_learning_rate:{}, update_lr:{}, finetunning_lr:{}".format(meta_learner.meta_optim.param_groups[0]['lr'],
                                                                      meta_learner.sub_train_lr,
                                                                      meta_learner.finetunning_lr))
if opts.is_train:
    if opts.meta_algorithm == "maml":
        maml_learner = meta_learner
        for epoch_id in range(opts.start_epoch, opts.start_epoch + opts.training_epochs):
            print("epoch_id:{}, meta_learning_rate:{}, update_lr:{}, finetunning_lr:{}".format(epoch_id,
                                                                                               maml_learner.meta_optim.param_groups[
                                                                                                   0]['lr'],
                                                                                               maml_learner.sub_train_lr,
                                                                                               maml_learner.finetunning_lr))
            query_dataset = problem.make_dataset(
                size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
            query_dataloader = DataLoader(query_dataset, batch_size=opts.batch_size, num_workers=0)
            for batch_id, query_batch in enumerate(tqdm(query_dataloader)):
                maml_learner(query_batch)
                if batch_id % 30 == 0:
                    weight = torch.tensor([0.0, 1.0])
                    actor_model, baseline = maml_learner.finetunning(weight)
                    cost_obj = maml_learner.validate(val_dataset, actor_model)
                    print("validate_cost_obj1:{}, validate_cost_obj2:{}".format(cost_obj[0], cost_obj[1]))
            torch.save(
                {
                    'model': maml_learner.actor_model.state_dict(),
                    'optimizer': maml_learner.meta_optim.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                    # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'baseline': maml_learner.baseline.state_dict()
                },
                os.path.join(opts.save_dir, 'meta-model-epoch-{}.pt'.format(epoch_id))
            )
    elif opts.meta_algorithm == "reptile":
        reptile_learner = meta_learner
        for meta_iteration_id in tqdm(range(opts.start_meta_iteration, opts.meta_iterations)):
            meta_lr = opts.meta_lr * (1. - meta_iteration_id / float(opts.meta_iterations))
            if meta_iteration_id % 30 == 0:
                print("meta_iteration_id:{}, meta_learning_rate:{}, update_lr:{}, finetunning_lr:{}".format(
                    meta_iteration_id,
                    reptile_learner.meta_lr,
                    reptile_learner.sub_train_lr,
                    reptile_learner.finetunning_lr))
                weight = torch.tensor([0.0, 1.0])
                actor_model, baseline = reptile_learner.finetunning(weight)
                cost_obj = reptile_learner.validate(val_dataset, actor_model)
                print("validate_cost_obj1:{}, validate_cost_obj2:{}".format(cost_obj[0], cost_obj[1]))
            # t1 = time.time()
            reptile_learner(meta_lr=meta_lr)
            # t2 = time.time()
            # print("delta_time:", t2-t1)
            if meta_iteration_id % 100 == 0:
                torch.save(
                    {
                        'model': reptile_learner.actor_model.state_dict(),
                        'optimizer': reptile_learner.meta_optim.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                        # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': reptile_learner.baseline.state_dict()
                    },
                    os.path.join(opts.save_dir, 'meta-model-{}.pt'.format(meta_iteration_id))
                )
        torch.save(
            {
                'model': reptile_learner.actor_model.state_dict(),
                'optimizer': reptile_learner.meta_optim.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': reptile_learner.baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'meta-model-{}.pt'.format(meta_iteration_id))
        )

if opts.is_test and opts.is_transfer is not True:
    test_save_dir = os.path.join(
        "sub_outputs",
        "{}_{}".format(opts.problem, opts.graph_size),
        "{}_{}".format(opts.npy_save_path[4:-4], time.strftime("%Y%m%dT%H%M%S"))
    )
    os.makedirs(test_save_dir)
    w1 = torch.tensor([i * 1.0 / 99 for i in range(100)]).unsqueeze(1)
    # w1 = torch.tensor([i * 1.0 / 9 for i in range(10)]).unsqueeze(1)
    w1 = w1.to(opts.device)
    w2 = 1.0 - w1
    weights = torch.cat((w1, w2), dim=-1)

    cost_obj = meta_learner.validate(val_dataset)
    print("cost_obj1:{}, cost_obj2:{}".format(cost_obj[0].item(), cost_obj[1].item()))
    f1 = []
    f2 = []
    for i in range(100):
        if i < opts.resume_i:
            continue
        weight = weights[i]
        actor_model, baseline = meta_learner.finetunning(weight)

        cost_obj = meta_learner.validate(val_dataset, actor_model)
        print("cost_obj1:{}, cost_obj2:{}".format(cost_obj[0].item(), cost_obj[1].item()))

        f1.append(cost_obj[0].item())
        f2.append(cost_obj[1].item())
        if opts.use_cuda and torch.cuda.device_count() > 1 and len(opts.CUDA_VISIBLE_ID) > 1:
            torch.save(
                {
                    'model': actor_model.module.state_dict(),
                    'optimizer': meta_learner.meta_optim.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                    # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'baseline': baseline.state_dict()
                },
                os.path.join(test_save_dir, 'model-{}.pt'.format(i))
            )
        else:
            torch.save(
                {
                    'model': actor_model.state_dict(),
                    'optimizer': meta_learner.meta_optim.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                    # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'baseline': baseline.state_dict()
                },
                os.path.join(test_save_dir, 'model-{}.pt'.format(i))
            )

    f1 = np.array(f1)
    f2 = np.array(f2)
    f1 = f1[:, np.newaxis]
    f2 = f2[:, np.newaxis]
    result = np.concatenate((f1, f2), axis=1)
    # np.save(opts.npy_save_path, result)
    print(result)

if opts.is_transfer:
    transfer_save_dir = os.path.join(
        "transfer_outputs",
        "{}_{}".format(opts.problem, opts.graph_size),
        "transfer_step{}_{}".format(opts.update_step_test, time.strftime("%Y%m%dT%H%M%S"))
    )
    os.makedirs(transfer_save_dir)
    w1 = torch.tensor([i * 1.0 / 99 for i in range(100)]).unsqueeze(1)
    # w1 = torch.tensor([i * 1.0 / 9 for i in range(10)]).unsqueeze(1)
    w1 = w1.to(opts.device)
    w2 = 1.0 - w1
    weights = torch.cat((w1, w2), dim=-1)

    cost_obj = meta_learner.validate(val_dataset)
    print("cost_obj1:{}, cost_obj2:{}".format(cost_obj[0].item(), cost_obj[1].item()))

    for i in range(100):
        if opts.is_load and i == 0:
            # i = int(re.search(r'model-(.*?).pt', opts.load_path, re.M | re.I).group(1))
            cost_obj = meta_learner.validate(val_dataset)
            print("cost_obj1:{}, cost_obj2:{}".format(cost_obj[0].item(), cost_obj[1].item()))
            torch.save(
                {
                    'model': meta_learner.actor_model.state_dict(),
                    'optimizer': meta_learner.meta_optim.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                    # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'baseline': meta_learner.baseline.state_dict()
                },
                os.path.join(transfer_save_dir, 'model-{}.pt'.format(i))
            )
            continue
        weight = weights[i]

        actor_model, baseline = meta_learner.finetunning(weight)

        cost_obj = meta_learner.validate(val_dataset, actor_model)
        print("cost_obj1:{}, cost_obj2:{}".format(cost_obj[0].item(), cost_obj[1].item()))

        torch.save(
            {
                'model': actor_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(device=opts.device),
                # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(transfer_save_dir, 'model-{}.pt'.format(i))
        )

        meta_learner = Meta(actor_model, baseline, opts)  # , optimizer)
        actor_model.train()
        set_decode_type(actor_model, "sampling")


import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, weight = None):
    # Validate
    print('Validating...')
    cost, cost_obj = rollout(model, dataset, opts, weight)
    avg_cost = cost.mean()
    avg_cost_obj1 = cost_obj[0].mean()
    avg_cost_obj2 = cost_obj[1].mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    if weight is None:
        return avg_cost
    else:
        return avg_cost, [avg_cost_obj1, avg_cost_obj2]


def rollout(model, dataset, opts, weight=None):

    def eval_model_bat(bat):
        with torch.no_grad():
            cost_obj, _ = model(move_to(bat, opts.device))
        cost = weight[0] * cost_obj[0] + weight[1] * cost_obj[1]
        return cost.data.cpu(), [cost_obj[0].data.cpu(), cost_obj[1].data.cpu()]

    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    if weight is None:
        print("weight is None")
        weight = torch.FloatTensor([1, 0])
        weight = weight.to(opts.device)
        return
    else:
        lcost=[]
        lcost_obj1 = []
        lcost_obj2 = []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            cost, cost_obj=eval_model_bat(bat)
            lcost.append(cost)
            lcost_obj1.append(cost_obj[0])
            lcost_obj2.append(cost_obj[1])
        return torch.cat(lcost, 0), [torch.cat(lcost_obj1, 0),torch.cat(lcost_obj2, 0)]






def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, weight):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch

    #training_dataset = baseline.wrap_dataset(problem.make_dataset(
    #    size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    #training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    training_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            weight
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        weight
):
    #x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(batch, opts.device)
    #bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost, weight)
    c = weight[0] * cost[0] + weight[1] * cost[1]
    # Calculate loss
    reinforce_loss = ((c - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    # reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    # loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(c, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

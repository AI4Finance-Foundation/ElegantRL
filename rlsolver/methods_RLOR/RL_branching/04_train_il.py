# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the imitation learning method of Gasse et al.               #
# Output is saved to experiments/<problem>/04_train_il/<seed>_<timestamp>       #
# Usage: python 04_train_il.py <problem> <sample_dir> -s <seed> -g <cudaId>     #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import data
import json
import glob
import time
import argparse
import util
import model as ml
import torch as th
import wandb as wb
import torch_geometric

from util import log, extract_MLP_statistics

from torch.utils.data import DataLoader


class Scheduler(th.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.step_result = -1

    def step(self, metrics, epoch=...):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.num_bad_epochs += 1
        self.step_result = -1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.step_result = 0  # NEW_BEST
        elif self.num_bad_epochs == self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = float(param_group['lr'])
                new_lr = old_lr * self.factor
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
            self.step_result = 1  # NO_PATIENCE
        elif self.num_bad_epochs == 2 * self.patience:
            self.step_result = 2  # ABORT


def process(policy, data_loader, optimizer=None):
    avg_loss = 0
    avg_acc = 0
    num_samples = 0

    training = optimizer is not None
    with th.set_grad_enabled(training):
        for state, action in data_loader:
            # batch = batch.to(device)
            target = action.float()
            output = policy(*state).squeeze()

            # Loss calculation for binary output
            loss = th.nn.BCELoss()(output, target)
            y_pred = th.round(output)

            if training:
                optimizer.zero_grad()
                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly

            avg_loss += loss.item() * action.shape[0]
            avg_acc += th.sum(th.eq(y_pred, target)).item()
            num_samples += action.shape[0]

    avg_loss /= max(num_samples, 1)
    avg_acc /= max(num_samples, 1)

    return avg_loss, avg_acc


if __name__ == "__main__":
    # read default config file
    with open("config.json", 'r') as f:
        config = json.load(f)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=config['problems']
    )
    parser.add_argument(
        'dir',
        help='The k_sols and sampling type directory.',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        default=config['seed'],
        type=int,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        default=config['gpu'],
        type=int,
    )
    args = parser.parse_args()

    # --- HYPER PARAMETERS --- #
    max_epochs = 10000
    batch_train = 32
    batch_valid = 128

    # --- PYTORCH SETUP --- #
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"
    th.manual_seed(args.seed)

    # --- POLICY AND DATA --- #
    difficulty = config['difficulty'][args.problem]
    sample_dir = f'data/{args.problem}/samples/{args.dir}/valid_{difficulty}'
    valid_files = [str(file) for file in glob.glob(sample_dir + '/*.pkl')]
    sample_dir = f'data/{args.problem}/samples/{args.dir}/train_{difficulty}'
    train_files = [str(file) for file in glob.glob(sample_dir + '/*.pkl')]

    if config['model'] == "MLP":
        model = ml.MLPPolicy().to(device)

        train_data = data.Dataset(train_files)
        valid_data = data.Dataset(valid_files)
        train_loader = DataLoader(train_data, batch_train, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_valid, shuffle=False)

        sample_data = data.Dataset(train_files + valid_files)
        sample_loader = DataLoader(sample_data, batch_size=1024)
        extract_MLP_statistics(sample_loader, len(sample_data))
    elif config['model'] == "GNN":
        model = ml.GNNPolicy().to(device)

        train_data = data.GraphDataset(train_files)
        valid_data = data.GraphDataset(valid_files)

        follow_batch = ['constraint_features_s',
                        'constraint_features_t',
                        'variable_features_s',
                        'variable_features_t']

        train_loader = torch_geometric.loader.DataLoader(train_data, batch_train, shuffle=True, follow_batch=follow_batch)
        valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_valid, shuffle=False, follow_batch=follow_batch)
    else:
        raise NotImplementedError

    optimizer = th.optim.Adam(model.parameters(), lr=config['lr_train_il'])
    scheduler = Scheduler(optimizer, factor=0.2, patience=config['patience'])

    # --- LOG --- #

    # Create timestamp to save weights
    timestamp = time.strftime('%Y-%m-%d--%H.%M.%S')
    experiment_dir = f'experiments/{args.problem}/04_train_il'
    running_dir = experiment_dir + f'/{args.seed}_{timestamp}'
    os.makedirs(running_dir, exist_ok=True)
    logfile = running_dir + '/il_train_log.txt'
    wb.init(project="rl2select", config=config)

    log(f"training files: {len(train_files)}", logfile)
    log(f"validation files: {len(valid_files)}", logfile)
    log(f"batch size (train): {batch_train}", logfile)
    log(f"batch_size (valid): {batch_valid}", logfile)
    log(f"max epochs: {config['num_epochs']}", logfile)
    log(f"learning rate: {config['lr_train_il']}", logfile)
    log(f"model size: {sum(p.numel() for p in model.parameters())}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"difficulty: {difficulty}", logfile)
    log(f"seed {args.seed}", logfile)
    log(f"gpu: {args.gpu}", logfile)

    best_epoch = 0
    total_elapsed_time = 0
    for epoch in range(config['num_epochs'] + 1):
        log(f"** Epoch {epoch}", logfile)
        start_time = time.time()

        # TRAIN
        train_loss, train_acc = process(model, train_loader, optimizer)
        log(f"  train loss: {train_loss:.3f} | accuracy: {train_acc:.3f}", logfile)
        wb.log({'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)

        # TEST
        valid_loss, valid_acc = process(model, valid_loader)
        log(f"  valid loss: {valid_loss:.3f} | accuracy: {valid_acc:.3f}", logfile)
        wb.log({'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)

        elapsed_time = time.time() - start_time
        total_elapsed_time += elapsed_time
        log(f"  elapsed time: {elapsed_time:.3f}s | total: {total_elapsed_time:.3f}s", logfile)

        scheduler.step(valid_loss)
        if scheduler.step_result == 0:  # NEW_BEST
            log(f"  found best model so far, valid_loss: {valid_loss:.3f}, acc: {valid_acc:.3f}", logfile)
            th.save(model.state_dict(), f'{running_dir}/best_params_il.pkl')
            best_epoch = epoch
        elif scheduler.step_result == 1:  # NO_PATIENCE
            log(f"  {scheduler.patience} epochs without improvement, lowering learning rate", logfile)
        elif scheduler.step_result == 2:  # ABORT
            log(f"  no improvements for {2 * scheduler.patience} epochs, early stopping", logfile)
            break

    model.load_state_dict(th.load(f'{running_dir}/best_params_il.pkl'))
    valid_loss, valid_acc = process(model, valid_loader)
    log(f"PROCESS COMPLETED: BEST MODEL FOUND IN EPOCH {best_epoch}", logfile)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} | BEST VALID ACCURACY: {valid_acc:0.3f}", logfile)
    os.makedirs(f'actor/{args.problem}', exist_ok=True)
    th.save(model.state_dict(), f'actor/{args.problem}/il_{args.dir}.pkl')

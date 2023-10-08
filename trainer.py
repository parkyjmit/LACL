import os
import sys
import pickle
import logging
from tqdm import tqdm
import torch
from torch import optim
import numpy as np
from time import time

from model.bgrl import BGRL
from model.edieggc import EDiEGGC
from data.data import QM9Dataloader

# torch.manual_seed(1234)
# np.random.seed(1234)


def train_run(args, directory, train_loader, valid_loader, test_loader, test_indices, lacl):
    # Prepare model
    model = EDiEGGC(args)
    model.to(args.device)
    if lacl:
        model.encoder.load_state_dict(torch.load(os.path.join(directory, args.exp_name+'_enc')))
        for param in model.encoder.parameters():
            param.requires_grad = False

    logging.info('Train start')
    start_time = time()

    history_file = os.path.join(directory, 'history_' + args.exp_name + '.pickle')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif args.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
        )
    early_stopping = EarlyStopping(patience=30)

    history = {'train': [], 'validation': [], 'test': []}
    history['test_indices'] = test_indices
    min_train_loss = 1e5
    min_valid_loss = 1e5
    best_epoch = 0
    for epoch in range(args.epochs):
        # Train
        train_loss = evaluate_model(args, args.device, model, train_loader, optimizer, 'train') / args.num_train
        history['train'].append(train_loss)

        # Validation
        valid_loss = evaluate_model(args, args.device, model, valid_loader, optimizer, 'valid') / args.num_valid
        history['validation'].append(valid_loss)

        if args.scheduler == 'plateau':
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(), os.path.join(directory, args.exp_name))
            min_train_loss = train_loss
            min_valid_loss = valid_loss
            best_epoch = epoch
            save_history(history, history_file)

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            logging.info(early_stopping.message)
            break

        logging.info(f'Epoch {epoch + 1}: Train RMSE: {train_loss:.5f}, Validation MAE: {valid_loss:.5f}  '
              f'Time elapsed: {(time() - start_time)/3600:.5f}')

    logging.info(f'Best result at epoch: {best_epoch}, '
          f'Train RMSE: {min_train_loss:.5f}, Validation MAE: {min_valid_loss:.5f}')

    end_time = time()
    # Test
    model.load_state_dict(torch.load(os.path.join(directory, args.exp_name)))
    test_run_g, test_run_sg = evaluate_model(args, args.device, model, test_loader, optimizer, 'test')
    test_loss_g, test_loss_sg = test_run_g / args.num_test, test_run_sg / args.num_test
    history['test'].append((test_loss_g, test_loss_sg))
    save_history(history, history_file)
    logging.info(f'Test MAE: g: {test_loss_g:.5f} sg: {test_loss_sg:.5f} ')
    logging.info(f'Time elapsed: {(end_time - start_time)/3600:.5f}')

def save_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


def evaluate_model(args, device, model, loader, optimizer, split):
    if split == 'test':
        running_loss_g = running_loss_sg = 0.0
        model.eval()

        for g1, lg1, g2, lg2, label in tqdm(loader):
            g1 = g1.to(device)
            lg1 = lg1.to(device)
            g2 = g2.to(device)
            lg2 = lg2.to(device)
            label = label.to(device)
            with torch.no_grad():
                loss_g = calculate_loss(args, model, g1, lg1, label, optimizer, split)
                loss_sg = calculate_loss(args, model, g2, lg2, label, optimizer, split)
                running_loss_g += loss_g * g1.batch_size
                running_loss_sg += loss_sg * g2.batch_size
        return running_loss_g, running_loss_sg
    else:
        running_loss = 0.0
        if split == 'train':
            model.train()
        elif split == 'valid':
            model.eval()

        for g1, lg1, g2, lg2, label in tqdm(loader):
            g = g1 if args.set == 'src' else g2
            lg = lg1 if args.set == 'src' else lg2
            g = g.to(device)
            lg = lg.to(device)
            label = label.to(device)
            if split == 'train':
                loss = calculate_loss(args, model, g, lg, label, optimizer, split)
                running_loss += loss * g.batch_size
            else:
                with torch.no_grad():
                    loss = calculate_loss(args, model, g, lg, label, optimizer, split)
                    running_loss += loss * g.batch_size
        return running_loss



def calculate_loss(args, model, g, lg, label, optimizer, split):
    pred, _, _, _, _ = model(g, lg)
    if split == 'train':
        loss = torch.sqrt(torch.nn.functional.mse_loss(pred, label))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
    else:
        loss = torch.nn.functional.l1_loss(pred, label)
    return loss.item()



class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.message = ''

    def __call__(self, val_loss):
        if val_loss != val_loss:
            self.early_stop = True
            self.message = 'Early stopping: NaN appear'
        elif self.best_score is None:
            self.best_score = val_loss
        elif self.best_score < val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.message = 'Early stopping: No progress'
        else:
            self.best_score = val_loss
            self.counter = 0


class ModelTrainer:
    def __init__(self, args, directory, train_loader, valid_loader, test_loader, test_indices):
        self.args = args
        self.directory = directory
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.exp_name = args.exp_name
        self.device = args.device
        self.len_data = args.num_train
        # self.model = BGRL(args).to(self.device)
        self.model = BGRL(args)
        if self.args.finetune:
            self.model.load_state_dict(torch.load('logs/20230802_173538_qmugs_20_cgcf_fe_b/qmugs_20_cgcf_fe_b'))
            if self.args.freeze:
                self.model.online_encoder.requires_grad_(False)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        if args.scheduler == "none":
            # always return multiplier of 1 (i.e. do nothing)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)
        elif args.scheduler == "onecycle":
            steps_per_epoch = len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=args.learning_rate,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
            )
        elif args.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer)
        elif args.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5)
        elif args.scheduler == 'bgrl':
            scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
                        else (1 + np.cos((epoch-1000) * np.pi / (self.args.epochs - 1000))) * 0.5
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=scheduler)

        self.early_stopping = EarlyStopping(patience=30)

        self.history = {'train': [], 'validation': [], 'test': []}
        self.history['test_indices'] = test_indices
        self.history_file = os.path.join(directory, 'history_' + args.exp_name + '.pickle')
        self.min_train_loss = 1e5
        self.min_bgrl_loss = 1e5
        self.min_valid_loss = 1e5
        self.best_epoch = 0

    def train(self):
        # get initial test results
        logging.info("start training!")
        start_time = time()

        # start training
        for epoch in range(self.args.epochs):
            train_loss, bgrl_loss = self.evaluate_model(self.train_loader, 'train')
            self.history['train'].append(train_loss/self.args.num_train)
            valid_loss_1, valid_loss_2 = self.evaluate_model(self.valid_loader, 'valid')
            self.history['validation'].append(valid_loss_1/self.args.num_valid)

            logging.info(f'Epoch {epoch + 1}/{self.args.epochs}   '
                  f'Train loss: {train_loss/self.args.num_train:.5f}, BGRL loss: {bgrl_loss/self.args.num_train:.5f}   '
                  f'Valid MAE: g: {valid_loss_1/self.args.num_valid:.5f}, sg: {valid_loss_2/self.args.num_valid:.5f}  '
                  f'Time elapsed: {(time() - start_time)/3600:.5f}')

            self.scheduler_step(epoch, train_loss, bgrl_loss, valid_loss_2/self.args.num_valid)
            if self.early_stopping.early_stop:
                logging.info(self.early_stopping.message)
                break

        logging.info("Training Done!")
        logging.info(f'Best result at epoch: {self.best_epoch+1}, '
              f'Train RMSE: {self.min_train_loss/self.args.num_train:.5f}, '
              f'BGRL: {self.min_bgrl_loss/self.args.num_train:.5f} '
              f'Validation MAE: {self.min_valid_loss/self.args.num_valid:.5f}')

        end_time = time()
        # Test
        self.model.load_state_dict(torch.load(os.path.join(self.directory, self.args.exp_name)))
        test_loss_g, test_loss_sg = self.evaluate_model(self.test_loader, 'test')
        self.history['test'].append((test_loss_g/self.args.num_test, test_loss_sg/self.args.num_test))
        save_history(self.history, self.history_file)
        logging.info(f'Test MAE: g: {test_loss_g/self.args.num_test:.5f} sg: {test_loss_sg/self.args.num_test:.5f} ')
        logging.info(f'Time elapsed: {(end_time - start_time)/3600:.5f}')

    def scheduler_step(self, epoch, train_loss, bgrl_loss, valid_loss):
        if self.args.scheduler == 'plateau':
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

        if valid_loss < self.min_valid_loss:
            torch.save(self.model.state_dict(), os.path.join(self.directory, self.args.exp_name))
            self.min_train_loss = train_loss
            self.min_bgrl_loss = bgrl_loss
            self.min_valid_loss = valid_loss
            self.best_epoch = epoch
            save_history(self.history, self.history_file)

        self.early_stopping(valid_loss)

    def evaluate_model(self, loader, split):
        if split == 'train':
            running_loss = running_bgrl_loss = 0.0
            self.model.train()

            for g1, lg1, g2, lg2, label in tqdm(loader):
                g1 = g1.to(self.device)
                lg1 = lg1.to(self.device)
                g2 = g2.to(self.device)
                lg2 = lg2.to(self.device)
                label = label.to(self.device)
                loss_1, loss_2, bgrl_loss = self.calculate_loss(g1, lg1, g2, lg2, label, split)
                running_loss += loss_1 * g1.batch_size + loss_2 * g2.batch_size
                running_bgrl_loss += bgrl_loss * g1.batch_size
            return running_loss, running_bgrl_loss

        else:
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            running_bgrl_loss = 0.0
            self.model.eval()

            for g1, lg1, g2, lg2, label in tqdm(loader):
                g1 = g1.to(self.device)
                lg1 = lg1.to(self.device)
                g2 = g2.to(self.device)
                lg2 = lg2.to(self.device)
                label = label.to(self.device)
                with torch.no_grad():
                    loss_1, loss_2, bgrl_loss = self.calculate_loss(g1, lg1, g2, lg2, label, split)
                    running_loss_1 += loss_1 * g1.batch_size
                    running_loss_2 += loss_2 * g2.batch_size
                    running_bgrl_loss += bgrl_loss * g1.batch_size   
            if self.args.loss == 'contrastive':
                return running_bgrl_loss, running_bgrl_loss
            elif self.args.loss == 'prediction':
                return running_bgrl_loss, running_loss_2   
            elif self.args.loss == 'contrastive+prediction':
                return running_loss_1, running_loss_2

    def calculate_loss(self, g1, lg1, g2, lg2, label, split):
        pred_1, pred_2, bgrl_loss, _, _, _, _ = self.model(g1, lg1, g2, lg2)
        if split == 'train':
            label_loss_1 = torch.sqrt(torch.nn.functional.mse_loss(pred_1, label))
            label_loss_2 = torch.sqrt(torch.nn.functional.mse_loss(pred_2, label))
            if self.args.loss == 'contrastive':
                loss = bgrl_loss
            elif self.args.loss == 'prediction':
                loss = label_loss_2
            elif self.args.loss == 'contrastive+prediction':
                loss = bgrl_loss + label_loss_1
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            self.optimizer.step()
            if self.args.update_moving_average:
                self.model.update_moving_average()
        else:
            label_loss_1 = torch.nn.functional.l1_loss(pred_1, label)
            label_loss_2 = torch.nn.functional.l1_loss(pred_2, label)
        return label_loss_1.item(), label_loss_2.item(), bgrl_loss.item()

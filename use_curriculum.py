import models
import superloss
import utils
import variables
import evaluate

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import sys

batch_loss_history = {}
global_batch_counter = 0
loss_history = {}
trend_history = {}
conf_history = {}
tau_history = {}
tau_adjusted_history = {}
loss_trend_conf_history = {}


def update_trend_history(batch, current_trend_tensor):
    global trend_history
    for key, ctt in zip(batch.key, current_trend_tensor):
        key = tuple(key)
        if trend_history.get(key) is None:
            trend_history[key] = [ctt]
        else:
            trend_history[key].append(ctt)


def update_loss_history(batch, current_loss_tensor):
    global loss_history
    for key, clt in zip(batch.key, current_loss_tensor):
        key = tuple(key)
        if loss_history.get(key) is None:
            loss_history[key] = [clt]
        else:
            loss_history[key].append(clt)


def update_conf_history(batch, conf):
    global conf_history
    for key, cnf in zip(batch.key, conf):
        key = tuple(key)

        if conf_history.get(key) is None:
            conf_history[key] = [cnf]
        else:
            conf_history[key].append(cnf)


def update_tau_history(batch, tau):
    global tau_history
    for key, t in zip(batch.key, tau):
        key = tuple(key)

        if tau_history.get(key) is None:
            tau_history[key] = [t]
        else:
            tau_history[key].append(t)


def update_tau_adjusted_history(batch, tau_adjusted):
    global tau_adjusted_history
    for key, t in zip(batch.key, tau_adjusted):
        key = tuple(key)

        if tau_adjusted_history.get(key) is None:
            tau_adjusted_history[key] = [t]
        else:
            tau_adjusted_history[key].append(t)


def calculate_original_superloss(b_loss, batch, super_loss):
    conf, tau, tau_adjusted = super_loss(b_loss, None, None)
    tau = [tau] * b_loss.shape[0]
    tau_adjusted = [tau_adjusted] * b_loss.shape[0]
    sl_loss = b_loss * conf
    sl_loss = sl_loss.mean()
    final_loss = sl_loss

    update_loss_history(batch, b_loss)
    update_conf_history(batch, conf)
    update_tau_history(batch, tau)
    update_tau_adjusted_history(batch, tau_adjusted)

    return final_loss


def calculate_trend_using_previous_k_loss(b_loss, e, batch, nb_prev_loss):
    current_trend = []
    for k, clt in zip(batch.key, b_loss):

        k = tuple(k)
        if not loss_history.get(k) is None and e > 0:
            history = loss_history[k]

            history_k = history[-(nb_prev_loss - 1):] + [clt.item()]
        else:
            history_k = [0, clt.item()]

        sum_result = 0
        sum_result_abs = 0
        for i in range(len(history_k) - 1):
            sum_result = sum_result + history_k[i + 1] - history_k[i]

            sum_result_abs = sum_result_abs + abs(history_k[i + 1] - history_k[i])

        if sum_result_abs == 0:
            delta_i = 0

        else:

            delta_i = sum_result / sum_result_abs

        current_trend.append(delta_i)
    current_trend_tensor = torch.tensor(current_trend)
    return current_trend_tensor


def calculate_trend_superloss(b_loss, e, batch, device, alpha, super_loss, nb_prev_loss):

    current_trend_tensor = calculate_trend_using_previous_k_loss(b_loss, e, batch,nb_prev_loss)
    current_trend_tensor = current_trend_tensor.to(device)

    conf, tau, tau_adjusted = super_loss(b_loss, alpha, current_trend_tensor)
    tau = [tau] * b_loss.shape[0]

    sl_loss = b_loss * conf
    sl_loss = sl_loss.mean()
    final_loss = sl_loss

    update_trend_history(batch, current_trend_tensor)
    update_loss_history(batch, b_loss)
    update_conf_history(batch, conf)
    update_tau_history(batch, tau)
    update_tau_adjusted_history(batch, tau_adjusted)

    return final_loss


def calculate_superloss(b_loss, step, batch, training_type, device, alpha,super_loss,nb_prev_loss):
    if training_type == "regular":
        update_loss_history(batch, b_loss)
        return b_loss.mean()
    elif training_type == "sl":
        return calculate_original_superloss(b_loss, batch, super_loss)
    elif training_type == "sl_trend":
        return calculate_trend_superloss(b_loss, step, batch, device, alpha, super_loss,nb_prev_loss)

def init_model(args):
    
    node_feature_dim = utils.get_features_dim(args)
    add_additional_feature = args.add_additional_feature
    device = variables.device
    gs_dim = 100
    additional_feature_dim = utils.get_additional_features_dim(args)
    fusion_type = args.fusion_type
    model_type = args.model_type
    model = models.GTNN(node_feature_dim, add_additional_feature, device, gs_dim, additional_feature_dim,
                        fusion_type, model_type)
    
    return model

def train_model(train_loader, val_loader,  args):

    model = init_model(args).to(device)
    model = model.to(device) 
    
    lr = float(args.lr)
    L2 = float(args.l2)

    alpha = float(args.alpha)
    training_type = args.training_type
    sl_lambda = float(args.sl_lambda)
    mode = args.mode
    prev_k_loss = args.prev_k_loss
    nb_epoch = args.nb_epoch
    dataset = args.dataset
    neg_x = args.neg_x

    cls_weight = compute_class_weight("balanced", [0, 1], [0] * neg_x + [1])
    cls_weight = cls_weight.tolist()


    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=L2)
    bce_loss = nn.BCELoss(reduction='none')
    if training_type == "regular": 
        super_loss = None        
    else:
        super_loss = superloss.SuperLoss(training_type, lam=sl_lambda, mode=mode)

    nb_prev_loss = prev_k_loss
    print("nb_prev_loss ", nb_prev_loss)

    best_pref = 0

    for step in range(nb_epoch):
        model.train()

        losses = []
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            label = batch.y
            prediction = model(batch)
            loss = bce_loss(prediction.float(), label.float())

            if dataset == 'gdpr' or dataset == 'wiki':
                weight = torch.tensor(cls_weight).to(device)
                weight_ = weight[label.data.view(-1).long()].view_as(label)
                loss = loss * weight_
            else:
                pass

            loss = calculate_superloss(loss, step, batch, training_type, device, alpha, super_loss, nb_prev_loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            sys.stdout.flush()

        p, r, f1, _ = evaluate.eval_model(model, val_loader, device)
        print("e = {0:d} loss = {1:.6f} p = {2:.4f} r = {3:.4f} f1 = {4:.4f} ".format(step,sum(losses) / len(losses), p,r, f1))
        current_pref = f1
        if current_pref > best_pref or step == 0:
            best_pref = current_pref
            utils.save_the_best_model(model, step, optimizer, {"p": p, "r": r, "f1": f1}, args)
        else:
            pass
        sys.stdout.flush()

    return model



















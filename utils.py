from torch_geometric.data import DataLoader
import variables
import pickle
import torch
import numpy as np
import random
import os


def get_additional_features_dim(args):
    dataset = args.dataset

    if dataset == 'pgr':
        additional_feature_dim = 768
    elif dataset == 'wiki':
        additional_feature_dim = 50 * 2
    else:
        additional_feature_dim = 4
    return additional_feature_dim

def get_features_dim(args):
    dataset = args.dataset
    embedding_type = args.embedding_type

    if dataset == 'wiki':
        node_feat_dim = 50
    elif dataset =='pgr':
        if embedding_type == 'doc2vec' or embedding_type == 'random':
            node_feat_dim = 300
        else:
            node_feat_dim = 768

    elif dataset =='gdpr':
        if embedding_type == 'doc2vec' or embedding_type == 'random':
            node_feat_dim = 300
        else:
            node_feat_dim = 768

    return node_feat_dim


def load_datasets(args):
    bs = int(args.batch_size)
    dataset = args.dataset
    embedding_type = args.embedding_type
    num_workers = args.num_workers

    dataloader_loc = variables.dir_data + "/{}/{}_train_test_val_{}.pkl".format(dataset,dataset, embedding_type)
    train_set, val_set, test_set = pickle.load(open(dataloader_loc, "rb"))

    print("Train size = {}, val size = {}, test size = {}".format(len(train_set), len(val_set), len(test_set)))

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model_name(args):

    dataset = args.dataset
    ''' # uncomment for experiments with different combinations
    
    lr = args.lr
    L2 = args.l2
    add_additional_feature = args.add_additional_feature
    embedding_type = args.embedding_type
    neg_x = args.neg_x
    sl_lambda = args.sl_lambda
    fusion_type = args.fusion_type
    use_original_superloss = args.use_original_superloss
    use_superloss = args.use_superloss
    trend_method = args.trend_method
    mode = args.mode
    nb_epoch = args.nb_epoch
    alpha = args.alpha  # if 0 <= args.alpha <=1 else -1.0
    beta = args.beta if 0 <= args.beta <= 1 else -1.0
    prev_k_loss = args.prev_k_loss
    model_type = args.model_type
    # %%
    seed = args.seed

    model_var_order = [
        dataset,
        embedding_type,
        lr,
        L2,
        str(add_additional_feature),
        fusion_type,
        use_superloss,
        use_original_superloss,
        mode,
        neg_x,
        nb_epoch,
        sl_lambda,
        trend_method,
        alpha,
        beta,
        prev_k_loss,
        seed,
        model_type
    ]
    model_name = "{}_{}_lr_{}_l2_{}_add_feature_{}_fusion_{}_sl_{}_o_{}_m_{}_neg_x_{}_max_e_{}_sl_lambda_{}_tm_{}_a_{}_b_{}_prev_k_loss_{}_cls_weight_ezdf_es_acl_seed_{}_{}".format(
        *model_var_order)
    '''
    model_name = dataset
    return model_name


def save_the_best_model(model, epoch, optimizer, performance, args):
    model_name = get_model_name(args)
    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  "performance": performance}

    torch.save(checkpoint, variables.dir_model + '/{}_best.pth'.format(model_name))

    print("model saved.")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

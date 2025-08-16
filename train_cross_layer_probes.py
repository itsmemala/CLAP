import os
import sys
import argparse
import datetime
import pickle
from tqdm import tqdm
from copy import deepcopy

from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import wandb

from losses import SupConLoss
from utils import HF_NAMES # Variables
from utils import list_of_ints, list_of_floats, list_of_strs, load_labels, load_acts, get_act_dims, get_best_threshold # Functions
from utils import My_Transformer_Layer, My_Projection_w_Classifier_Layer, LogisticRegression_Torch, Ens_Att_Pool # Classes

torch.set_default_dtype(torch.float64)
act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise','layer_maxpool':'layer_wise','layer_att_res':'layer_wise'}


def combine_acts(idx,file_name,args):
    device = args.device
    file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
    if args.token=='prompt_last_and_answer_last':
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_prompt_last/{args.model_name}_{file_name}_prompt_last_{act_type[args.using_act]}_{file_end}.pkl'
        act1 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_answer_last/{args.model_name}_{file_name}_answer_last_{act_type[args.using_act]}_{file_end}.pkl'
        act2 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        act = torch.concatenate([act1[:,None,:],act2[:,None,:]],dim=1)
        # print(act1.shape,act.shape)
    elif args.token=='least_likely_and_last':
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_least_likely/{args.model_name}_{file_name}_least_likely_{act_type[args.using_act]}_{file_end}.pkl'
        act1 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_answer_last/{args.model_name}_{file_name}_answer_last_{act_type[args.using_act]}_{file_end}.pkl'
        act2 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        act = torch.concatenate([act1[:,None,:],act2[:,None,:]],dim=1)
    elif args.token=='prompt_last_and_least_likely_and_last':
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_prompt_last/{args.model_name}_{file_name}_prompt_last_{act_type[args.using_act]}_{file_end}.pkl'
        act1 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_least_likely/{args.model_name}_{file_name}_least_likely_{act_type[args.using_act]}_{file_end}.pkl'
        act2 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_answer_last/{args.model_name}_{file_name}_answer_last_{act_type[args.using_act]}_{file_end}.pkl'
        act3 = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        act = torch.concatenate([act1[:,None,:],act2[:,None,:],act3[:,None,:]],dim=1)
    return act


def compute_wp_dist(outputs,labels,device,metric='euclidean'):
    dist_same, dist_opp = [], []
    if metric=='euclidean':
        for i,o_i in enumerate(outputs):
            o_dist_same, o_dist_opp = [], []
            for j,o_j in enumerate(outputs):
                # print(o_i.shape, o_j.shape, torch.cdist(o_i[None,:], o_j[None,:], p=2.0)[0])
                if i!=j and labels[i]==labels[j]: 
                    o_dist_same.append(torch.cdist(o_i[None,:], o_j[None,:], p=2.0)[0]) # L2 distance between two samples
                elif i!=j and labels[i]!=labels[j]: 
                    o_dist_opp.append(torch.cdist(o_i[None,:], o_j[None,:], p=2.0)[0]) # L2 distance between two samples
            dist_same.append(torch.cat(o_dist_same).mean() if len(o_dist_same)>0 else torch.tensor(-10000).to(device))
            dist_opp.append(torch.cat(o_dist_opp).mean() if len(o_dist_opp)>0 else torch.tensor(10000).to(device))
        dist_same = torch.stack(dist_same)
        dist_opp = torch.stack(dist_opp)
        dist = torch.stack([dist_same, dist_opp], dim=1)
    elif metric=='cosine':
        outputs = F.normalize(outputs, p=2, dim=-1)
        for i,o_i in enumerate(outputs):
            o_dist_same, o_dist_opp = [], []
            for j,o_j in enumerate(outputs):
                # print(o_i.shape, o_j.shape, 1-F.cosine_similarity(o_i, o_j, dim=-1))
                if i!=j and labels[i]==labels[j]: 
                    o_dist_same.append(1-F.cosine_similarity(o_i, o_j, dim=-1)) # cosine distance between two samples
                elif i!=j and labels[i]!=labels[j]: 
                    o_dist_opp.append(1-F.cosine_similarity(o_i, o_j, dim=-1)) # cosine distance between two samples
            dist_same.append(torch.stack(o_dist_same).mean() if len(o_dist_same)>0 else torch.tensor(-10000).to(device)) # stack instead of cat since these are 0d tensors
            dist_opp.append(torch.stack(o_dist_opp).mean() if len(o_dist_opp)>0 else torch.tensor(10000).to(device))
        dist_same = torch.stack(dist_same)
        dist_opp = torch.stack(dist_opp)
        dist = torch.stack([dist_same, dist_opp], dim=1)
    elif metric=='cosine_individual':
        dist = []
        outputs = F.normalize(outputs, p=2, dim=-1)
        for i,o_i in enumerate(outputs):
            o_dist = []
            for j,o_j in enumerate(outputs):
                if i!=j: o_dist.append(1-F.cosine_similarity(o_i, o_j, dim=-1)) # cosine distance between two samples
            dist.append(torch.stack(o_dist))
        dist = torch.stack(dist)
    # print(dist,dist.shape)
    # sys.exit()
    return dist


def main(): 
    """
    Train probes on LLM activations.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='trivia_qa')
    parser.add_argument('--using_act', type=str, default='layer')
    parser.add_argument('--token', type=str, default='answer_last')
    parser.add_argument('--max_answer_tokens', type=int, default=20)
    parser.add_argument('--use_pe', type=bool, default=False)
    parser.add_argument('--method', type=str, default='transformer') # One of {'clap','linear','project_linear','project_non_linear'}
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--use_layers_list', type=list_of_ints, default=None)
    parser.add_argument('--ind_att_pool_probes_path', type=str, default=None)
    parser.add_argument('--use_dropout', type=bool, default=False)
    parser.add_argument('--use_batch_norm', type=bool, default=False)
    parser.add_argument('--no_bias', type=bool, default=False)
    parser.add_argument('--norm_emb', type=bool, default=False)
    parser.add_argument('--norm_cfr', type=bool, default=False)
    parser.add_argument('--cfr_no_bias', type=bool, default=False)
    parser.add_argument('--tfr_d_model', type=int, default=128)
    parser.add_argument('--no_act_proj', type=bool, default=False)
    parser.add_argument('--use_supcon_loss', type=bool, default=False)
    parser.add_argument('--supcon_temp', type=float, default=0.1)
    parser.add_argument('--sc1_wgt', type=float, default=1)
    parser.add_argument('--sc2_wgt', type=float, default=1)
    parser.add_argument('--len_dataset', type=int, default=5000)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--test_num_samples', type=int, default=1)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_list', type=list_of_floats, default=0.00005,0.0005,0.005,0.05,0.5)
    parser.add_argument('--scheduler', type=str, default='warmup_cosanneal')
    parser.add_argument('--best_using_auc', type=bool, default=False)
    parser.add_argument('--best_as_last', type=bool, default=False)
    parser.add_argument('--use_class_wgt', type=bool, default=False)
    parser.add_argument('--no_batch_sampling', type=bool, default=False)
    parser.add_argument('--acts_per_file', type=int, default=100)
    parser.add_argument('--save_probes',type=bool, default=False)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument("--train_file_name", type=str, default=None)
    parser.add_argument("--test_file_name", type=str, default=None)
    parser.add_argument("--train_labels_file_name", type=str, default=None)
    parser.add_argument("--test_labels_file_name", type=str, default=None)
    parser.add_argument('--wp_dist', type=bool, default=False)
    parser.add_argument('--wpdist_metric', type=str, default='euclidean')
    parser.add_argument('--test_bs', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--seed_list', type=list_of_ints, default=42,101,2650)
    parser.add_argument('--supcon_temp_list', type=list_of_floats, default=None)
    parser.add_argument('--wnb_plot_name', type=str, default=None) # Wandb args
    parser.add_argument('--tag', type=str, default=None) # Wandb args
    args = parser.parse_args()
    
    # torch.set_default_dtype(torch.float16)
    device = "cuda" if args.device is None else args.device

    print("Loading labels..")
    labels = load_labels(args.save_path,args.model_name,args.dataset_name,args.train_file_name,args.train_labels_file_name,args.num_samples)
    test_labels = load_labels(args.save_path,args.model_name,args.dataset_name,args.test_file_name,args.test_labels_file_name,args.test_num_samples)
    
    test_idxs = np.arange(len(test_labels))
    train_idxs = np.arange(len(labels)) # np.arange(args.len_dataset)
    
    print("Loading acts...")
    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise','layer_att_res':'layer_wise'}
    my_train_acts = load_acts(args,args.train_file_name,train_idxs)
    my_test_acts = load_acts(args,args.test_file_name,test_idxs)
    print(my_train_acts.shape)

    num_layers = my_train_acts.shape[1]
    args.use_layers_list = np.array(args.use_layers_list) if args.use_layers_list is not None else np.array([k for k in range(num_layers)])

    for seed_itr,save_seed in enumerate(args.seed_list):
        supcon_temp_search_list = [args.supcon_temp] if args.supcon_temp_list is None else args.supcon_temp_list
        for supcon_temp in supcon_temp_search_list:
            args.supcon_temp = supcon_temp
            for lr in args.lr_list:
                print('Training SEED',save_seed)
                print('Training sc temp',args.supcon_temp)
                print('Training lr',lr)
                args.lr=lr
                args.expt_name = f'{args.expt_name}_lr{args.lr}_sctemp{args.supcon_temp}_seed{save_seed}'

                # Probe training
                np.random.seed(save_seed)
                torch.manual_seed(save_seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed(save_seed)

                # Create dirs if does not exist:
                if not os.path.exists(f'{args.save_path}/probes/models/{args.expt_name}'):
                    os.makedirs(f'{args.save_path}/probes/models/{args.expt_name}', exist_ok=True)
                if not os.path.exists(f'{args.save_path}/probes/{args.expt_name}'):
                    os.makedirs(f'{args.save_path}/probes/{args.expt_name}', exist_ok=True)

                if 'sampled' in args.train_file_name:
                    num_prompts = int(args.len_dataset/args.num_samples)
                    labels_sample_dist = []
                    for k in range(num_prompts):
                        cur_prompt_idx = k*args.num_samples
                        sample_dist = sum(labels[cur_prompt_idx:cur_prompt_idx+args.num_samples])
                        if sample_dist==args.num_samples:
                            labels_sample_dist.append(0)
                        elif sample_dist==0:
                            labels_sample_dist.append(1)
                        elif sample_dist <= int(args.num_samples/3):
                            labels_sample_dist.append(2)
                        elif sample_dist > int(2*args.num_samples/3):
                            labels_sample_dist.append(3)
                        else:
                            labels_sample_dist.append(4)
                    if labels_sample_dist.count(0)==1 or labels_sample_dist.count(3)==1: labels_sample_dist[labels_sample_dist.index(3)] = 0
                    if labels_sample_dist.count(1)==1 or labels_sample_dist.count(2)==1: labels_sample_dist[labels_sample_dist.index(2)] = 1
                    if labels_sample_dist.count(4)==1: labels_sample_dist[labels_sample_dist.index(4)] = 1
                    if labels_sample_dist.count(0)==1: labels_sample_dist[labels_sample_dist.index(4)] = 0
                    if labels_sample_dist.count(1)==1: labels_sample_dist[labels_sample_dist.index(4)] = 1
                    train_prompt_idxs, val_prompt_idxs, _, _ = train_test_split(np.arange(num_prompts), labels_sample_dist, stratify=labels_sample_dist, test_size=0.2)
                    train_set_idxs = np.concatenate([np.arange(k*args.num_samples,(k*args.num_samples)+args.num_samples,1) for k in train_prompt_idxs], axis=0)
                    val_set_idxs = np.concatenate([np.arange(k*args.num_samples,(k*args.num_samples)+args.num_samples,1) for k in val_prompt_idxs], axis=0)
                else:
                    train_set_idxs, val_set_idxs, _, _ = train_test_split(train_idxs, labels, stratify=labels,test_size=0.2)

                y_train_supcon = np.stack([labels[i] for i in train_set_idxs], axis = 0)
                y_train = np.stack([[labels[i]] for i in train_set_idxs], axis = 0)
                y_val = np.stack([[labels[i]] for i in val_set_idxs], axis = 0)
                np.stack([test_labels[i] for i in test_idxs], axis = 0)
                
                cur_probe_y_train = np.stack([[labels[i]] for i in train_set_idxs], axis = 0)
                train_target = np.stack([labels[j] for j in train_set_idxs], axis = 0)
                class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                weight = 1. / class_sample_count
                samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight)) # Default: replacement=True
                if 'sampled' in args.train_file_name:
                    ds_train = Dataset.from_dict({"inputs_idxs": train_prompt_idxs}).with_format("torch")
                    ds_train = DataLoader(ds_train, batch_size=int(args.bs/args.num_samples), shuffle=True)
                else:
                    ds_train = Dataset.from_dict({"inputs_idxs": train_set_idxs, "labels": cur_probe_y_train}).with_format("torch")
                    ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler) if args.no_batch_sampling==False else DataLoader(ds_train, batch_size=args.bs)
                ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                ds_val = DataLoader(ds_val, batch_size=args.bs)
                ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                ds_test = DataLoader(ds_test, batch_size=args.bs if args.test_bs is None else args.test_bs)

                act_dims =  get_act_dims(args.model_name)['layer']
                bias = not args.no_bias
                n_blocks = args.n_blocks
                num_layers_to_use = num_layers if args.use_layers_list is None else len(args.use_layers_list)
                if args.using_act=='layer_maxpool': 
                    if args.method=='linear':
                        nlinear_model = LogisticRegression_Torch(n_inputs=act_dims, n_outputs=1, bias=bias, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias).to(device)
                    else:
                        nlinear_model = My_SupCon_NonLinear_Classifier4(input_size=act_dims, output_size=1, bias=bias, use_dropout=args.use_dropout, supcon=False, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias).to(device)
                elif args.method=='project_linear': 
                    nlinear_model = My_Projection_w_Classifier_Layer(n_inputs=act_dims, n_layers=num_layers_to_use, n_outputs=1, bias=bias, batch_norm=args.use_batch_norm, supcon=args.use_supcon_loss, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias, d_model=args.tfr_d_model, no_act_proj=args.no_act_proj).to(device)
                elif args.method=='project_non_linear':
                    nlinear_model = My_Projection_w_Classifier_Layer(n_inputs=act_dims, n_layers=num_layers_to_use, n_outputs=1, bias=bias, batch_norm=args.use_batch_norm, supcon=args.use_supcon_loss, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias, d_model=args.tfr_d_model, no_act_proj=args.no_act_proj, non_linear=True).to(device)
                elif args.method=='ens_att_pool': 
                    ind_att_pool_probes_path = f'{args.save_path}/probes/models/{args.ind_att_pool_probes_path}_model{i}'
                    nlinear_model = Ens_Att_Pool(n_inputs=my_train_acts.shape[1], n_outputs=1, bias=bias, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias, args.expt_name=ind_att_pool_probes_path).to(device)
                else:
                    nlinear_model = My_Transformer_Layer(n_inputs=act_dims, n_layers=num_layers_to_use, n_outputs=1, bias=bias, n_blocks=n_blocks, use_pe=args.use_pe, batch_norm=args.use_batch_norm, supcon=args.use_supcon_loss, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias, d_model=args.tfr_d_model, no_act_proj=args.no_act_proj).to(device)
                nlinear_model = nlinear_model.to(my_train_acts.dtype)
                
                wgt_0 = np.sum(cur_probe_y_train)/len(cur_probe_y_train)
                criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.BCEWithLogitsLoss()
                use_supcon_pos = True if 'supconv2_pos' in args.method else False
                sc_num_samples = args.num_samples if 'wp' in args.method else None
                if (use_supcon_pos) and (sc_num_samples is not None):
                    criterion_supcon1 = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=use_supcon_pos,num_samples=None) # operates on greedy samples only
                    criterion_supcon2 = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=False,num_samples=sc_num_samples,bs=args.bs) # operates within prompt only
                elif 'supconv2_reg_wp' in args.method:
                    criterion_supcon1 = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=False,num_samples=None) # operates on all
                    criterion_supcon2 = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=False,num_samples=sc_num_samples,bs=args.bs) # operates within prompt only
                else:
                    criterion_supcon = SupConLoss(temperature=args.supcon_temp,use_supcon_pos=use_supcon_pos,num_samples=sc_num_samples,bs=args.bs) if 'supconv2' in args.method else NTXentLoss()
                
                # Training
                print('\n\nStart time of train:',datetime.datetime.now(),'\n\n')
                supcon_train_loss, supcon1_train_loss, supcon2_train_loss, train_loss, val_loss, val_auc = [], [], [], [], [], []
                best_val_loss, best_val_auc = torch.inf, 0
                best_model_state = deepcopy(nlinear_model.state_dict())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                named_params = list(nlinear_model.named_parameters())
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.lr},
                    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
                ]
                optimizer = torch.optim.AdamW(optimizer_grouped_parameters) #torch.optim.AdamW(optimizer_grouped_parameters)
                steps_per_epoch = int(len(train_set_idxs)/args.bs)  # number of steps in an epoch
                warmup_period = steps_per_epoch * 5
                T_max = (steps_per_epoch*args.epochs) - warmup_period # args.epochs-warmup_period
                scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_period)
                scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)
                scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1) if args.scheduler=='static' else torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_period])
                for epoch in tqdm(range(args.epochs)):
                    num_samples_used, num_val_samples_used, epoch_train_loss, epoch_supcon_loss, epoch_supcon1_loss, epoch_supcon2_loss = 0, 0, 0, 0, 0, 0
                    nlinear_model.train()
                    for step,batch in enumerate(ds_train):
                        optimizer.zero_grad()
                        activations, batch_target_idxs = [], []
                        batch_input_idxs = np.concatenate([np.arange(k*args.num_samples,(k*args.num_samples)+args.num_samples,1) for k in batch['inputs_idxs']], axis=0) if 'sampled' in args.train_file_name else batch['inputs_idxs']
                        activations = my_train_acts[batch_input_idxs].to(device)
                        inputs = activations
                        if 'sampled' in args.train_file_name:
                            targets = torch.from_numpy(np.stack([[labels[k]] for k in batch_input_idxs], axis=0))
                        else:
                            targets = batch['labels']
                        if 'supcon' in args.method:
                            # SupCon backward
                            emb = nlinear_model.forward_upto_classifier(inputs)
                            norm_emb = F.normalize(emb, p=2, dim=-1)
                            emb_projection = nlinear_model.projection(norm_emb)
                            emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                            if 'supconv2' in args.method:
                                if (use_supcon_pos) and (sc_num_samples is not None):
                                    greedy_features_index = [k for k in range(emb_projection.shape[0]) if k%args.num_samples==(args.num_samples-1)]
                                    if 'wp_all' in args.method:
                                        supcon1_loss = criterion_supcon1(emb_projection[:,None,:],torch.squeeze(targets).to(device)) # operates on all samples
                                    else:
                                        supcon1_loss = criterion_supcon1(emb_projection[greedy_features_index,None,:],torch.squeeze(targets[greedy_features_index]).to(device)) # operates on greedy samples only
                                    supcon2_loss = criterion_supcon2(emb_projection[:,None,:],torch.squeeze(targets).to(device)) # operates within prompt only
                                    supcon_loss = args.sc1_wgt*supcon1_loss + args.sc2_wgt*supcon2_loss
                                elif 'supconv2_reg_wp' in args.method:
                                    supcon1_loss = criterion_supcon1(emb_projection[:,None,:],torch.squeeze(targets).to(device)) # operates on all
                                    supcon2_loss = criterion_supcon2(emb_projection[:,None,:],torch.squeeze(targets).to(device)) # operates within prompt only
                                    supcon_loss = args.sc1_wgt*supcon1_loss + args.sc2_wgt*supcon2_loss
                                else:
                                    supcon_loss = criterion_supcon(emb_projection[:,None,:],torch.squeeze(targets).to(device))
                            else:
                                logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                                supcon_loss = criterion_supcon(logits, torch.squeeze(targets).to(device))
                            # print(supcon_loss.item())
                            epoch_supcon_loss += supcon_loss.item()
                            if ((use_supcon_pos) and (sc_num_samples is not None)) or 'supconv2_reg_wp' in args.method:
                                epoch_supcon1_loss += supcon1_loss.item()
                                epoch_supcon2_loss += supcon2_loss.item()
                            supcon_loss.backward()
                            # Cross Entropy backward
                            emb = nlinear_model.forward_upto_classifier(inputs).detach()
                            norm_emb = F.normalize(emb, p=2, dim=-1)
                            outputs = nlinear_model.classifier(norm_emb) # norm before passing here?
                            loss = criterion(outputs, targets.to(device).float())
                            loss.backward()
                        else:
                            outputs = nlinear_model(inputs)
                            loss = criterion(outputs, targets.to(device).float())
                            loss.backward()
                        optimizer.step()
                        scheduler.step()
                        epoch_train_loss += loss.item()
                    if 'supcon' in args.method: epoch_supcon_loss = epoch_supcon_loss/(step+1)
                    epoch_supcon1_loss = epoch_supcon1_loss/(step+1)
                    epoch_supcon2_loss = epoch_supcon2_loss/(step+1)
                    epoch_train_loss = epoch_train_loss/(step+1)

                    # Get val loss
                    nlinear_model.eval()
                    epoch_val_loss = 0
                    val_preds, val_true = [], []
                    for step,batch in enumerate(ds_val):
                        optimizer.zero_grad()
                        activations, batch_target_idxs = [], []
                        activations = my_train_acts[batch['inputs_idxs']].to(device)
                        inputs = activations
                        targets = batch['labels']
                        outputs = nlinear_model(inputs)
                        epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
                        val_preds_batch = torch.sigmoid(outputs.data)
                        val_preds += val_preds_batch.tolist()
                        val_true += targets.tolist()
                    epoch_val_loss = epoch_val_loss/(step+1)
                    epoch_val_auc = roc_auc_score(val_true, val_preds)
                    supcon_train_loss.append(epoch_supcon_loss)
                    supcon1_train_loss.append(epoch_supcon1_loss)
                    supcon2_train_loss.append(epoch_supcon2_loss)
                    train_loss.append(epoch_train_loss)
                    val_loss.append(epoch_val_loss)
                    val_auc.append(epoch_val_auc)
                    # print('Loss:', epoch_supcon_loss, epoch_train_loss, epoch_val_loss)
                    # Choose best model
                    best_model_state_using_last = deepcopy(nlinear_model.state_dict())
                    if epoch_val_auc > best_val_auc:
                        best_val_auc = epoch_val_auc
                        best_model_state_using_auc = deepcopy(nlinear_model.state_dict())
                    if epoch_val_loss < best_val_loss:
                            best_val_loss = epoch_val_loss
                            best_model_state_using_loss = deepcopy(nlinear_model.state_dict())
                    if args.best_using_auc:
                        best_model_state = best_model_state_using_auc
                    elif args.best_as_last:
                        best_model_state = best_model_state_using_last
                    else:
                        best_model_state = best_model_state_using_loss
                    
                    if args.save_probes:
                        probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_epoch{epoch}_model{i}'
                        torch.save(nlinear_model, probe_save_path)

                    # Early stopping
                    # patience, min_val_loss_drop, is_not_decreasing = 5, 0.01, 0
                    # if len(val_loss)>=patience:
                    #     for epoch_id in range(1,patience,1):
                    #         val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                    #         if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
                    #     if is_not_decreasing==patience-1: break

                print('\n\nEnd time of train:',datetime.datetime.now(),'\n\n')
                all_supcon_train_loss = np.array(supcon_train_loss)
                all_supcon1_train_loss = np.array(supcon1_train_loss)
                all_supcon2_train_loss = np.array(supcon2_train_loss)
                all_train_loss = np.array(train_loss)
                all_val_loss = np.array(val_loss)
                all_val_auc = np.array(val_auc)
                
                if args.save_probes:
                    nlinear_model.load_state_dict(best_model_state)
                    probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_model'
                    torch.save(nlinear_model, probe_save_path)

                    nlinear_model.load_state_dict(best_model_state_using_auc)
                    probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_bestusingauc_model'
                    torch.save(nlinear_model, probe_save_path)

                    nlinear_model.load_state_dict(best_model_state_using_last)
                    probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_bestusinglast_model'
                    torch.save(nlinear_model, probe_save_path)

                    nlinear_model.load_state_dict(best_model_state_using_loss)
                    probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_bestusingloss_model'
                    torch.save(nlinear_model, probe_save_path)
                
                nlinear_model.load_state_dict(best_model_state)
            
                # Val and Test performance
                print('\n\nStart time of val and test perf:',datetime.datetime.now(),'\n\n')
                pred_correct = 0
                y_val_pred, y_val_true = [], []
                val_preds = []
                val_logits = []
                val_sim = []
                with torch.no_grad():
                    nlinear_model.eval()
                    for step,batch in enumerate(ds_val):
                        activations = []
                        for k,idx in enumerate(batch['inputs_idxs']):
                            act = my_train_acts[idx].to(device)
                            activations.append(act)
                        inputs = torch.stack(activations,axis=0)
                        val_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                        y_val_true += batch['labels'].tolist()
                        val_preds.append(val_preds_batch)
                        val_logits.append(nlinear_model(inputs))
                val_preds = torch.cat(val_preds).cpu().numpy()
                all_val_preds = val_preds
                all_y_true_val = y_val_true
                all_val_logits = torch.cat(val_logits)
                print('Val AUROC:',"%.3f" % roc_auc_score(y_val_true, val_preds))
                best_val_t = get_best_threshold(y_val_true, val_preds)
                log_val_auc = roc_auc_score(y_val_true, val_preds)

                pred_correct = 0
                y_test_pred, y_test_true = [], []
                test_preds = []
                test_logits, test_wpdist = [], []
                test_sim = []
                with torch.no_grad():
                    nlinear_model.eval()
                    for step,batch in enumerate(ds_test):
                        activations = []
                        for k,idx in enumerate(batch['inputs_idxs']):
                            act = my_test_acts[idx].to(device)
                            activations.append(act)
                        inputs = torch.stack(activations,axis=0)
                        test_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                        if args.wp_dist:
                            outputs = nlinear_model.forward_upto_classifier(inputs)
                            wpdist_metric = args.wpdist_metric
                            if 'sc_proj' in args.wpdist_metric:
                                wpdist_metric = args.wpdist_metric.replace('_sc_proj','')
                                norm_emb = F.normalize(outputs, p=2, dim=-1)
                                outputs = nlinear_model.projection(norm_emb)
                            test_wpdist.append(compute_wp_dist(outputs,batch['labels'].tolist(),device,wpdist_metric))
                        y_test_true += batch['labels'].tolist()
                        test_preds.append(test_preds_batch)
                        test_logits.append(nlinear_model(inputs))
                test_preds = torch.cat(test_preds).cpu().numpy()
                all_test_preds = test_preds
                all_y_true_test = y_test_true
                print('AuROC:',"%.3f" % roc_auc_score(y_test_true, test_preds))
                log_test_auc = roc_auc_score(y_test_true, test_preds)
                all_test_logits = torch.cat(test_logits)
                if args.wp_dist: all_test_wpdist = torch.cat(test_wpdist)

                np.save(f'{args.save_path}/probes/{args.expt_name}_val_auc.npy', all_val_auc)
                np.save(f'{args.save_path}/probes/{args.expt_name}_val_loss.npy', all_val_loss)
                np.save(f'{args.save_path}/probes/{args.expt_name}_train_loss.npy', all_train_loss)
                np.save(f'{args.save_path}/probes/{args.expt_name}_supcon_train_loss.npy', all_supcon_train_loss)
                np.save(f'{args.save_path}/probes/{args.expt_name}_supcon1_train_loss.npy', all_supcon1_train_loss)
                np.save(f'{args.save_path}/probes/{args.expt_name}_supcon2_train_loss.npy', all_supcon2_train_loss)
                np.save(f'{args.save_path}/probes/{args.expt_name}_val_pred.npy', all_val_preds)
                np.save(f'{args.save_path}/probes/{args.expt_name}_val_f1.npy', all_val_f1s)
                np.save(f'{args.save_path}/probes/{args.expt_name}_val_true.npy', all_y_true_val)
                np.save(f'{args.save_path}/probes/{args.expt_name}_val_logits.npy', all_val_logits)

                np.save(f'{args.save_path}/probes/{args.expt_name}_test_pred.npy', all_test_preds)
                np.save(f'{args.save_path}/probes/{args.expt_name}_test_f1.npy', all_test_f1s)
                np.save(f'{args.save_path}/probes/{args.expt_name}_test_true.npy', all_y_true_test)
                np.save(f'{args.save_path}/probes/{args.expt_name}_test_logits.npy', all_test_logits)
                if args.wp_dist: np.save(f'{args.save_path}/probes/{args.expt_name}_test_wpdist_{args.wpdist_metric}.npy', all_test_wpdist)

                if args.wnb_plot_name is not None:
                    val_auc = np.load(f'{args.save_path}/probes/{args.expt_name}_val_auc.npy', allow_pickle=True).item()
                    val_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_val_loss.npy', allow_pickle=True).item()
                    train_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_train_loss.npy', allow_pickle=True).item()
                    if args.use_supcon_loss:
                        supcon_train_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_supcon_train_loss.npy', allow_pickle=True).item()
                    else:
                        supcon_train_loss = []
                    if (use_supcon_pos) and (sc_num_samples is not None):
                        supcon1_train_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_supcon1_train_loss.npy', allow_pickle=True).item()
                        supcon2_train_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_supcon2_train_loss.npy', allow_pickle=True).item()
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(val_loss, label='val_ce_loss')
                    plt.plot(train_loss, label='train_ce_loss')
                    plt.plot(supcon_train_loss, label='train_supcon_loss')
                    if (use_supcon_pos) and (sc_num_samples is not None):
                        plt.plot(supcon1_train_loss, label='train_supcon1_loss')
                        plt.plot(supcon2_train_loss, label='train_supcon2_loss')
                    plt.legend(loc="upper left")
                    plt.subplot(1, 2, 2)
                    plt.plot(val_auc, label='val_auc')
                    plt.legend(loc="upper left")
                    # plt.savefig(f'{args.save_path}/testfig.png')

                    wandb.init(
                    project="LLM-Hallu-Detection",
                    config={
                    "run_name": args.expt_name,
                    "model": args.model_name,
                    "dataset": test_dataset_name,
                    "act_type": args.using_act,
                    "token": args.token,
                    "method": args.method,
                    "bs": args.bs,
                    "lr": args.lr,
                    "tag": args.tag, #'design_choices',
                    "with_pe": args.use_pe
                    },
                    name=str(save_seed)+'-'+args.wnb_plot_name
                    )
                    tbl = wandb.Table(columns=['Val AUC', 'Test AUC'],
                                data=[[log_val_auc, log_test_auc]])
                    wandb.log({'chart': plt,
                                'metrics': tbl
                    })
                    wandb.finish()

if __name__ == '__main__':
    main()
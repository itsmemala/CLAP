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
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import wandb

from losses import SupConLoss
from utils import HF_NAMES  # Variables
from utils import list_of_ints, list_of_floats, list_of_strs, load_labels, load_acts, get_num_heads, get_act_dims, get_best_threshold # Functions
from utils import My_SupCon_NonLinear_Classifier4, LogisticRegression_Torch, Att_Pool_Layer # Classes


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
    parser.add_argument('--method', type=str, default='individual_linear')
    parser.add_argument('--use_dropout', type=bool, default=False)
    parser.add_argument('--no_bias', type=bool, default=False)
    parser.add_argument('--norm_emb', type=bool, default=False)
    parser.add_argument('--norm_cfr', type=bool, default=False)
    parser.add_argument('--cfr_no_bias', type=bool, default=False)
    parser.add_argument('--use_supcon_loss', type=bool, default=False)
    parser.add_argument('--supcon_temp', type=float, default=0.1)
    parser.add_argument('--len_dataset', type=int, default=5000)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--test_num_samples', type=int, default=1)
    parser.add_argument('--acts_per_file', type=int, default=100)
    parser.add_argument('--supcon_bs', type=int, default=128)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--supcon_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--supcon_lr', type=float, default=0.05)
    parser.add_argument('--lr',type=float, default=None)
    parser.add_argument('--scheduler', type=str, default='warmup_cosanneal')
    parser.add_argument('--best_using_auc', type=bool, default=False)
    parser.add_argument('--best_as_last', type=bool, default=False)
    parser.add_argument('--use_class_wgt', type=bool, default=False)
    parser.add_argument('--no_batch_sampling', type=bool, default=False)
    parser.add_argument('--save_probes', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--train_file_name", type=str, default=None)
    parser.add_argument("--test_file_name", type=str, default=None)
    parser.add_argument("--train_labels_file_name", type=str, default=None)
    parser.add_argument("--test_labels_file_name", type=str, default=None)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--seed_list', type=list_of_ints, default=42)
    parser.add_argument('--last_only', type=bool, default=False)
    parser.add_argument('--wnb_plot_name', type=str, default=None) # Wandb args
    parser.add_argument('--tag', type=str, default=None) # Wandb args
    args = parser.parse_args()

    device = "cuda"

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
    num_heads = get_num_heads(args.model_name)

    for seed_itr,save_seed in enumerate(args.seed_list):
        for lr in args.lr_list:
            print('Training SEED',save_seed)
            print('Training lr',lr)
            args.lr=lr
            args.expt_name = f'{args.expt_name}_lr{args.lr}_seed{save_seed}'            
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
            y_test = np.stack([test_labels[i] for i in test_idxs], axis = 0)
            
            loop_layers = range(my_train_acts.shape[1])
            for layer in tqdm(loop_layers):
                loop_heads = range(num_heads) if args.using_act == 'ah' else [0]
                for head in loop_heads:
                    cur_probe_train_set_idxs = train_set_idxs
                    val_set_idxs = val_set_idxs
                    cur_probe_y_train = np.stack([[labels[i]] for i in cur_probe_train_set_idxs], axis = 0)
                    y_val = np.stack([[labels[i]] for i in val_set_idxs], axis = 0)
                    train_target = np.stack([labels[j] for j in cur_probe_train_set_idxs], axis = 0)
                    class_sample_count = np.array([len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
                    weight = 1. / class_sample_count
                    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_target])).double()
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    ds_train = Dataset.from_dict({"inputs_idxs": cur_probe_train_set_idxs, "labels": cur_probe_y_train}).with_format("torch")
                    ds_train = DataLoader(ds_train, batch_size=args.bs, sampler=sampler) if args.no_batch_sampling==False else DataLoader(ds_train, batch_size=args.bs)
                    ds_val = Dataset.from_dict({"inputs_idxs": val_set_idxs, "labels": y_val}).with_format("torch")
                    ds_val = DataLoader(ds_val, batch_size=args.bs)
                    ds_test = Dataset.from_dict({"inputs_idxs": test_idxs, "labels": y_test}).with_format("torch")
                    ds_test = DataLoader(ds_test, batch_size=args.bs)

                    act_dims = get_act_dims(args.model_name)
                    bias = not args.no_bias
                    if args.method=='individual_linear':
                        nlinear_model = LogisticRegression_Torch(n_inputs=act_dims[args.using_act], n_outputs=1, bias=bias, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias).to(device) 
                    elif args.method=='individual_att_pool':
                        nlinear_model = Att_Pool_Layer(llm_dim=act_dims[args.using_act], n_outputs=1)
                    else:
                        nlinear_model = My_SupCon_NonLinear_Classifier4(input_size=act_dims[args.using_act], output_size=1, bias=bias, use_dropout=args.use_dropout, supcon=args.use_supcon_loss, norm_emb=args.norm_emb, norm_cfr=args.norm_cfr, cfr_no_bias=args.cfr_no_bias).to(device)
                    wgt_0 = np.sum(cur_probe_y_train)/len(cur_probe_y_train)
                    criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([wgt_0,1-wgt_0]).to(device)) if args.use_class_wgt else nn.BCEWithLogitsLoss()
                    criterion_supcon = SupConLoss(temperature=args.supcon_temp) if 'supconv2' in args.method else NTXentLoss()
                    
                    # Training
                    supcon_train_loss, train_loss, val_loss, val_auc = [], [], [], []
                    best_val_loss, best_spl_loss, best_val_auc = torch.inf, torch.inf, 0
                    best_model_state = deepcopy(nlinear_model.state_dict())
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    named_params = list(nlinear_model.named_parameters())
                    optimizer_grouped_parameters = [
                        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001, 'lr': args.lr},
                        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
                    ]
                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters) # torch.optim.Adam(optimizer_grouped_parameters)
                    steps_per_epoch = int(len(train_set_idxs)/args.bs)+1  # number of steps in an epoch
                    warmup_period = steps_per_epoch * 5
                    T_max = (steps_per_epoch*args.epochs) - warmup_period # args.epochs-warmup_period
                    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_period)
                    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)
                    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1) if args.scheduler=='static' else torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_period])
                    for epoch in range(args.epochs):
                        epoch_supcon_loss, epoch_train_loss, epoch_spl_loss = 0, 0, 0
                        nlinear_model.train()
                        for step,batch in enumerate(ds_train):
                            optimizer.zero_grad()
                            activations = []
                            for idx in batch['inputs_idxs']:
                                act = my_train_acts[idx][layer].to(device)
                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0) if args.token in single_token_types else torch.cat(activations,dim=0)
                            targets = batch['labels']
                            if 'supcon' in args.method:
                                # SupCon backward
                                emb = nlinear_model.forward_upto_classifier(inputs)
                                norm_emb = F.normalize(emb, p=2, dim=-1)
                                emb_projection = nlinear_model.projection(norm_emb)
                                emb_projection = F.normalize(emb_projection, p=2, dim=1) # normalise projected embeddings for loss calc
                                if 'supconv2' in args.method:
                                    supcon_loss = criterion_supcon(emb_projection[:,None,:],torch.squeeze(targets).to(device))
                                else:
                                    logits = torch.div(torch.matmul(emb_projection, torch.transpose(emb_projection, 0, 1)),args.supcon_temp)
                                    supcon_loss = criterion_supcon(logits, torch.squeeze(targets).to(device))
                                epoch_supcon_loss += supcon_loss.item()
                                supcon_loss.backward()
                                # Cross Entropy backward
                                emb = nlinear_model.forward_upto_classifier(inputs).detach()
                                norm_emb = F.normalize(emb, p=2, dim=-1)
                                outputs = nlinear_model.classifier(norm_emb) # norm before passing here?
                                loss = criterion(outputs, targets.to(device).float())
                                loss.backward()
                            else:
                                outputs = nlinear_model.linear(inputs) if 'individual_linear' in args.method else nlinear_model(inputs)
                                loss = criterion(outputs, targets.to(device).float())
                                loss.backward()
                            epoch_train_loss += loss.item()
                            optimizer.step()
                            scheduler.step()
                        if 'supcon' in args.method: epoch_supcon_loss = epoch_supcon_loss/(step+1)
                        epoch_train_loss = epoch_train_loss/(step+1)

                        
                        # Get val loss
                        nlinear_model.eval()
                        epoch_val_loss = 0
                        val_preds, val_true = [], []
                        for step,batch in enumerate(ds_val):
                            optimizer.zero_grad()
                            activations = []
                            for idx in batch['inputs_idxs']:
                                act = my_train_acts[idx][layer].to(device) 
                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0)
                            targets = batch['labels']
                            outputs = nlinear_model(inputs)
                            epoch_val_loss += criterion(outputs, targets.to(device).float()).item()
                            val_preds_batch = torch.sigmoid(nlinear_model(inputs).data) if args.token in single_token_types else torch.stack([torch.max(torch.sigmoid(nlinear_model(inp).data), dim=0)[0] for inp in inputs]) # For each sample, get max prob per class across tokens
                            val_preds += val_preds_batch.tolist()
                            val_true += batch['labels'].tolist()
                        epoch_val_loss = epoch_val_loss/(step+1)
                        epoch_val_auc = roc_auc_score(val_true, val_preds)
                        supcon_train_loss.append(epoch_supcon_loss)
                        train_loss.append(epoch_train_loss)
                        val_loss.append(epoch_val_loss)
                        val_auc.append(epoch_val_auc)
                        # print(epoch_spl_loss, epoch_supcon_loss, epoch_train_loss, epoch_val_loss)
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
                            probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_epoch{epoch}_model{i}_{layer}_{head}'
                            torch.save(nlinear_model, probe_save_path)

                        # Early stopping
                        # patience, min_val_loss_drop, is_not_decreasing = 5, 0.01, 0
                        # if len(val_loss)>=patience:
                        #     for epoch_id in range(1,patience,1):
                        #         val_loss_drop = val_loss[-(epoch_id+1)]-val_loss[-epoch_id]
                        #         if val_loss_drop > -1 and val_loss_drop < min_val_loss_drop: is_not_decreasing += 1
                        #     if is_not_decreasing==patience-1: break

                    all_supcon_train_loss = np.array(supcon_train_loss)
                    all_train_loss = np.array(train_loss)
                    all_val_loss = np.array(val_loss)
                    all_val_auc = np.array(val_auc)

                    if args.save_probes:
                        nlinear_model.load_state_dict(best_model_state)
                        probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_model_{layer}_{head}'
                        torch.save(nlinear_model, probe_save_path)

                        nlinear_model.load_state_dict(best_model_state_using_auc)
                        probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_bestusingauc_model_{layer}_{head}'
                        torch.save(nlinear_model, probe_save_path)

                        nlinear_model.load_state_dict(best_model_state_using_last)
                        probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_bestusinglast_model_{layer}_{head}'
                        torch.save(nlinear_model, probe_save_path)

                        nlinear_model.load_state_dict(best_model_state_using_loss)
                        probe_save_path = f'{args.save_path}/probes/models/{args.expt_name}_bestusingloss_model_{layer}_{head}'
                        torch.save(nlinear_model, probe_save_path)
                    
                    nlinear_model.load_state_dict(best_model_state)
                
                    # Val and Test performance
                    pred_correct = 0
                    y_val_pred, y_val_true = [], []
                    val_preds = []
                    val_logits = []
                    val_sim = []
                    with torch.no_grad():
                        nlinear_model.eval()
                        for step,batch in enumerate(ds_val):
                            activations = []
                            for idx in batch['inputs_idxs']:
                                act = my_train_acts[idx][layer].to(device)
                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                activations.append(act)
                            inputs = torch.stack(activations,axis=0)
                            val_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                            y_val_true += batch['labels'].tolist()
                            val_preds.append(val_preds_batch)
                            val_logits.append(nlinear_model(inputs))
                    all_val_logits = torch.cat(val_logits)
                    val_preds = torch.cat(val_preds).cpu().numpy()
                    all_val_preds = val_preds
                    all_y_true_val = y_val_true
                    
                    pred_correct = 0
                    y_test_pred, y_test_true = [], []
                    test_preds = []
                    test_logits = []
                    test_sim = []
                    with torch.no_grad():
                        nlinear_model.eval()
                        for step,batch in enumerate(ds_test):
                            activations = []
                            for idx in batch['inputs_idxs']:
                                act = my_test_acts[idx][layer].to(device)
                                if args.using_act=='ah': act = act[head*act_dims[args.using_act]:(head*act_dims[args.using_act])+act_dims[args.using_act]]
                                activations.append(act)
                            inputs = activations
                            test_preds_batch = torch.sigmoid(nlinear_model(inputs).data)
                            y_test_true += batch['labels'].tolist()
                            test_preds.append(test_preds_batch)
                            test_logits.append(nlinear_model(inputs))
                    test_preds = torch.cat(test_preds).cpu().numpy()
                    all_test_preds = test_preds
                    all_y_true_test = y_test_true
                    all_test_logits = torch.cat(test_logits)

            np.save(f'{args.save_path}/probes/{args.expt_name}_val_auc.npy', all_val_auc)
            np.save(f'{args.save_path}/probes/{args.expt_name}_val_loss.npy', all_val_loss)
            np.save(f'{args.save_path}/probes/{args.expt_name}_train_loss.npy', all_train_loss)
            np.save(f'{args.save_path}/probes/{args.expt_name}_supcon_train_loss.npy', all_supcon_train_loss)
            np.save(f'{args.save_path}/probes/{args.expt_name}_val_pred.npy', all_val_preds)
            np.save(f'{args.save_path}/probes/{args.expt_name}_val_true.npy', all_y_true_val)
            np.save(f'{args.save_path}/probes/{args.expt_name}_val_logits.npy', all_val_logits)
            
            np.save(f'{args.save_path}/probes/{args.expt_name}_test_pred.npy', all_test_preds)
            np.save(f'{args.save_path}/probes/{args.expt_name}_test_true.npy', all_y_true_test)
            np.save(f'{args.save_path}/probes/{args.expt_name}_test_logits.npy', all_test_logits)

            if args.wnb_plot_name is not None:
                val_auc = np.load(f'{args.save_path}/probes/{args.expt_name}_val_auc.npy', allow_pickle=True).item()
                val_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_val_loss.npy', allow_pickle=True).item()
                train_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_train_loss.npy', allow_pickle=True).item()
                if args.use_supcon_loss:
                    supcon_train_loss = np.load(f'{args.save_path}/probes/{args.expt_name}_supcon_train_loss.npy', allow_pickle=True).item()
                else:
                    supcon_train_loss = []

                if len(loop_layers)>1: 
                    val_loss = val_loss[-1] # Last layer only
                    train_loss = train_loss[-1] # Last layer only
                    supcon_train_loss = supcon_train_loss[-1] # Last layer only
                
                plt.subplot(1, 2, 1)
                plt.plot(val_loss, label='val_ce_loss')
                plt.plot(train_loss, label='train_ce_loss')
                plt.plot(supcon_train_loss, label='train_supcon_loss')
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
                "tag": args.tag #'design_choices'
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
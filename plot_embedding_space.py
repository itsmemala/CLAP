import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import json
from copy import deepcopy
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score, classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import llama
import argparse
from utils import LogisticRegression_Torch, tokenized_from_file
from utils import get_llama_activations_bau_custom, tokenized_mi, tokenized_from_file, tokenized_from_file_v2, get_token_tags

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'hl_llama_7B': 'huggyllama/llama-7b',
    'llama_2_7B': 'meta-llama/Llama-2-7b-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama_13B': 'huggyllama/llama-13b',
    'llama_30B': 'huggyllama/llama-30b',
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise'}

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='strqa')
    parser.add_argument('--using_act',type=str, default='mlp')
    parser.add_argument('--token',type=str, default='answer_last')
    parser.add_argument('--len_dataset',type=int, default=5000)
    parser.add_argument('--num_samples',type=int, default=None)
    parser.add_argument("--probes_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--train_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument("--test_labels_file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--plot_act',type=bool, default=False)
    parser.add_argument('--plot_aug',type=bool, default=False)
    parser.add_argument('--plot_3d',type=bool, default=False)
    parser.add_argument('--plot_pca',type=bool, default=False)
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--plot_name',type=str, default=None)
    args = parser.parse_args()

    device = 'cuda'

    # Load model
    nlinear_model = torch.load(f'{args.save_path}/probes/models/{args.probes_file_name}').to(device)

    MODEL = HF_NAMES[args.model_name] #if not args.model_dir else args.model_dir
    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)

    if args.dataset_name == 'gsm8k' or args.dataset_name == 'strqa' or ('baseline' in args.train_file_name or 'dola' in args.train_file_name):
        num_samples = args.num_samples if ('sampled' in args.train_file_name and args.num_samples is not None) else 9 if 'sampled' in args.train_file_name else 1
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file_v2(file_path, tokenizer, num_samples)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        labels = []
        num_samples_with_no_var = 0
        all_hallu_prompts, all_nh_prompts, hetero_prompts_sum = [], [], []
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            if 'baseline' in args.train_file_name or num_samples==1:
                # if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
                # if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
                label = 0 if data['is_correct'][i]==True else 1
                labels.append(label)
            else:
                sum_over_samples = 0
                for j in range(num_samples):
                    # if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i][j]==True else 0
                    # if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i][j]==True else 1
                    label = 0 if data['is_correct'][i][j]==True else 1
                    labels.append(label)
                    sum_over_samples += label
                if sum_over_samples==0 or sum_over_samples==num_samples: 
                    num_samples_with_no_var += 1
                    if sum_over_samples==num_samples: all_hallu_prompts.append(i)
                    if sum_over_samples==0: all_nh_prompts.append(i)
                else:
                    hetero_prompts_sum.append(sum_over_samples)
        labels = labels[:args.len_dataset]
    elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa' or args.dataset_name == 'tqa_gen':
        num_samples = args.num_samples if ('sampled' in args.train_file_name and args.num_samples is not None) else 11 if 'sampled' in args.train_file_name else 1
        file_path = f'{args.save_path}/responses/{args.train_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer, num_samples)
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = prompts[:args.len_dataset], tokenized_prompts[:args.len_dataset], answer_token_idxes[:args.len_dataset], prompt_tokens[:args.len_dataset]
        if 'se_labels' in args.train_labels_file_name:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.train_labels_file_name}.npy'
            labels = np.load(file_path)
        else:
            labels = []
            file_path = f'{args.save_path}/responses/{args.train_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.train_labels_file_name}.json'
            with open(file_path, 'r') as read_file:
                for line in read_file:
                    data = json.loads(line)
                    # for j in range(1,num_samples+1,1):
                    #     if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                    #     if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                    #     labels.append(label)
                    if 'greedy' in args.train_labels_file_name:
                        # if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                        # if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        labels.append(label)
                    else:
                        for j in range(1,num_samples+1,1):
                            # if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target_response'+str(j)]>0.3 else 0 # pos class is non-hallu
                            # if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                            label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                            labels.append(label)
        labels = labels[:args.len_dataset]
    if args.test_file_name is None:
        test_prompts, test_labels = [], [] # No test file
    elif 'gsm8k' in args.test_file_name or 'strqa' in args.test_file_name:
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file_v2(file_path, tokenizer)
        test_labels = []
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            # if 'hallu_pos' not in args.method: label = 1 if data['is_correct'][i]==True else 0
            # if 'hallu_pos' in args.method: label = 0 if data['is_correct'][i]==True else 1
            label = 0 if data['is_correct'][i]==True else 1
            test_labels.append(label)
    else:
        file_path = f'{args.save_path}/responses/{args.test_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_file_name}.json'
        test_prompts, test_tokenized_prompts, test_answer_token_idxes, test_prompt_tokens = tokenized_from_file(file_path, tokenizer)
        if 'se_labels' in args.test_labels_file_name:
            file_path = f'{args.save_path}/uncertainty/{args.model_name}_{args.test_labels_file_name}.npy'
            test_labels = np.load(file_path)
        else:
            test_labels = []
            file_path = f'{args.save_path}/responses/{args.test_labels_file_name}.json' if args.dataset_name == 'tqa_gen' else f'{args.save_path}/responses/{args.model_name}_{args.test_labels_file_name}.json'
            with open(file_path, 'r') as read_file:
                for line in read_file:
                    data = json.loads(line)
                    # if 'hallu_pos' not in args.method: label = 1 if data['rouge1_to_target']>0.3 else 0 # pos class is non-hallu
                    # if 'hallu_pos' in args.method: label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                    label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                    test_labels.append(label)

    fixed_indexes =  [46, 51, 54, 58] #[ 5, 11 ,23, 40 ,44, 62 ,71, 74, 85, 86, 89, 90]
    for idx in fixed_indexes:
        print(idx,test_prompts[idx],'\n')
    sys.exit()

    if args.dataset_name=='strqa':
        args.acts_per_file = 50
    elif args.dataset_name=='gsm8k':
        args.acts_per_file = 20
    else:
        args.acts_per_file = 100
    
    if 'strqa' in args.test_file_name:
        args.test_acts_per_file = 50
    elif 'gsm8k' in args.test_file_name:
        args.test_acts_per_file = 20
    else:
        args.test_acts_per_file = 100

    test_idxs = np.arange(len(test_labels))
    train_idxs = np.arange(args.len_dataset)

    # Load acts
    device_id, device = 0, 'cuda:0' # start with first gpu
    my_train_acts, my_test_acts = [], []
    # for idx in train_idxs:
        # file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
        # file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
        # if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
        #     # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        #     act = combine_acts(idx,args.train_file_name,args)
        #     if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
        #     if args.no_sep==False:
        #         sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
        #         act = torch.cat((act,sep_token), dim=1)
        #     act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
        # else:
        #     try:
        #         act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        #     except torch.cuda.OutOfMemoryError:
        #         device_id += 1
        #         device = 'cuda:'+str(device_id) # move to next gpu when prev is filled; test data load and rest of the processing can happen on the last gpu
        #         print('Loading on device',device_id)
        #         act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
        # my_train_acts.append(act)
    act_wise_file_paths, unique_file_paths = [], []
    for idx in train_idxs:
        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.train_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
        act_wise_file_paths.append(file_path)
        if file_path not in unique_file_paths: unique_file_paths.append(file_path)
    file_wise_data = {}
    for file_path in unique_file_paths:
        file_wise_data[file_path] = np.load(file_path,allow_pickle=True)
    for idx in train_idxs:
        if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
            # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.acts_per_file]).to(device)
            act = combine_acts(idx,args.train_file_name,args)
            if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
            if args.no_sep==False:
                sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                act = torch.cat((act,sep_token), dim=1)
            act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
        else:
            act = file_wise_data[act_wise_file_paths[idx]][idx%args.acts_per_file] #[-1]
        my_train_acts.append(act)
    my_train_acts = torch.from_numpy(np.stack(my_train_acts)).to(device)

    # if args.token=='tagged_tokens': my_train_acts = torch.nn.utils.rnn.pad_sequence(my_train_acts, batch_first=True)
    
    if args.test_file_name is not None:
        for idx in test_idxs:
            file_end = idx-(idx%args.test_acts_per_file)+args.test_acts_per_file # 487: 487-(87)+100
            test_dataset_name = args.test_file_name.split('_',1)[0].replace('nq','nq_open').replace('trivia','trivia_qa')
            file_path = f'{args.save_path}/features/{args.model_name}_{test_dataset_name}_{args.token}/{args.model_name}_{args.test_file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
            if args.token in ['prompt_last_and_answer_last','least_likely_and_last','prompt_last_and_least_likely_and_last']:
                # act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.test_acts_per_file]).to(device)
                act = combine_acts(idx,args.test_file_name,args)
                if args.tokens_first: act = torch.swapaxes(act, 0, 1) # (layers,tokens,act_dims) -> (tokens,layers,act_dims)
                if args.no_sep==False:
                    sep_token = torch.zeros(act.shape[0],1,act.shape[2]).to(device)
                    act = torch.cat((act,sep_token), dim=1)
                act = torch.reshape(act, (act.shape[0]*act.shape[1],act.shape[2])) # (layers,tokens,act_dims) -> (layers*tokens,act_dims)
            else:
                act = torch.from_numpy(np.load(file_path,allow_pickle=True)[idx%args.test_acts_per_file]).to(device) #[-1]
            my_test_acts.append(act)
        # if args.token=='tagged_tokens': my_test_acts = torch.nn.utils.rnn.pad_sequence(my_test_acts, batch_first=True)
    my_test_acts = torch.stack(my_test_acts)

    if args.plot_act:
        # TODO: norm input
        my_train_acts = torch.flatten(my_train_acts, start_dim=1).detach().cpu().numpy() # concatenate layers # my_train_acts.detach().cpu().numpy()
        my_test_acts = torch.flatten(my_test_acts, start_dim=1).detach().cpu().numpy() # concatenate layers # my_test_acts.detach().cpu().numpy()
        print(my_test_acts.shape)
        my_embs = np.concatenate([my_train_acts,my_test_acts],axis=0)
    else:
        nlinear_model.eval()
        my_train_embs = nlinear_model.forward_upto_classifier(my_train_acts).detach().cpu()
        if 'supcon' in args.probes_file_name: my_train_embs = F.normalize(my_train_embs, p=2, dim=-1)
        my_test_embs = nlinear_model.forward_upto_classifier(my_test_acts).detach().cpu()
        if 'supcon' in args.probes_file_name: my_test_embs = F.normalize(my_test_embs, p=2, dim=-1)
        my_embs = np.concatenate([my_train_embs.numpy(),my_test_embs.numpy()],axis=0)
        
    print(my_embs.shape)
    if 'sampled' in args.probes_file_name and args.plot_aug:
        train_labels_aug = []
        print(len(labels))
        for k,l in enumerate(labels):
            train_labels_aug.append(0 if (k%num_samples==(num_samples-1) and l==0) else 1 if (k%num_samples==(num_samples-1) and l==1) else 2 if (k%num_samples!=(num_samples-1) and l==0) else 3)
        my_plot_labels = train_labels_aug + [4 if l==0 else 5 for l in test_labels]
        my_plot_labels_dict = {0:'train_NH',1:'train_H',2:'train_NH_aug',3:'train_H_aug',4:'test_NH',5:'test_H'}
        my_cmap = colors.ListedColormap(['lightgreen','lightblue','orange','violet','darkgreen','darkblue'])
    else:
        my_plot_labels = labels + [2 if l==0 else 3 for l in test_labels]
        my_plot_labels_dict = {0:'train_NH',1:'train_H',2:'test_NH',3:'test_H'}
        my_cmap = colors.ListedColormap(['lightgreen','lightblue','darkgreen','darkblue'])
    my_plot_labels_name = [my_plot_labels_dict[l] for l in my_plot_labels]
    # my_plot_labels_cdict = {0:,1:,2:,3:}
    my_plot_labels_colors = my_plot_labels # [my_plot_labels_cdict[l] for l in my_plot_labels]
    clset = set(zip(my_plot_labels_colors, my_plot_labels_name))

    if args.plot_pca:
        pca = PCA(n_components=2) if args.plot_3d==False else PCA(n_components=3)
        X_transformed = pca.fit_transform(my_embs)
    else:
        tsne = TSNE(n_components=2, random_state=42) if args.plot_3d==False else TSNE(n_components=3, random_state=42)
        X_transformed = tsne.fit_transform(my_embs)
        print(tsne.kl_divergence_)
    fig, axs = plt.subplots(1,1) if args.plot_3d==False else plt.subplots(1,1,subplot_kw={'projection': '3d'})
    if 'sampled' in args.probes_file_name and args.plot_aug:
        X_plot = X_transformed
    elif 'sampled' in args.probes_file_name and args.plot_aug==False:
        greedy_idxs = np.array([(k*num_samples)+(num_samples-1) for k in range(int(args.len_dataset/num_samples))] + [len(my_train_acts)+k for k in range(len(my_test_acts))])
        X_plot = X_transformed[greedy_idxs]
        print(X_plot.shape)
        my_plot_labels_colors = np.array(my_plot_labels_colors)[greedy_idxs]
        my_plot_labels_name = np.array(my_plot_labels_name)[greedy_idxs]
    else:
        X_plot = X_transformed
    if args.plot_3d==False:
        sc = axs.scatter(x=X_plot[:, 0], y=X_plot[:, 1], c=my_plot_labels_colors, cmap= my_cmap) #label=my_plot_labels_name)
    else:
        sc = axs.scatter(xs=X_plot[:, 0], ys=X_plot[:, 1], zs=X_plot[:, 2], c=my_plot_labels_colors, cmap= my_cmap)
    handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c,l in clset ]
    labels = [l for c,l in clset]
    axs.legend(handles, labels)
    # fig.savefig(f'{args.save_path}/plotemb.png')
    args.plot_name += '_pca' if args.plot_pca else ''
    args.plot_name += '_3d' if args.plot_3d else ''
    fig.savefig(f'{args.save_path}/tsne_plots/{args.model_name}_{args.dataset_name}/{args.plot_name}.png')


if __name__ == '__main__':
    main()
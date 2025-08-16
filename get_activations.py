import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, get_llama_activations_bau_custom, get_token_tags, get_token_nll, get_num_layers
from utils import HF_NAMES, tokenized_from_file, tokenized_from_file_v2
# import llama
from transformers import AutoTokenizer
from base_transformers.models import llama3,gemma
import pickle
import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main(): 
    """
    Extract LLM activations.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default=None)
    parser.add_argument('dataset_name', type=str, default=None)
    parser.add_argument('--token',type=str, default='last')
    parser.add_argument('--mlp_l1',type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--num_samples',type=int, default=None)
    parser.add_argument('--acts_per_file',type=int, default=100)
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()
    device = "cuda"

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    if "llama3" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = llama3.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto").to(device)
        # model.forward = torch.compile(model.forward) #, mode="reduce-overhead") #, fullgraph=True)
    elif "gemma" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = gemma.GemmaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto").to(device)
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
        model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    num_layers = get_num_layers(args.model_name)

    print("Tokenizing prompts")
    if args.dataset_name == 'strqa' or args.dataset_name == 'gsm8k' or ('baseline' in args.file_name or 'dola' in args.file_name):
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file_v2(file_path, tokenizer, args.num_samples)
        np.save(f'{args.save_path}/responses/{args.model_name}_{args.file_name}_response_start_token_idx.npy', answer_token_idxes)
    elif args.dataset_name == 'nq_open' or args.dataset_name == 'cnn_dailymail' or args.dataset_name == 'trivia_qa' or args.dataset_name in ['city_country','movie_cast','player_date_birth']:
        file_path = f'{args.save_path}/responses/{args.model_name}_{args.file_name}.json'
        prompts, tokenized_prompts, answer_token_idxes, prompt_tokens = tokenized_from_file(file_path, tokenizer, args.num_samples)
        np.save(f'{args.save_path}/responses/{args.model_name}_{args.file_name}_response_start_token_idx.npy', answer_token_idxes)

    if 'tagged_tokens' in args.token:
        tagged_token_idxs = get_token_tags(prompts,prompt_tokens)
    else:
        tagged_token_idxs = [() for prompt in prompts]


    load_ranges = [(i*args.acts_per_file,(i*args.acts_per_file)+args.acts_per_file) for i in range(int(len(prompts)/args.acts_per_file)+1)] # Split save activations by prompt index due to disk space

    print(len(prompts))
    
    for start, end in load_ranges:
        all_layer_wise_activations = []
        all_head_wise_activations = []
        all_mlp_wise_activations = []
        all_mlp_l1_wise_activations = []
        all_attresoutput_wise_activations = []
        all_token_logprobs = []

        print("Getting activations for "+str(start)+" to "+str(end))
        for prompt,token_idx,tagged_idxs in tqdm(zip(tokenized_prompts[start:end],answer_token_idxes[start:end],tagged_token_idxs[start:end])):
            HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
            MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
            MLPS_L1 = [f"model.layers.{i}.mlp.up_proj_out" for i in range(model.config.num_hidden_layers)]
            ATT_RES_OUTS = [f"model.layers.{i}.att_res_out" for i in range(model.config.num_hidden_layers)]
            layer_wise_activations, head_wise_activations, mlp_wise_activations, mlp_l1_wise_activations, attresoutput_wise_activations = get_llama_activations_bau(model, prompt, device, HEADS=HEADS, MLPS=MLPS, MLPS_L1=MLPS_L1, ATT_RES_OUTS=ATT_RES_OUTS)
            if args.token=='answer_last': #last
                all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
                all_head_wise_activations.append(head_wise_activations[:,-1,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,-1,:])
                all_mlp_l1_wise_activations.append(mlp_wise_activations[:,-1,:])
                all_attresoutput_wise_activations.append(attresoutput_wise_activations[:,-1,:])
            elif args.token=='slt': #second last token
                all_layer_wise_activations.append(layer_wise_activations[:,-2,:])
                all_head_wise_activations.append(head_wise_activations[:,-2,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,-2,:])
            elif args.token=='prompt_last':
                all_layer_wise_activations.append(layer_wise_activations[:,token_idx-1,:])
                all_head_wise_activations.append(head_wise_activations[:,token_idx-1,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx-1,:])
            elif args.token=='least_likely':
                # print(print(tokenizer.decode(prompt[0], skip_special_tokens=True)))
                # print(tokenizer.decode(prompt[0], skip_special_tokens=True))
                least_likely_nll, least_likely_token_idx = 0, token_idx-1
                for next_token_idx in range(len(prompt[0][token_idx:])):
                    predicting_token_idx = token_idx+next_token_idx-1 # -1 since prob of every next token is given by prev token
                    predicted_token_id = prompt[0][token_idx+next_token_idx]
                    part_prompt = prompt[:,:predicting_token_idx]
                    # print(tokenizer.decode(part_prompt, skip_special_tokens=True))
                    nll = get_token_nll(model, part_prompt, device, predicted_token_id)
                    if nll > least_likely_nll:
                        least_likely_nll = nll
                        least_likely_token_idx = predicting_token_idx
                act = get_llama_activations_bau_custom(model, prompt, device, 'layer', -1, args.token, least_likely_token_idx)
                all_layer_wise_activations.append(act.numpy())
            elif args.token=='after_least_likely':
                # print(print(tokenizer.decode(prompt[0], skip_special_tokens=True)))
                # print(tokenizer.decode(prompt[0], skip_special_tokens=True))
                least_likely_nll, least_likely_token_idx = 0, token_idx-1
                for next_token_idx in range(len(prompt[0][token_idx:])):
                    predicting_token_idx = token_idx+next_token_idx-1 # -1 since prob of every next token is given by prev token
                    predicted_token_id = prompt[0][token_idx+next_token_idx]
                    part_prompt = prompt[:,:predicting_token_idx]
                    # print(tokenizer.decode(part_prompt, skip_special_tokens=True))
                    nll = get_token_nll(model, part_prompt, device, predicted_token_id)
                    if nll > least_likely_nll:
                        least_likely_nll = nll
                        least_likely_token_idx = predicting_token_idx + 1 # here, we want to look at generation of token after the least likely token
                act = get_llama_activations_bau_custom(model, prompt, device, 'layer', -1, args.token, least_likely_token_idx)
                all_layer_wise_activations.append(act.numpy())
            elif args.token=='random':
                # if len(prompt[0][token_idx:])==0: print(tokenizer.decode(prompt[0], skip_special_tokens=True))
                random_token_idx = token_idx-1 + np.random.choice(len(prompt[0][token_idx-1:]), 1)
                act = get_llama_activations_bau_custom(model, prompt, device, 'layer', -1, args.token, random_token_idx)
                all_layer_wise_activations.append(act.numpy())
            elif args.token=='prompt_last_and_answer_last':
                all_layer_wise_activations.append(np.stack((layer_wise_activations[:,token_idx-1,:],layer_wise_activations[:,-1,:]),axis=1))
                all_head_wise_activations.append(np.stack((head_wise_activations[:,token_idx-1,:],head_wise_activations[:,-1,:]),axis=1))
                all_mlp_wise_activations.append(np.stack((mlp_wise_activations[:,token_idx-1,:],mlp_wise_activations[:,-1,:]),axis=1))
            elif args.token=='maxpool_all':
                all_layer_wise_activations.append(np.max(layer_wise_activations,axis=1))
                all_head_wise_activations.append(np.max(head_wise_activations,axis=1))
                all_mlp_wise_activations.append(np.max(mlp_wise_activations,axis=1))
            elif args.token=='answer_first':
                all_layer_wise_activations.append(layer_wise_activations[:,token_idx,:])
                all_head_wise_activations.append(head_wise_activations[:,token_idx,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx,:])
            elif args.token=='answer_all':
                all_layer_wise_activations.append(layer_wise_activations[:,token_idx:,:])
                all_head_wise_activations.append(head_wise_activations[:,token_idx:,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx:,:])
            elif args.token=='all':
                all_layer_wise_activations.append(layer_wise_activations[:,:,:])
                all_head_wise_activations.append(head_wise_activations[:,:,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,:,:])
            elif args.token=='prompt_last_onwards':
                # all_layer_wise_activations.append(layer_wise_activations[:,:,:])
                all_head_wise_activations.append(head_wise_activations[:,token_idx-1:,:])
                all_mlp_wise_activations.append(mlp_wise_activations[:,token_idx-1:,:])
                all_attresoutput_wise_activations.append(attresoutput_wise_activations[:,token_idx-1:,:])
            elif args.token=='tagged_tokens' or args.token=='tagged_tokens_and_last':
                acts = []
                for layer in range(num_layers):
                    act = get_llama_activations_bau_custom(model, prompt, device, 'layer', layer, args.token, token_idx, tagged_idxs)
                    acts.append(act)
                # print(len(acts),acts[0].shape)
                acts = torch.stack(acts)
                all_layer_wise_activations.append(acts)
            # token_logprobs = []
            # for next_token_idx in range(len(prompt[0][token_idx:])):
            #     predicting_token_idx = token_idx+next_token_idx-1 # -1 since prob of every next token is given by prev token
            #     predicted_token_id = prompt[0][token_idx+next_token_idx]
            #     part_prompt = prompt[:,:predicting_token_idx]
            #     # print(tokenizer.decode(part_prompt, skip_special_tokens=True))
            #     token_logprobs.append(-get_token_nll(model, part_prompt, device, predicted_token_id)) # apply neg to match sign returned by openai API for token logprobs
            # all_token_logprobs.append(token_logprobs)
        
        # #     break
        # # break

        print("Saving layer wise activations")
        if 'tagged_tokens' in args.token:
            with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_layer_wise_{end}.pkl', 'wb') as outfile:
                torch.save(all_layer_wise_activations, outfile, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_layer_wise_{end}.npy', all_layer_wise_activations)
            with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_layer_wise_{end}.pkl', 'wb') as outfile:
                pickle.dump(all_layer_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)
        
        print("Saving head wise activations")
        # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_head_wise_{end}.npy', all_head_wise_activations)
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_head_wise_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_head_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

        print("Saving mlp wise activations")
        # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_mlp_wise_{end}.npy', all_mlp_wise_activations)
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_mlp_wise_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_mlp_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

        print("Saving mlp l1 activations")
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_mlp_l1_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_mlp_l1_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

        print("Saving att res out wise activations")
        # np.save(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}_mlp_wise_{end}.npy', all_mlp_wise_activations)
        with open(f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{args.file_name}_{args.token}_attresout_wise_{end}.pkl', 'wb') as outfile:
            pickle.dump(all_attresoutput_wise_activations, outfile, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
import os
from tqdm import tqdm
import numpy as np
import pickle
import json
import jsonlines
import random
import argparse
from multiprocessing import Pool

import datasets
from datasets import load_dataset
import evaluate
import torch
import llama
from transformers import AutoTokenizer

from utils import HF_NAMES # Variables


def my_func(args):
    id_,tokenizer,model,prompt,sample,device = args
    tokenized_input = tokenizer([prompt+sample], return_tensors = 'pt').input_ids.to(device)
    tokenized_prompt = tokenizer([prompt], return_tensors = 'pt').input_ids
    target_ids = tokenized_input.clone().to(device)
    target_ids[0][:len(tokenized_prompt[0])] = -100
    model_output = model(tokenized_input, labels=target_ids, output_hidden_states=True)
    average_neg_log_likelihood = model_output['loss'] # len normalised predictive entropy
    neg_log_likelihood = average_neg_log_likelihood * (len(tokenized_input[0]) - len(tokenized_prompt[0])) # sequence predictive entropy
    return id_, np.array([average_neg_log_likelihood.detach().cpu().numpy(),neg_log_likelihood.detach().cpu().numpy()])


def main(): 
    """"
    Extract Uncertainty of LLM responses.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='trivia_qa')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_processes', type=int, default=2)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_cache_dir", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    device='cuda'

    print('Loading model..')
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
    device = "cuda"

    print('Loading model responses..')
    prompts, responses = [], []
    if 'strqa' in args.dataset_name or 'gsm8k' in args.dataset_name:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.file_name}.json', 'r') as read_file:
            data = json.load(read_file)
        if 'baseline' in args.file_name or args.num_samples==1:
            for i in range(len(data['full_input_text'])):
                prompts.append(data['full_input_text'][i])
                response = data['model_completion'][i] if 'strqa' in args.dataset_name else data['model_answer'][i] # For strqa, we want full COT response
                responses.append(response)
        else:
            for i in range(len(data['full_input_text'])):
                prompts.append(data['full_input_text'][i][0])
                samples = []
                for j in range(args.num_samples):
                    response = data['model_completion'][i][j] if 'strqa' in args.dataset_name else data['model_answer'][i][j] # For strqa, we want full COT response
                    samples.append(response)
                responses.append(samples)
    else:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.file_name}.json', 'r') as read_file:
            for line in read_file:
                data = json.loads(line)
                prompts.append(data['prompt'])
                if 'greedy' in args.file_name:
                    responses.append(data['response1'])
                else:
                    samples = []
                    for i in range(1,args.num_samples+1,1):
                        samples.append(data['response'+str(i)])
                    responses.append(samples)

    
    print('Getting token probability scores..')
    # Get token probabilities
    scores = []
    my_iter = -1
    for prompt,response in tqdm(zip(prompts,responses)):
        my_iter += 1
        if my_iter%50==0: print(my_iter)
        if 'greedy' in args.file_name or 'baseline' in args.file_name:
            tokenized_input = tokenizer([prompt+response], return_tensors = 'pt').input_ids.to(device)
            # tokenized_input = tokenized_input[tokenized_input != tokenizer.pad_token_id]
            tokenized_prompt = tokenizer([prompt], return_tensors = 'pt').input_ids
            # tokenized_prompt = tokenized_prompt[tokenized_prompt != tokenizer.pad_token_id]
            # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
            target_ids = tokenized_input.clone().to(device)
            target_ids[0][:len(tokenized_prompt[0])] = -100
            # model_output = model(torch.reshape(tokenized_input, (1, -1)), labels=target_ids, output_hidden_states=True)
            model_output = model(tokenized_input, labels=target_ids, output_hidden_states=True)
            average_neg_log_likelihood = model_output['loss'] # len normalised predictive entropy
            neg_log_likelihood = average_neg_log_likelihood * (len(tokenized_input[0]) - len(tokenized_prompt[0])) # sequence predictive entropy
            scores.append(np.array([average_neg_log_likelihood.detach().cpu().numpy(),neg_log_likelihood.detach().cpu().numpy()]))
        else:
            sample_wise_score = [] #np.empty((len(response),len(response[0])))
            pool = Pool(processes=args.num_processes)  # You can adjust the number of processes
            tasks = [(id_,tokenizer,model,prompt,sample,device) for id_,sample in enumerate(response)]
            results = pool.imap_unordered(my_func, tasks)
            pool.close()
            pool.join()
            for id_, val in results:
                sample_wise_score.append(val) #sample_wise_score[id_] = val
            scores.append(np.stack(sample_wise_score)) # scores.append(sample_wise_score)
            

    print('Saving token probability scores..')
    np.save(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.file_name}_uncertainty_scores.npy', scores)
    

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
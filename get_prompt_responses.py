import os
import datasets
import datetime, time
from tqdm import tqdm
from collections import Counter
import pickle
import json
import jsonlines
import random
import re, string
import zipfile
import argparse

from datasets import load_dataset
import numpy as np
import torch
# import llama
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from transformers import AutoTokenizer
from base_transformers.models import llama3,gemma
import evaluate

from config import ANS_RE, INVALID_ANS, N_SHOT, COT_FLAG, DEBUG, ANSWER_TRIGGER, SHORT_ANSWER_TRIGGER
from utils import HF_NAMES # Variables
from utils import load_jsonl, download_url, is_correct, build_prompt, clean_answer, load_jsonl_gsm8k, is_correct_gsm8k, create_demo_text_gsm8k, clean_answer_gsm8k, my_squad_f1_score # Functions


def main(): 
    """
    Extract LLM responses.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='trivia_qa')
    parser.add_argument('--len_dataset', type=int, default=0)
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--use_split', type=str, default='train')
    parser.add_argument('--hallu_check_prompt', type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--num_ret_seq', type=int, default=1)
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument('--device', type=int, default=0)    
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_cache_dir", type=str, default=None)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBILE_DEVICES"] = "0,1,2,3"
    kwargs = InitProcessGroupKwargs()
    kwargs.timeout = datetime.timedelta(seconds=7200)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    print('Loading model..')
    if "llama3" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = llama3.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map={"": accelerator.process_index})
    elif "gemma" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = gemma.GemmaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map={"": accelerator.process_index})
    else:
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
        model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map={"": accelerator.process_index})
    if args.num_ret_seq>1 and args.model_name=='llama_2_7B': model = model.bfloat16() # Numerical instability; Solution from: https://github.com/meta-llama/llama/issues/380
    device = accelerator.device
    # device = 'cpu' # for debugging
    # model = model.cpu()

    print('Loading data..')
    print(args.len_dataset,args.start_at,args.use_split)
    len_dataset = args.len_dataset
    start_at = args.start_at
    if args.dataset_name=='strqa':
        fp = os.path.join(args.save_path, 'strategyqa_train.json')
        if not os.path.exists(fp):
            download_url(
                'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip', args.save_path)
            # Once the file is downloaded, unzip it
            with zipfile.ZipFile(os.path.join(args.save_path, 'strategyqa_dataset.zip'), 'r') as zip_ref:
                zip_ref.extractall(args.save_path)
        list_data_dict = load_jsonl(fp)
        all_input_texts, all_gt_answers, tokenized_prompts = [], [], []
        for sample in list_data_dict:
            all_gt_answers.append(sample['answer'])
            input_text = build_prompt(sample['question'], N_SHOT, COT_FLAG, args.do_shuffle, args.dataset_name)
            all_input_texts.append(input_text)
            tokenized_prompt = tokenizer(input_text, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
    elif args.dataset_name=='gsm8k':
        fp = os.path.join(args.save_path, 'gsm8k_'+args.use_split+'.json')
        download_path = 'https://raw.githubusercontent.com/openai/'\
                +'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'\
                +'grade_school_math/data/'+args.use_split+'.jsonl'
        if not os.path.exists(fp):
            download_url(download_path, args.save_path)
            os.rename(args.save_path+'/'+args.use_split+'.jsonl', fp)
        list_data_dict = load_jsonl_gsm8k(fp, instruction='question', output='answer')
        all_input_texts, all_gt_answers, tokenized_prompts = [], [], []
        for sample in list_data_dict[:args.len_dataset]:
            all_gt_answers.append(sample['output'])
            input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG, args.do_shuffle, args.dataset_name)
            all_input_texts.append(input_text)
            tokenized_prompt = tokenizer(input_text, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
    elif args.dataset_name in ['city_country','movie_cast','player_date_birth']:
        prompts = []
        tokenized_prompts = []
        all_gt_answers = []
        with open(f'{args.dataset_name}.json') as f:
            file_data = json.load(f)
        train_len = int(0.8*len(file_data))
        if args.use_split=='train' and args.len_dataset==0:
            args.len_dataset = train_len
            start_row, end_row = 0, train_len
        elif args.use_split=='train':
            start_row, end_row = 0, args.len_dataset
        else:
            args.len_dataset = len(file_data) - train_len
            start_row, end_row = train_len, len(file_data)
        for i in range(start_row,end_row,1):
            prompt_question = file_data[i]['prompt']
            cur_prompt = prompt_question
            prompts.append(cur_prompt)
            tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
            all_gt_answers.append(file_data[i]['correct_answer'])
    else:
        if args.dataset_name=='nq_open':
            hf_dataset_name = 'nq_open'
            dataset = load_dataset(hf_dataset_name, streaming= True)[args.use_split]
        elif args.dataset_name=='trivia_qa':
            hf_dataset_name = 'mandarjoshi/trivia_qa'
            dataset = load_dataset(hf_dataset_name, 'rc.nocontext', streaming= True)[args.use_split]
        elif args.dataset_name=='cnn_dailymail':
            hf_dataset_name = 'cnn_dailymail'
            dataset = load_dataset(hf_dataset_name, streaming= True)[args.use_split]
        if args.hallu_check_prompt is None:
            prompts = []
            tokenized_prompts = []
            for idx,val in enumerate(list(dataset.take(len_dataset))[start_at:]):
                if args.dataset_name=='nq_open':
                    question = val['question']
                    cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
                elif args.dataset_name=='trivia_qa':
                    question = val['question']
                    cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
                elif args.dataset_name=='cnn_dailymail':
                    article = val['article']
                    cur_prompt = f"Article: {article}\n Summarize the article in two to three sentences. Summary: "
                prompts.append(cur_prompt)
                tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
                tokenized_prompts.append(tokenized_prompt)
        else:
            # Load greedy responses
            greedy_resp_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_greedy_responses_{args.use_split}{args.len_dataset}.json'
            with open(greedy_resp_fname, 'r') as read_file:
                greedy_resp_data = []
                for line in read_file:
                    greedy_resp_data.append(json.loads(line))
            prompts = []
            tokenized_prompts = []
            for row,val in zip(greedy_resp_data,list(dataset.take(len_dataset))[start_at:]):
                if args.hallu_check_prompt==1:
                    cur_prompt = row['prompt'] + row['response1'] + "\n The above generated answer is incorrect. Revised answer: "
                if args.hallu_check_prompt==2:
                    cur_prompt = row['prompt'] + row['response1'] + "\n The above answer may be incorrect. The actual correct answer is: "
                if args.hallu_check_prompt==3 and (args.dataset_name=='trivia_qa' or args.dataset_name=='nq_open'):
                    question = val['question']
                    cur_prompt = f"This is a bot that correctly answers questions. Consider the below question and a possible answer, which may or may not be correct. Provide the correct answer to the question. \n Q: {question} Possible answer: {row['response1']}\n Correct answer:"
                prompts.append(cur_prompt)
                tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
                tokenized_prompts.append(tokenized_prompt)
    
    print('Getting model responses..')
    # Get model responses
    responses = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []} #, 'raw_model_generation': []}
    correct_rate = 0
    oom_err_idxs = []
    if args.dataset_name=='strqa':
        period_token_id = tokenizer("\n")['input_ids']
        eos_tokens = ["Q:", "\n\n##"]
        checkgens = ["Q:", "\n\n##"]
    if args.dataset_name=='gsm8k':
        period_token_id = None
        eos_tokens = ["Q:", "\end{code}"]
        checkgens = ["Q:", "\end{code}"]
    elif args.dataset_name=='nq_open' or args.dataset_name=='trivia_qa' or args.dataset_name in ['city_country','movie_cast','player_date_birth']:
        period_token_id = tokenizer('.')['input_ids']
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                        'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\"]
        checkgens = ['QA2:','Q.', 'B:']
    elif args.dataset_name=='cnn_dailymail':
        period_token_id = tokenizer('\n')['input_ids']
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                        'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\", 'Summary:', ' Summary:']
        checkgens = ['Summary:']
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in eos_tokens]
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start=time.time()
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes({'tokenized_prompts':tokenized_prompts,'all_gt_answers':all_gt_answers}) as split_dict:
        tokenized_prompts_split, all_gt_answers_split = split_dict['tokenized_prompts'], split_dict['all_gt_answers']
        for i,tokenized_prompt in enumerate(tqdm(tokenized_prompts_split)):
            tokenized_prompt = tokenized_prompt.to("cuda")#.to(device)
            response = model.generate(tokenized_prompt, max_new_tokens=512,
                                        temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample, num_return_sequences=args.num_ret_seq,
                                        eos_token_id=period_token_id,
                                        bad_words_ids=question_framing_ids + [tokenized_prompt.tolist()[0]]
                                        )[:, tokenized_prompt.shape[-1]:]
            if args.num_ret_seq==1:
                if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
                    cur_response = tokenizer.decode(response[0], skip_special_tokens=True)
                    for check_gen in checkgens: # Fix generation stopping errors
                        cur_response = cur_response.split(check_gen)[0]
                    model_completion = cur_response
                    cur_model_answer = clean_answer(cur_response) if args.dataset_name=='strqa' else clean_answer_gsm8k(cur_response)
                    model_answer = cur_model_answer
                    is_cor = is_correct(cur_model_answer, all_gt_answers[i]) if args.dataset_name=='strqa' else is_correct_gsm8k(cur_model_answer, all_gt_answers[i])
                    input_text = all_input_texts[i]
                    correct_rate += is_cor
                    print('\n# Correct answers:',correct_rate,'\n')
                    result_dict['is_correct'].append(is_cor)
                    result_dict['model_answer'].append(model_answer)
                    result_dict['model_completion'].append(model_completion)
                    result_dict['full_input_text'].append(input_text)
                    results=[result_dict]
                else:
                    response = tokenizer.decode(response[0], skip_special_tokens=True)
                    for check_gen in checkgens: # Fix generation stopping errors
                        response = response.split(check_gen)[0]
                    responses.append({'prompt':prompts[i],
                                        'response1':response})
                    results=responses
            else:
                if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
                    is_cor, model_answer, model_completion, input_text = [], [], [], []
                    for j in range(args.num_ret_seq):
                        cur_response = tokenizer.decode(response[j], skip_special_tokens=True)
                        for check_gen in checkgens: # Fix generation stopping errors
                            cur_response = cur_response.split(check_gen)[0]
                        model_completion.append(cur_response)
                        cur_model_answer = clean_answer(cur_response) if args.dataset_name=='strqa' else clean_answer_gsm8k(cur_response)
                        model_answer.append(cur_model_answer)
                        is_cor.append(is_correct(cur_model_answer, all_gt_answers_split[i]) if args.dataset_name=='strqa' else is_correct_gsm8k(cur_model_answer, all_gt_answers_split[i]))
                        input_text.append(all_input_texts[i])
                    correct_rate += sum(is_cor)
                    print('\n# Correct answers:',correct_rate,'\n')
                    result_dict['is_correct'].append(is_cor)
                    result_dict['model_answer'].append(model_answer)
                    result_dict['model_completion'].append(model_completion)
                    result_dict['full_input_text'].append(input_text)
                    results=[result_dict]
                else:
                    resp_dict = {'prompt':prompts[i]}
                    for j in range(args.num_ret_seq):
                        cur_response = tokenizer.decode(response[j], skip_special_tokens=True)
                        for check_gen in checkgens: # Fix generation stopping errors
                            cur_response = cur_response.split(check_gen)[0]
                        resp_dict['response'+str(j+1)] = cur_response
                    responses.append(resp_dict)
                    results=responses
        timediff=time.time()-start
        print("GPU {}: {} prompts received, generated in {} seconds".format(
            accelerator.process_index,
            len(tokenized_prompts_split),
            timediff,
            ))
    # collect results from all the GPUs
    results_gathered=gather_object(results)
    # print('\n\n',results_gathered)
    
    if accelerator.is_main_process:
        # print('\n\n',results_gathered)
        print('Saving model responses..')
        if args.hallu_check_prompt is None:
            gen_type = 'sampled' if args.do_sample else 'greedy'
            save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}.json'
        else:
            save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_{args.use_split}{args.len_dataset}.json'
        
        if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
            result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []} #, 'raw_model_generation': []}
            for k in range(len(results_gathered)): # k = num of gpus/processes
                result_dict['is_correct'] += results_gathered[k]['is_correct']
                result_dict['model_answer'] += results_gathered[k]['model_answer']
                result_dict['model_completion'] += results_gathered[k]['model_completion']
                result_dict['full_input_text'] += results_gathered[k]['full_input_text']
            with open(save_fname, 'w') as f:
                json.dump(result_dict, f)
            np.save(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}_oom_idxs.npy',oom_err_idxs)
        else:
            responses=results_gathered
            with open(save_fname, 'w') as outfile:
                for entry in responses:
                    json.dump(entry, outfile)
                    outfile.write('\n')
        
        print('Getting labels for model responses..')
        labels = []
        if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
            pass # Labels have already been generated above
        else:
            if args.dataset_name in ['city_country','movie_cast','player_date_birth']:
                for i in range(len(responses)):
                    if args.num_ret_seq==1:
                        labels_dict = {'rouge1_to_target':0.0} # Using "rouge" only for consistency with rest of code for trivia/nq
                    else:
                        labels_dict = {}
                        for j in range(args.num_ret_seq):
                            labels_dict['rouge1_to_target_response'+str(j+1)]=0.0
                    for j in range(args.num_ret_seq):
                        cur_response = responses[i]['response'+str(j+1)]
                        resp_wise_label_name = '_response'+str(j+1) if args.num_ret_seq>1 else ''
                        if 'city_country' in args.dataset_name:
                            labels_dict['rouge1_to_target' + resp_wise_label_name] = int(check_name_correctness(cur_response,all_gt_answers[i]))
                        elif 'player_date_birth' in args.dataset_name:
                            labels_dict['rouge1_to_target' + resp_wise_label_name] = int(string_match(cur_response,all_gt_answers[i]))
                        elif 'movie_cast' in args.dataset_name:
                            labels_dict['rouge1_to_target' + resp_wise_label_name] = int(string_match_in_list(cur_response,all_gt_answers[i]))
                    labels.append(labels_dict)
            else:
                rouge = evaluate.load('rouge')
                exact_match_metric = evaluate.load("exact_match")
                squad_metrics = evaluate.load('squad')
                for i,batch in tqdm(enumerate(list(dataset.take(args.len_dataset))[start_at:])): # one row at a time
                    if args.num_ret_seq==1:
                        labels_dict = {'exact_match': 0.0,
                                        'rouge1_to_target':0.0,
                                        'rouge2_to_target':0.0,
                                        'rougeL_to_target':0.0,
                                        'squad_f1':0.0}
                    else:
                        labels_dict = {}
                        for j in range(args.num_ret_seq):
                            labels_dict['exact_match_response'+str(j+1)]=0.0
                            labels_dict['rouge1_to_target_response'+str(j+1)]=0.0
                            labels_dict['rouge2_to_target_response'+str(j+1)]=0.0
                            labels_dict['rougeL_to_target_response'+str(j+1)]=0.0
                            labels_dict['squad_f1_response'+str(j+1)]=0.0
                    if args.dataset_name=='nq_open':
                        reference_answers = batch['answer'] 
                    elif args.dataset_name=='trivia_qa':
                        reference_answers_unformatted = batch['answer']
                        reference_answers = reference_answers_unformatted['aliases'] + reference_answers_unformatted['normalized_aliases'] # [reference_answers_unformatted['normalized_value']]
                    elif args.dataset_name=='cnn_dailymail':
                        reference_answers = [batch['highlights']]
                    for answer in reference_answers:
                        for j in range(args.num_ret_seq):
                            resp_wise_label_name = '_response'+str(j+1) if args.num_ret_seq>1 else ''
                            predictions = [responses[i]['response'+str(j+1)].lstrip()]
                            references = [answer]
                            results = exact_match_metric.compute(predictions=predictions,
                                                                    references=references,
                                                                    ignore_case=True,
                                                                    ignore_punctuation=True)
                            labels_dict['exact_match' + resp_wise_label_name] = max(results['exact_match'], labels_dict['exact_match' + resp_wise_label_name])
                            rouge_results = rouge.compute(predictions=predictions, references=references)
                            for rouge_type in ['rouge1','rouge2','rougeL']:
                                labels_dict[rouge_type + '_to_target' + resp_wise_label_name] = max(rouge_results[rouge_type],
                                                                                labels_dict[rouge_type + '_to_target' + resp_wise_label_name])
                            squad_f1 = my_squad_f1_score(predictions[0],references[0])
                            labels_dict['squad_f1' + resp_wise_label_name] = max(squad_f1, labels_dict['squad_f1' + resp_wise_label_name])
                    labels.append(labels_dict)


            print('Saving labels..')
            if args.hallu_check_prompt is None:
                save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_labels_{args.use_split}{args.len_dataset}.json'
            else:
                save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_labels_{args.use_split}{args.len_dataset}.json'
            with open(save_fname, 'w') as outfile:
                for entry in labels:
                    json.dump(entry, outfile)
                    outfile.write('\n')
    

if __name__ == '__main__':
    main()
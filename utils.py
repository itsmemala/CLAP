import os
import sys
import json
import math
import numpy as np
import random
import ssl
from tqdm import tqdm
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import llama
from baukit import Trace, TraceDict
from sklearn.linear_model import LogisticRegression
import spacy

from config import ANS_RE, INVALID_ANS, N_SHOT, COT_FLAG, DEBUG, ANSWER_TRIGGER, SHORT_ANSWER_TRIGGER

torch.set_default_dtype(torch.float64)

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
    'flan_33B': 'timdettmers/qlora-flan-33b',
    'llama3.1_8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1_8B_Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'gemma_2B': 'google/gemma-2b',
    'gemma_7B': 'google/gemma-7b'
}

class Att_Pool_Layer(torch.nn.Module):    
    # build the constructor
    def __init__(self, llm_dim, n_outputs, device='cuda'):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(llm_dim)).to(device).to(torch.float64)
        self.classifier = torch.nn.Linear(llm_dim, n_outputs, bias=False).to(device).to(torch.float64)

    # make predictions
    def forward(self, x): # x: (bs, llm_dim)
        x = self.forward_upto_classifier(x)
        y_pred = self.classifier(x)
        return y_pred
    
    def forward_upto_classifier(self, x): # x: (bs, n_tokens, llm_dim)
        qt_h = torch.matmul(x,self.query) # qt_h: (bs, n_tokens)
        att_wgts = nn.functional.softmax(qt_h, dim=-1)  # att_wgts: (bs, n_tokens)
        att_out = []
        for sample in range(x.shape[0]):
            att_out.append(torch.matmul(att_wgts[sample],x[sample]))  # att_out: (llm_dim)
        att_out = torch.stack(att_out) # att_out: (bs, llm_dim)
        return att_out

class My_Projection_w_Classifier_Layer(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_layers, n_outputs, bias, batch_norm=False, supcon=False, norm_emb=False, norm_cfr=False, cfr_no_bias=False, d_model=128, no_act_proj=False, non_linear=False, device='cuda'):
        super().__init__()
        self.no_act_proj = no_act_proj
        d_model = d_model #128 # 256
        self.batch_norm = batch_norm
        self.linear = torch.nn.Linear(n_inputs, d_model, bias)
        self.batch_norm_layer = torch.nn.BatchNorm1d(n_layers+1)
        self.supcon=supcon
        self.norm_emb=norm_emb
        self.norm_cfr=norm_cfr
        if non_linear:
            self.non_linear = True
            self.linear1 = nn.Linear(n_layers*d_model, 256)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(128, 64)
            self.relu3 = nn.ReLU()
            self.classifier = torch.nn.Linear(64, n_outputs, bias=not cfr_no_bias)
        else:
            self.non_linear = False
            self.classifier = torch.nn.Linear(n_layers*d_model, n_outputs, bias=not cfr_no_bias)

    # make predictions
    def forward(self, x): # x: (bs, n_layers, n_inputs)
        x = self.forward_upto_classifier(x)
        if self.supcon or self.norm_emb: x = F.normalize(x, p=2, dim=-1) # unit normalise, setting dim=-1 since inside forward() we define ops for one sample only
        if self.norm_cfr and self.training==False:
            norm_cfr_wgts = F.normalize(self.classifier.weight, p=2, dim=-1)
            y_pred = torch.sum(x * norm_cfr_wgts, dim=-1)
            y_pred = (y_pred + 1)/2 # re-scale to yield probability values
            assert y_pred.min().item()>=0 and y_pred.max().item()<=1
            return y_pred[:,None] # ensure same shape of output between eval() and train()
        y_pred = self.classifier(x)
        return y_pred
    
    def forward_upto_classifier(self, x): # x: (bs, n_layers, n_inputs)
        layer_wise_x = []
        for layer in range(x.shape[-2]):
            if self.no_act_proj:
                layer_wise_x.append(torch.squeeze(x[:,layer,:]))
            else:
                layer_wise_x.append(self.linear(torch.squeeze(x[:,layer,:])))
        x = torch.stack(layer_wise_x, dim=-2) # x: (bs, n_layers, d_model)
        if len(x.shape)==2: x = x[None,:,:] # Add back bs dimension as torch.squeeze in prev line would remove it when bs=1  # x: (bs, n_layers, d_model)
        x = x.reshape([x.shape[0],x.shape[1]*x.shape[2]])  # x: (bs, n_layers * d_model)
        if self.batch_norm: x = self.batch_norm_layer(x)
        if self.non_linear:
            x = self.linear1(x)
            x = self.relu1(x)
            x = self.linear2(x)
            x = self.relu2(x)
            x = self.linear3(x)
            x = self.relu3(x)
        return x

class My_Transformer_Layer(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_layers, n_outputs, bias, n_blocks=1, use_pe=False, batch_norm=False, supcon=False, norm_emb=False, norm_cfr=False, cfr_no_bias=False, d_model=128, no_act_proj=False, device='cuda'):
        super().__init__()
        self.no_act_proj = no_act_proj
        d_model = d_model #128 # 256
        dim_feedforward = 1024 # 256
        nhead = 16 # 16 # 8
        max_length = 512*n_layers # max_new_tokens in generation config x num_layers
        self.use_pe =  use_pe
        self.batch_norm = batch_norm
        self.n_blocks = n_blocks
        self.linear = torch.nn.Linear(n_inputs, d_model, bias)
        self.class_token = torch.nn.Parameter(torch.randn(1,1,d_model))
        self.batch_norm_layer = torch.nn.BatchNorm1d(n_layers+1)
        self.transfomer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transfomer2 = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.supcon=supcon
        self.norm_emb=norm_emb
        self.norm_cfr=norm_cfr
        self.projection = torch.nn.Linear(d_model,int(d_model/2),bias=False)
        self.classifier = torch.nn.Linear(d_model, n_outputs, bias=not cfr_no_bias)
        torch.nn.init.normal_(self.class_token, std=0.02)

        self.query = torch.nn.Parameter(torch.randn(n_inputs)).to(device)

        # Positional Encoding: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        # self.dropout = nn.Dropout(p=dropout)      
        self.pe = torch.zeros(max_length, d_model).to(device) # create tensor of 0s
        k = torch.arange(0, max_length).unsqueeze(1) # create position column  
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # calc divisor for positional encoding
        self.pe[:, 0::2] = torch.sin(k * div_term) # calc sine on even indices
        self.pe[:, 1::2] = torch.cos(k * div_term)  # calc cosine on odd indices 
        self.pe = self.pe.unsqueeze(0) # add dimension      
        # self.register_buffer("pe", self.pe) # buffers are saved in state_dict but not trained by the optimizer # TODO: fix attribute already exists err

    def forward(self, x): # x: (bs, n_layers, n_inputs)
        x = self.forward_upto_classifier(x)
        if self.supcon or self.norm_emb: x = F.normalize(x, p=2, dim=-1) # unit normalise, setting dim=-1 since inside forward() we define ops for one sample only
        if self.norm_cfr and self.training==False:
            norm_cfr_wgts = F.normalize(self.classifier.weight, p=2, dim=-1)
            y_pred = torch.sum(x * norm_cfr_wgts, dim=-1)
            y_pred = (y_pred + 1)/2 # re-scale to yield probability values
            assert y_pred.min().item()>=0 and y_pred.max().item()<=1
            return y_pred[:,None] # ensure same shape of output between eval() and train()
        y_pred = self.classifier(x)
        return y_pred
    
    def forward_upto_classifier(self, x): # x: (bs, n_layers, n_inputs) or (bs, n_layers, n_tokens, n_inputs) # n_inputs=llm_dim
        x = x.to(torch.float64)
        layer_wise_x = []
        layers_dim_idx = -3 if len(x.shape)==4 else -2
        for layer in range(x.shape[layers_dim_idx]):
            layer_x = torch.squeeze(x[:,layer])
            if len(x.shape)==4: # if pooling across tokens at each layer
                qt_h = torch.matmul(layer_x,self.query.to(layer_x.dtype)) # qt_h: (bs, n_tokens)
                att_wgts = nn.functional.softmax(qt_h, dim=-1)  # att_wgts: (bs, n_tokens)
                att_out = []
                for sample in range(layer_x.shape[0]):
                    att_out.append(torch.matmul(att_wgts[sample],layer_x[sample]))  # att_out: (llm_dim)
                layer_x = torch.stack(att_out) # layer_x: (bs, llm_dim)
            if self.no_act_proj:
                layer_wise_x.append(layer_x)
            else:
                # print(layer_x.shape)
                layer_wise_x.append(self.linear(layer_x))
        x = torch.stack(layer_wise_x, dim=-2) # x: (bs, n_layers, d_model)
        if len(x.shape)==2: x = x[None,:,:] # Add back bs dimension as torch.squeeze in prev line would remove it when bs=1
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=-2) # x: (bs, n_layers+1, d_model)
        if self.use_pe: x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        if self.batch_norm: x = self.batch_norm_layer(x)
        x = self.transfomer(x) # x: (bs, n_layers, d_model)
        if self.n_blocks==2: x = self.transfomer2(x)
        x = x[:,0,:] # Take first token embedding (CLS token)
        return x

class LogisticRegression_Torch(torch.nn.Module):    
    def __init__(self, n_inputs, n_outputs, bias, norm_emb=False, norm_cfr=False, cfr_no_bias=False): # bias should not be used; retained for backward compatibility; use cfr_no_bias instead for alignment with other NLP and Tr networks
        super().__init__()
        self.norm_emb=norm_emb
        self.norm_cfr=norm_cfr
        self.linear = torch.nn.Linear(n_inputs, n_outputs, bias=not cfr_no_bias)
    def forward(self, x):
        if self.norm_emb: x = F.normalize(x, p=2, dim=-1) # unit normalise, setting dim=-1 since inside forward() we define ops for one sample only
        if self.norm_cfr and self.training==False:
            norm_cfr_wgts = F.normalize(self.linear.weight, p=2, dim=-1)
            y_pred = torch.sum(x * norm_cfr_wgts, dim=-1)
            y_pred = (y_pred + 1)/2 # re-scale to yield probability values
            assert y_pred.min().item()>=0 and y_pred.max().item()<=1
            return y_pred[:,None] # ensure same shape of output between eval() and train()
        y_pred = self.linear(x)
        return y_pred
    def forward_upto_classifier(self, x):
        return x

class Ens_Att_Pool(torch.nn.Module):    
    def __init__(self, n_inputs, n_outputs, bias, norm_emb=False, norm_cfr=False, cfr_no_bias=False, probes_file_name=None): # bias should not be used; retained for backward compatibility; use cfr_no_bias instead for alignment with other NLP and Tr networks
        super().__init__()
        self.norm_emb=norm_emb
        self.norm_cfr=norm_cfr
        self.linear = torch.nn.Linear(n_inputs, n_outputs, bias=not cfr_no_bias)
        self.probes_file_name = probes_file_name
    def forward(self, x):
        x = self.forward_upto_classifier(x)
        if self.norm_emb: x = F.normalize(x, p=2, dim=-1) # unit normalise, setting dim=-1 since inside forward() we define ops for one sample only
        if self.norm_cfr and self.training==False:
                norm_cfr_wgts = F.normalize(self.linear.weight, p=2, dim=-1)
                y_pred = torch.sum(x * norm_cfr_wgts, dim=-1)
                y_pred = (y_pred + 1)/2 # re-scale to yield probability values
                assert y_pred.min().item()>=0 and y_pred.max().item()<=1
                return y_pred[:,None] # ensure same shape of output between eval() and train()
        y_pred = self.linear(x)
        return y_pred
    def forward_upto_classifier(self, x): # x: (bs, n_layers, n_tokens, llm_dim)
        ind_att_pool_out = []
        for layer in range(x.shape[1]):
            layer_x = torch.squeeze(x[:,layer])
            model_path = f'{self.probes_file_name}_{layer}_0'
            ind_att_pool_model = torch.load(model_path,map_location='cuda')
            model_out = torch.squeeze(torch.sigmoid(ind_att_pool_model(layer_x)).data)
            ind_att_pool_out.append(model_out)
        ind_att_pool_out = torch.stack(ind_att_pool_out, dim=1)
        # print(ind_att_pool_out.shape)
        return ind_att_pool_out # (bs, n_layers)

class My_SupCon_NonLinear_Classifier4(nn.Module):
    def __init__(self, input_size, output_size=2, bias=True, use_dropout=False, supcon=False, norm_emb=False, norm_cfr=False, cfr_no_bias=False, path=None):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.supcon=supcon
        self.norm_emb=norm_emb
        self.norm_cfr=norm_cfr
        self.projection = torch.nn.Linear(64,32,bias=False)
        self.classifier = nn.Linear(64, output_size, bias=not cfr_no_bias)
    def forward(self,x):
        x = self.forward_upto_classifier(x)
        if self.supcon or self.norm_emb: x = F.normalize(x, p=2, dim=-1) # unit normalise, setting dim=-1 since inside forward() we define ops for one sample only
        if self.norm_cfr and self.training==False:
            norm_cfr_wgts = F.normalize(self.classifier.weight, p=2, dim=-1)
            y_pred = torch.sum(x * norm_cfr_wgts, dim=-1)
            y_pred = (y_pred + 1)/2 # re-scale to yield probability values
            assert y_pred.min().item()>=0 and y_pred.max().item()<=1
            return y_pred[:,None] # ensure same shape of output between eval() and train()
        output = self.classifier(x)
        return output
    def forward_upto_classifier(self, x):
        if self.use_dropout: x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        return x

class My_NonLinear_Classifier():
    def __init__(self, input_size, path=None):
        input_size = input_size
        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(0.2))
        self.model.add_module(f"linear1", nn.Linear(input_size, 256))
        self.model.add_module(f"relu1", nn.ReLU())
        self.model.add_module(f"linear2", nn.Linear(256, 2))
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location = "cpu")["model_state_dict"])
        # self.model.to(args.device)

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def list_of_floats(arg):
    return list(map(float, arg.split(',')))
def list_of_strs(arg):
    return list(map(str, arg.split(',')))

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def my_squad_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def load_jsonl(file_path, is_gzip=False):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                # term=item.get('term', None),
                # description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                # facts=item.get('facts', []),
                # decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=6, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("Do hamsters provide food for any animals?")
    chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
    answer.append("yes")

    question.append("Could Brooke Shields succeed at University of Pennsylvania?")
    chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
    answer.append("yes")

    question.append("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?")
    chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
    answer.append("no")

    question.append("Yes or no: Is it common to see frost during some college commencements?")
    chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
    answer.append("yes")

    question.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
    chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
    answer.append("no")

    question.append("Yes or no: Would a pear sink in water?")
    chain.append("The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
    answer.append("no")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text

def build_prompt(input_text, n_shot, cot_flag, shuffle, dataset_name):
    demo = create_demo_text(n_shot, cot_flag, shuffle) if dataset_name=='strqa' else create_demo_text_gsm8k(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred, random_guess=False):
    model_pred = model_pred.lower()

    if "Thus, yes." in model_pred:
        preds = "yes"
    elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
        preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
        if random_guess:
            preds = "no" if "no" in model_pred else "yes"
        else:
            return None
    if preds not in ["yes", "no"]:
        print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
        if random_guess:
            preds = "no"
        else:
            return None

    return (preds == "yes")

def load_jsonl_gsm8k(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def is_correct_gsm8k(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text_gsm8k(n_shot=8, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text

def clean_answer_gsm8k(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def tokenized_from_file(file_path, tokenizer, num_samples=1, device='cuda'): 

    all_prompts, all_tokenized_prompts, resp_tokenized = [], [], []
    answer_token_idxes = []
    with open(file_path, 'r') as read_file:
        data = []
        for line in read_file:
            data.append(json.loads(line))
    for row in data:
        question = row['prompt']
        for j in range(1,num_samples+1,1):
            answer = row['response'+str(j)]
            prompt = question + answer
            all_prompts.append(prompt)
            tokenized_prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_tokenized_prompts.append(tokenized_prompt.to(device))
            resp_tokenized.append([tokenizer.decode(input_tokid) for input_tokid in tokenized_prompt[0]])
            answer_token_idxes.append(len(tokenizer(question, return_tensors = 'pt').input_ids[0]))
        
    return all_prompts, all_tokenized_prompts, answer_token_idxes, resp_tokenized

def tokenized_from_file_v2(file_path, tokenizer, num_samples=1, device='cuda'): 

    all_prompts, all_tokenized_prompts, resp_tokenized = [], [], []
    answer_token_idxes = []
    with open(file_path, 'r') as read_file:
        data = json.load(read_file)
    for i in range(len(data['full_input_text'])):
        if num_samples==1:
            question = data['full_input_text'][i]
            # answer = data['model_completion'][i]
            answer = data['model_completion'][i] if 'strqa' in file_path else data['model_answer'][i] # For strqa, we want full COT response
            prompt = question + answer
            if prompt==[]: continue # skip empty lines (i.e. lines that caused oom during generation)
            all_prompts.append(prompt)
            tokenized_prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_tokenized_prompts.append(tokenized_prompt.to(device))
            resp_tokenized.append([tokenizer.decode(input_tokid) for input_tokid in tokenized_prompt[0]])
            answer_token_idxes.append(len(tokenizer(question, return_tensors = 'pt').input_ids[0]))
        else:
            question = data['full_input_text'][i][0]
            for j in range(num_samples):
                answer = data['model_completion'][i][j] if 'strqa' in file_path else data['model_answer'][i][j] # For strqa, we want full COT response
                prompt = question + answer
                all_prompts.append(prompt)
                tokenized_prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
                all_tokenized_prompts.append(tokenized_prompt.to(device))
                resp_tokenized.append([tokenizer.decode(input_tokid) for input_tokid in tokenized_prompt[0]])
                answer_token_idxes.append(len(tokenizer(question, return_tensors = 'pt').input_ids[0]))
        
    return all_prompts, all_tokenized_prompts, answer_token_idxes, resp_tokenized

def load_prompt_responses(args):
    file_path = f'{args.my_save_path}/{args.model_name}_{args.file_name}.json'
    if args.dataset_name=='strqa':
        with open(file_path, 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            if args.num_samples==1:
                questions.append(data['full_input_text'][i])
                answers.append(data['model_completion'][i].lower())
            else:
                for j in range(args.num_samples):
                    questions.append(data['full_input_text'][i][0])
                    answers.append(data['model_completion'][i][j].lower())
    else:
        with open(file_path, 'r') as read_file:
            data = []
            for line in read_file:
                data.append(json.loads(line))
        for row in data:
            for j in range(1,args.num_samples+1,1):
                questions.append(row['prompt'])
                answers.append(row['response'+str(j)].lower())
    return questions,answers

def load_labels(save_path,model_name,dataset_name,file_name,labels_file_name,num_samples):
    if dataset_name == 'gsm8k' or dataset_name == 'strqa': # or 'dola' in file_name:
        if 'se_labels' in labels_file_name:
            file_path = f'{save_path}/uncertainty/{model_name}_{labels_file_name}.npy'
            labels = np.load(file_path)
        else:
            labels = []
            with open(file_path, 'r') as read_file:
                data = json.load(read_file)
            for i in range(len(data['full_input_text'])):
                if num_samples==1:
                    label = 0 if data['is_correct'][i]==True else 1
                    labels.append(label)
                else:
                    for j in range(num_samples):
                        label = 0 if data['is_correct'][i][j]==True else 1
                        labels.append(label)
            labels = labels[:len_dataset]
    else:
        if 'se_labels' in labels_file_name:
            file_path = f'{save_path}/uncertainty/{model_name}_{labels_file_name}.npy'
            labels = np.load(file_path).tolist()
        else:
            labels = []
            file_path = f'{save_path}/responses/{labels_file_name}.json' if dataset_name == 'tqa_gen' else f'{save_path}/responses/{model_name}_{labels_file_name}.json'
            with open(file_path, 'r') as read_file:
                for line in read_file:
                    data = json.loads(line)
                    if 'greedy' in labels_file_name:
                        label = 0 if data['rouge1_to_target']>0.3 else 1 # pos class is hallu
                        labels.append(label)
                        if(len(labels))==len_dataset: break
                    else:
                        for j in range(1,num_samples+1,1):
                            label = 0 if data['rouge1_to_target_response'+str(j)]>0.3 else 1 # pos class is hallu
                            labels.append(label)
        labels = labels[:len_dataset]
    return labels

def load_acts(args,file_name,prompt_idxs)
    act_type = {'mlp':'mlp_wise','mlp_l1':'mlp_l1','ah':'head_wise','layer':'layer_wise','layer_att_res':'layer_wise'}
    temp_idxs = prompt_idxs
    acts, act_wise_file_paths, unique_file_paths = [], [], []
    for idx in temp_idxs:
        file_end = idx-(idx%args.acts_per_file)+args.acts_per_file # 487: 487-(87)+100
        file_path = f'{args.save_path}/features/{args.model_name}_{args.dataset_name}_{args.token}/{args.model_name}_{file_name}_{args.token}_{act_type[args.using_act]}_{file_end}.pkl'
        act_wise_file_paths.append(file_path)
        if file_path not in unique_file_paths: unique_file_paths.append(file_path)
    file_wise_data = {}
    print("Loading files..")
    for file_path in unique_file_paths:
        file_wise_data[file_path] = np.load(file_path,allow_pickle=True)
        if args.using_act=='layer_att_res':
            file_path2 = file_path.replace('layer_wise','attresout_wise')
            file_wise_data[file_path2] = np.load(file_path2,allow_pickle=True)
    actual_answer_width = []
    for idx in tqdm(temp_idxs):
        act = file_wise_data[act_wise_file_paths[idx]][idx%args.acts_per_file]
        if args.using_act=='layer_att_res': 
            file_path2 = act_wise_file_paths[idx].replace('layer_wise','attresout_wise')
            act2 = file_wise_data[file_path2][idx%args.acts_per_file]
            act = np.concatenate([act,act2],axis=0)
            if args.token=='prompt_last_onwards':
                actual_answer_width.append(act.shape[1])
                max_tokens = args.max_answer_tokens
                if act.shape[1]<max_tokens: # Let max num of answer tokens be max_tokens
                    pads = np.zeros([act.shape[0],max_tokens-act.shape[1],act.shape[2]])
                    act = np.concatenate([act,pads],axis=1)
                elif act.shape[1]>max_tokens:
                    act = act[:,-max_tokens:,:] # Only most recent tokens
        acts.append(act)
    # print(np.histogram(actual_answer_width), np.max(actual_answer_width))
    acts = torch.from_numpy(np.stack(acts)).to(device).to(torch.float64)
    return 

def get_token_tags(responses,resp_tokenized):
    # Load the small English model
    nlp = spacy.load("en_core_web_sm")
    # Tag tokens
    issues = []
    tagged_token_idxs = []
    for i,response in tqdm(enumerate(responses)):
        doc = nlp(response)
        text_tokens = [token.text for token in doc if token.pos_ in ['PROPN','NOUN','NUM'] and token.text not in ['bot','questions','Q','*',"'"]] # what about 'A'?
        cur_idxs = []
        for text in text_tokens: # This will only find the first mention of the text
            for j,token in enumerate(resp_tokenized[i]):
                found = False
                if token in text: # since llama tokens may be smaller than spacy tokens
                    # print(text,j)
                    k = 1
                    while j+k<=len(resp_tokenized[i]):
                        if ''.join(resp_tokenized[i][j:j+k])==text:
                            found=True
                            cur_idxs.append((j,j+k))
                            break
                        k += 1
                if found==True:
                    break
        assert len(cur_idxs)<=len(text_tokens)
        if len(cur_idxs)<len(text_tokens):
            issues.append(i)
        tagged_token_idxs.append(cur_idxs)
    return tagged_token_idxs

def get_llama_activations_bau(model, prompt, device, HEADS=None, MLPS=None, MLPS_L1=None, ATT_RES_OUTS=None): 

    # HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    # MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    # MLPS_L1 = [f"model.layers.{i}.mlp.up_proj_out" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        # prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS+MLPS_L1+ATT_RES_OUTS) as ret:
            output = model(prompt, output_hidden_states = True, use_cache=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().to(torch.float32).numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().to(torch.float32).numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().to(torch.float32).numpy()
        mlp_l1_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS_L1]
        mlp_l1_wise_hidden_states = torch.stack(mlp_l1_wise_hidden_states, dim = 0).squeeze().to(torch.float32).numpy()
        attresoutput_wise_hidden_states = [ret[att].output.squeeze().detach().cpu() for att in ATT_RES_OUTS]
        attresoutput_wise_hidden_states = torch.stack(attresoutput_wise_hidden_states, dim = 0).squeeze().to(torch.float32).numpy()
        del output
    
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, mlp_l1_wise_hidden_states, attresoutput_wise_hidden_states

def get_llama_activations_bau_custom(model, prompt, device, using_act, layer, token, answer_token_idx=-1, tagged_token_idxs=[]):

    if using_act=='mlp':
        ANALYSE = [f"model.layers.{layer}.mlp"]
    elif using_act=='ah':
        ANALYSE = [f"model.layers.{layer}.self_attn.head_out"]
    elif using_act=='mlp_l1':
        ANALYSE = [f"model.layers.{layer}.mlp.up_proj_out"]
    else:
        ANALYSE = []

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, ANALYSE) as ret:
            output = model(prompt, output_hidden_states = True)
        if using_act in ['mlp','mlp_l1','ah']: activation = ret[ANALYSE[0]].output.squeeze().detach().to(torch.float32)
        if layer==-1: 
            layer_activation = torch.stack(output.hidden_states, dim=0).squeeze().detach().cpu().to(torch.float32)
        else:
            layer_activation = output.hidden_states[layer].squeeze().detach().cpu().to(torch.float32)
        del output

    if using_act=='layer' and token=='answer_last':
        return layer_activation[-1,:]
    if token=='answer_last':
        return activation[-1,:]
    elif token=='prompt_last':
        return activation[answer_token_idx-1,:]
    elif token=='maxpool_all':
        return torch.max(activation,dim=0)[0]
    elif using_act=='layer' and token in ['least_likely','after_least_likely','random']:
        return layer_activation[:,answer_token_idx,:]
    elif using_act=='layer' and token=='tagged_tokens':
        tagged_token_idxs = tagged_token_idxs if len(tagged_token_idxs)>0 else [(1,layer_activation.shape[0])] # Skip the first token
        return torch.cat([layer_activation[a-1:b-1,:] for a,b in tagged_token_idxs],dim=0)
    elif using_act=='layer' and token=='tagged_tokens_and_last':
        tagged_acts = [layer_activation[a-1:b-1,:] for a,b in tagged_token_idxs] + [layer_activation[-1:,:]] if len(tagged_token_idxs)>0 else [layer_activation[1:,:]]
        return torch.cat(tagged_acts,dim=0)
    elif token=='tagged_tokens':
        tagged_token_idxs = tagged_token_idxs if len(tagged_token_idxs)>0 else [(1,activation.shape[0])] # Skip the first token
        return torch.cat([activation[a-1:b-1,:] for a,b in tagged_token_idxs],dim=0)
    else:
        return activation

def get_token_nll(model, prompt, device, predicted_token_id):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = model(prompt)
        logits = output.logits[0,-1,:] # output.logits: (bs, tokens, vocab)
        nll = -F.log_softmax(logits, dim=-1)[predicted_token_id].item()
        # output = model.generate(prompt,max_new_tokens=1,do_sample=False,return_dict_in_generate=True,output_scores=True)
        # scores = output.scores[0][0,predicted_token_id].item() # output.scores[0]: (bs, vocab)
        # print(logits,scores) # Checked for a few - the two values are the same
        # print(nll)
    return nll

def get_llama_logits(model, prompt, device): 

    model.eval()
    with torch.no_grad(): 
        prompt = prompt.to(device)
        logits = model(prompt).logits
        logits = logits.detach().cpu()
        return logits

def get_num_layers(model_name):
    # TODO: Replace with dictionary using model names
    if '7B' in model_name or '8B' in model_name:
        return 33
    elif '13B' in model_name:
        return 40
    elif '33B' in model_name:
        return 60
    elif '2B' in model_name:
        return 18 
    else:
        raise ValueError("Unknown model size.")

def get_num_layers(model_name):
    # TODO: Replace with dictionary using model names
    if 'gemma_2B' in model_name:
        return 8
    elif 'gemma_7B' in model_name:
        return 16
    elif 'llama' in model_name:
        return 32
    else:
        raise ValueError("Unknown model size.")

def get_act_dims(model_name):
    if 'gemma_2B' in model_name:
        return {'layer_att_res':2048,'layer':2048,'mlp':2048,'mlp_l1':None,'ah':256} # TODO: Update mlp_l1
    elif 'gemma_7B' in model_name:
        return {'layer_att_res':3072,'layer':3072,'mlp':3072,'mlp_l1':None,'ah':128} # TODO: Update mlp_l1
    elif 'llama' in model_name:
        return {'layer_att_res':4096,'layer':4096,'mlp':4096,'mlp_l1':11008,'ah':128}
    else:
        raise ValueError("Unknown model size.")

def get_best_threshold(val_true, val_preds):
    best_val_perf, best_t = 0, 0
    thresholds = np.histogram_bin_edges(val_preds, bins='sqrt')
    # print(np.histogram(val_preds, bins=thresholds))
    for t in thresholds:
        val_pred_at_thres = deepcopy(val_preds) # Deep copy so as to not touch orig values
        val_pred_at_thres[val_pred_at_thres>t] = 1
        val_pred_at_thres[val_pred_at_thres<=t] = 0
        cls1_f1 = f1_score(val_true,val_pred_at_thres)
        cls0_f1 = f1_score(val_true,val_pred_at_thres,pos_label=0)
        perf = np.mean((cls1_f1,cls0_f1))
        if perf>best_val_perf:
            best_val_perf, best_t = perf, t
    # print(best_val_perf,best_t)
    return best_t
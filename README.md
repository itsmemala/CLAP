# Cross Layer Attention Probing

This repository provides the code for implementing cross-layer attention probing (CLAP) and baseline probing methods for detecting factual hallucinations in LLM responses.

Several parts of the code (repository structure, environment, extracting activations, etc.) are derived from the Inference-Time Intervention repository (https://github.com/likenneth/honest_llama). Code for implementing Semantic Entropy Probing is from https://github.com/OATML/semantic-entropy-probes.
<!-- Code for implementing SupCon loss is adapted from (). -->

## Abstract
With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitigation approaches. Finally, we show that CLAP maintains high reliability even when applied out-of-distribution.


## Table of Contents
1. [Installation](#installation)
2. [Workflow](#workflow)
<!-- 3. [How to Cite](#how-to-cite) -->


## Installation
Run the following commands to set things up.
```
git clone https://github.com/itsmemala/CLAP.git
cd CLAP
conda env create -f environment.yaml
conda activate clap
```


## Workflow

(1) Generate LLM responses to prompts. Use the --do_sample and --num_ret_seq arguments to control whether greedy or sampling strategy is used and number of responses to sample per prompt, respectively. Supported dataset options are 'trivia_qa', 'nq_open', 'strqa', 'gsm8k', 'city_country', 'movie_cast', 'player_date_birth'.
```
python get_prompt_responses.py gemma_7B trivia_qa --use_split train 
python get_prompt_responses.py gemma_7B trivia_qa --use_split validation --len_dataset 2000
```

(2) Save LLM activations.
```
python get_activations.py gemma_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_train5000
python get_activations.py gemma_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_validation2000
```

(3) Calculate Uncertainty and Semantic Entropy of responses. (for uncertainty-based and semantic entropy probing baselines).
```
python get_uncertainty_scores.py gemma_7B trivia_qa --file_name greedy_responses_train5000 --num_samples 1
python get_uncertainty_scores.py gemma_7B trivia_qa --file_name greedy_responses_validation2000 --num_samples 1
python get_uncertainty_scores.py gemma_7B trivia_qa --file_name sampledplus_responses_train5000 --num_samples 10
python get_uncertainty_scores.py gemma_7B trivia_qa --file_name sampledplus_responses_validation2000 --num_samples 10
python get_semantic_entropy.py gemma_7B trivia_qa --file_name sampledplus_responses_train5000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10
```

(4) Train and test cross-layer probes. Use the --method argument to choose between 'clap' and other cross-layer methods (eg. 'project_linear'). Use the --using_act and --token arguments to control which component (eg. 'layer', 'mlp', 'head') and token (eg. 'prompt_last','answer_last') level activations are used, respectively.
```
python train_cross_layer_probes.py gemma_7B trivia_qa --train_file_name trivia_qa_greedy_responses_train5000 --test_file_name trivia_qa_greedy_responses_validation2000 --train_labels_file_name trivia_qa_greedy_responses_labels_train5000 --test_labels_file_name trivia_qa_greedy_responses_labels_validation2000 --using_act layer --token answer_last
```

(5) Train and test layer-wise probes. Use the --method argument to choose between methods (eg. 'individual_linear', 'individual_non_linear'). Use the --using_act and --token arguments to control which component (eg. 'layer', 'mlp', 'head') and token (eg. 'prompt_last','answer_last') level activations are used, respectively.
```
python train_layer_wise_probes.py gemma_7B trivia_qa --train_file_name trivia_qa_greedy_responses_train5000 --test_file_name trivia_qa_greedy_responses_validation2000 --train_labels_file_name trivia_qa_greedy_responses_labels_train5000 --test_labels_file_name trivia_qa_greedy_responses_labels_validation2000 --using_act layer --token answer_last
```

(6) Train and test layer-wise Semantic Entropy (SEP) probes. Use the --using_act and --token arguments to control which component (eg. 'layer', 'mlp', 'head') and token (eg. 'prompt_last','answer_last') level activations are used, respectively.
```
python train_layer_wise_probes.py gemma_7B trivia_qa --train_file_name trivia_qa_greedy_responses_train5000 --test_file_name trivia_qa_greedy_responses_validation2000 --train_labels_file_name trivia_qa_greedy_responses_train5000_se_labels --test_labels_file_name trivia_qa_greedy_responses_labels_validation2000 --using_act layer --token answer_last --method individual_linear
```

<!-- ## How to Cite

```

``` -->

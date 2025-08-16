# Cross Layer Attention Probing

This repository provides the code for implementing cross-layer attention probing (CLAP)[] and baseline probing methods for detecting factual hallucinations in LLM responses.

Several parts of the code (repository structure, extracting activations, etc.) are derived from the Inference-Time Intervention repository (). Code for implementing Semantic Entropy Probing is adapted from () and ().
<!-- Code for implementing SupCon loss is adapted from (). -->

## Abstract
With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which
processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained
detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitigation approaches. Finally, we show that CLAP maintains
high reliability even when applied out-of-distribution.


## Table of Contents
1. [Installation](#installation)
2. [Workflow](#workflow)
<!-- 3. [How to Cite](#how-to-cite) -->


## Installation
In this the root folder of this repo, run the following commands to set things up.
```
conda env create -f environment.yaml
conda activate clap
git clone https://github.com/itsmemala/CLAP.git
```


## Workflow

(1) 

(2) 

(3) 


<!-- ## How to Cite

```

``` -->

## Hierarchical Encoder for Text Summarization
&nbsp; This repository focuses on implementing a Hierarchical Encoder Transformer Seq2Seq model as a method to effectively handle long sequences in document summarization tasks. 
The emphasis is on directly implementing the model and verifying its results. 
The Hierarchical Encoder consists of lower and upper encoders. 
The lower encoder captures local information, while the upper encoder integrates global information for utilization.

<br> <br>


## Background

**Dataset** <br>
> This refers to the typical encoder structure used in the Standard Transformer model. It takes in long text as input and processes it through N identical encoder layers to produce hidden states, which are used as memory.

<br>

**Hierarchical Encoder Transformer Model** <br>
> As the name suggests, 

<br>

**Hierarchical Encoder Transformer Model** <br>
> As the name suggests, 

<br><br>

## Setup
| **Vocab Setup**                                    | **Model Setup**                         | **Training Setup**                |
| :---                                               | :---                                    | :---                                     |
| **`Tokenizer Type:`** &hairsp; `BPE`               | **`Input Dimension:`** `15,000`         | **`Epochs:`** `10`                       |
| **`Vocab Size:`** &hairsp; `15,000`                | **`Output Dimension:`** `15,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &nbsp; | **`Hidden Dimension:`** `512` &nbsp;    | **`Learning Rate:`** `1e-3`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`PFF Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &nbsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br><br>

## Result

| Model Type | &emsp; Hidden Dim &emsp; | &emsp; Num Layers &emsp; | &emsp; Num Heads &emsp; | &emsp; Params &emsp; | &emsp; Score &emsp; |
| :---: | :---: | :---: | :---: | :---: | :---: |
| &emsp; Base Encoder Model &emsp; | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |


<br><br>


## How to Use

```
├── ckpt             --this dir saves model checkpoints and training logs
├── config.yaml      --this file is for setting up arguments for model, training, and tokenizer 
├── data             --this dir is for saving Training, Validataion and Test Datasets
├── model            --this dir contains files for Deep Learning Model
│   ├── __init__.py
│   └── seq2seq.py
├── module           --this dir contains a series of modules
│   ├── data.py
│   ├── generate.py
│   ├── __init__.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── README.md
├── run.py          --this file includes codes for actual tasks such as training, testing, and inference to carry out the practical aspects of the work
└── setup.py        --this file contains a series of codes for preprocessing data, training a tokenizer, and saving the dataset.
```

<br>

clone git repo into your own env <br>

```
git clone https://github.com/moon23k/Sum_Encoders.git
``` 

prepare datasets and train tokenizer with setup file <br>

```
python3 -m setup.py
``` 

```
python3 -m run.py -mode [train, test, inference] -encoder [standard, hierarchical]
```

Configurations can be changed by modifying config.yaml file 

<br><br>

## Reference
[**Attention is all you need**]() <br>
[**Hierarchical Transformer**]()
<br>

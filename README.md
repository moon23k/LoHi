## Encoder Ablation Studies for Abstractive Text Summarization Task 
&emsp; Text summarization is a challenging natural language generation task, cause it deals with long input sequences. Therefore, the encoder is crucial to capture the contents of lengthy texts effectively. However, there is a lack of ablation studies that focus solely on the encoder's impact, including its structures and hyperparameters. Therefore, this repository aims to address this research lack by conducting ablation studies to compare the performance of summarization models based on different **Encoder Types**, **Hidden Dimension Sizes**, **Number of Layers**, and **Number of Heads**.

<br> <br>


## Ablation Variables

**Standard Trasnformer Encoder** <br>
> This refers to the typical encoder structure used in the Standard Transformer model. It takes in long text as input and processes it through N identical encoder layers to produce hidden states, which are used as memory.

<br>

**Hierarchical Transformer Encoder** <br>
> As the name suggests, this is a unified encoder consisting of two hierarchical encoders. The lower encoder extracts hidden states for each sentence. And then higher encoder it captures the relationship among sentences.

<br>

**Hidden Dimension** <br>
> The hidden dimension represents the dimension of hidden states used in all components of the Transformer encoder. The size of the hidden dimension impacts the model's ability to capture diverse contextual information. A larger hidden dimension allows for more information to be captured but comes with increased model size and computational cost.

<br>

**Num Layers** <br>
> This denotes the number of encoder layers. A smaller number of layers simplifies the model and accelerates training, but it may struggle with more complex tasks. On the other hand, increasing the number of layers enhances the model's expressive power but comes with increased model size and computational cost.

<br>

**Num Heads** <br>
> This refers to the number of heads in the Multi-Head Attention mechanism, a key component of the Transformer. Increasing the number of heads enables the model to learn more diverse patterns but comes with increased model parameters and computational cost.

<br><br>

## Result

| Model Type | &emsp; Hidden Dim &emsp; | &emsp; Num Layers &emsp; | &emsp; Num Heads &emsp; | &emsp; Params &emsp; | &emsp; Score &emsp; |
| :---: | :---: | :---: | :---: | :---: | :---: |
| &emsp; Base Encoder Model &emsp; | 256 | 3 |  8 |  | - |
|        Base Encoder Model        | 512 | 3 |  8 |  | - |
|        Base Encoder Model        | 256 | 6 |  8 |  | - |
|        Base Encoder Model        | 256 | 3 | 16 |  | - |
|        Base Encoder Model        | 512 | 6 | 16 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |
|        Hier Encoder Model        | 256 | 3 |  8 |  | - |


<br><br>

## How to Use
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

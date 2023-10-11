# Baby Llama From Scratch
This is an implementation of Llama from scratch. This project is inspired by [Llama from scratch](https://github.com/bkitano/llama-from-scratch/tree/ce3e52f4f154ce16345927c4f2c5082b5ecfea13).

TODO: include differences, e.g. pytorch lighting, architecture a bit different, optimizer, tokenizer.


It includes writing our own attention mechanism that include the three main components of the Llama paper:
1. RMSNorm for pre-normalization
2. RoPE (Rotary Positional Embedding)
3. SwiGLU activation function

Here is a diagram illustrating a single block:
![](/imgs/diagram1.png)

Boring part first: set up an environment and install dependencies with poetry
```
conda create -n llama python=3.10
conda activate llama

pip install poetry
poetry install
```

Good, now that we have our environment, let's have an intution of the three main Llama components and implement them!

## RMSNorm (Root Mean Square Layer Normalization)
RMSNorm is used to normalize the input of each transformer sub-layer. The inspiration of including pre-normalization is taken from GPT3, which showed it improves training stability with respect to normalizing the output. 

RMSNorm is a simplification of the LayerNorm, it is computationally simpler and thus more efficient than LayerNorm. The main differences between the two are that RMSNorm 1) uses the root mean square, instead of the standard devation, for re-scaling and 2) it's not re-centering invariant.

For more mathematical and implementation details about RMSNorm and its differences with Batch Normalization and Layer Normalization, refer to our dedicated [blog post](https://afterhoursresearch.hashnode.dev/batch-normalization-layer-normalization-and-root-mean-square-layer-normalization-a-comprehensive-guide-with-python-implementations).

## RoPE (Rotary Positional Embedding)
RoPE is based on the idea of embedding the position of a token in a sequence by rotating queries and keys in the attention mechanism, with a different rotation at each position.

We wrote a [blog post](https://afterhoursresearch.hashnode.dev/rope-rotary-positional-embedding) on RoPE, focusing on its mathematical formulation and its practical implementation in PyTorch.

## SwiGLU
SwiGLU activation function: Use a dimension of $\frac{2}{3}4d$. Also used in PaLM (their dimension is $4d$).


## Train a Tokenizer
```
Token Ids: [2, 725, 12, 68, 67, 5327, 137, 6799, 68, 67, 9936, 104, 227, 4150, 120, 9025, 8, 109, 771, 371, 68, 67, 4391, 3236, 289, 80, 1005, 10, 3]]

Encoded Tokens : ['[BOS]', 'CORIOLANUS', ':', 'Ġ', 'Ċ', 'ĠIt', 'Ġis', 'Ġapart', 'Ġ', 'Ċ', 'ĠThat', 'ĠI', 'Ġshall', 'Ġblush', 'Ġin', 'Ġacting', ',', 'Ġand', 'Ġmight', 'Ġwell', 'Ġ', 'Ċ', 'ĠBe', 'Ġtaken', 'Ġfrom', 'Ġthe', 'Ġpeople', '.', '[EOS]']

Decoded Tokens: CORIOLANUS: 
 It is apart 
 That I shall blush in acting, and might well 
 Be taken from the people.
 ```
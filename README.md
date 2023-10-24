# Baby Llama From Scratch
Welcome to this deep dive into building Llama from scratch. This project is inspired by [Llama from scratch](https://github.com/bkitano/llama-from-scratch/tree/ce3e52f4f154ce16345927c4f2c5082b5ecfea13), but it diverges in several ways. For instance, we make various architectural adjustments, such as modifications to the placement of residuals and RMS normalization within each attention block, among other changes. We train a Byte-Pair Encoding (BPE) tokenizer instead of using a simple character-level tokenizer. As for optimization, we utilize the AdamW optimizer along with a cosine learning rate schedule and gradient clipping, which aligns with what is used in the original paper, rather than a basic Adam optimizer. Our implementation also uses PyTorch Lightning for more structured and maintainable code. Finally, we incorporate Weights and Biases (wandb) for experiment tracking and Hydra for configuration management.


Our project is comprehensive and, among other things, includes constructing our own attention mechanism that incorporates the three key components specified in the original Llama paper:
1. RMSNorm for pre-normalization
2. RoPE (Rotary Positional Embedding)
3. SwiGLU activation function

To help visualize the architecture, here's a diagram illustrating a single block of our model:
![](/imgs/diagram1.png)


## Setting Up the Environment
First things first: let's set up our development environment to ensure that everything runs smoothly. For this project, we'll be using Python 3.10 and manage our dependencies using Poetry. Here's how you can set it up:

```
# Create a new Conda environment named 'llama'
conda create -n llama python=3.10

# Activate the Conda environment
conda activate llama

# Install Poetry for dependency management
pip install poetry

# Install project dependencies
poetry install
```
With the environment set up, you're now ready to dive into the intricacies of building Baby Llama from scratch.

## Tokenizer Training
Given the domain-specific language characteristics of our dataset, we opted for training a custom Byte-Pair Encoding (BPE) tokenizer. This allows for more accurate and efficient tokenization specific to our corpus.

Our code snippet for training the tokenizer involves several components:
1. Initialization of a BPE tokenizer.
2. Setting pre-tokenizers and decoders to ByteLevel.
3. Configuration of special tokens and post-processors.
4. Training the tokenizer on a specific dataset specified in the cfg.path.

#### Code Walkthrough

```python
# Initialize the BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
```
Here, we initialize a BPE tokenizer. We specify the unknown token as `[UNK]`, which is what the tokenizer will use for any character sequences it hasn't seen before.

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
```
These lines set the pre-tokenizer and decoder to use Byte-Level tokenization, a foundational part of BPE. This allows the BPE tokenizer to use bytes as the base vocabulary, providing an initial vocabulary size of 256.

Here, add_prefix_space=False indicates that no space will be prefixed to each word at the beginning of a sentence. 

```python
# Define the trainer and special tokens
trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
```
Here, we specify the training settings and declare special tokens that have specific roles during both training and inference. During training, BPE identifies the most frequently occurring pairs of consecutive bytes and merges them to create new tokens. These new tokens are then represented by new bytes that don't occur in the original dataset, thus effectively expanding the vocabulary.

```python
# Add post-processor for special tokens
tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
)
```
Post-processing is configured to automatically add `[BOS]` and `[EOS]` tokens at the beginning and end of each sequence (represented as `$A`), respectively. The numbers `2`and `3` specify the indices of `[BOS]` and `[EOS]` based on their order in the special tokens list, so they must match.

```python
# Train the tokenizer on the dataset
tokenizer.train([cfg.path], trainer)
```
Training is triggered using the `.train()` method, and it's here that all the previously set configurations come into play. The tokenizer is trained on the data specified in `cfg.path`.

```python
# Save the pretrained tokenizer
pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
pretrained_tokenizer.save_pretrained(cfg.tokenizer_path)
```
Finally, we save the trained tokenizer using the Transformers library's `PreTrainedTokenizerFast` class. 
Upon running the pretrained_tokenizer.save_pretrained(cfg.tokenizer_path) line, three files will be created within the folder specified by cfg.tokenizer_path. These files contain necessary configurations to reload the tokenizer for future use.

#### Example: Encoding and Decoding

To illustrate the tokenizer's functionality, let's encode and decode a sample sentence:

```python
encodings = tokenizer.encode("CORIOLANUS: \n It is apart \n That I shall blush in acting, and might well \n Be taken from the people.")
decodings = tokenizer.decode(encodings.ids)
print(f"Token Ids: {encodings.ids}")
print(f"Encoded Tokens : {encodings.tokens}")
print(f"Decoded Tokens: {decodings}")
```

This produces the following output:

```
Token Ids: [2, 725, 12, 68, 67, 5327, 137, 6799, 68, 67, 9936, 104, 227, 4150, 120, 9025, 8, 109, 771, 371, 68, 67, 4391, 3236, 289, 80, 1005, 10, 3]
Encoded Tokens : ['[BOS]', 'CORIOLANUS', ':', 'Ġ', 'Ċ', 'ĠIt', 'Ġis', 'Ġapart', 'Ġ', 'Ċ', 'ĠThat', 'ĠI', 'Ġshall', 'Ġblush', 'Ġin', 'Ġacting', ',', 'Ġand', 'Ġmight', 'Ġwell', 'Ġ', 'Ċ', 'ĠBe', 'Ġtaken', 'Ġfrom', 'Ġthe', 'Ġpeople', '.', '[EOS]']
Decoded Tokens: CORIOLANUS: 
 It is apart 
 That I shall blush in acting, and might well 
 Be taken from the people.
```
Here, the example output includes the following encoded tokens: `['[BOS]', 'CORIOLANUS', ':', 'Ġ', 'Ċ', 'ĠIt', 'Ġis', 'Ġapart', ...]`. You'll notice the special character `Ġ` in the encoded tokens. This character signifies a space before a word within a sentence and is a product of the ByteLevel pre-tokenization. In ByteLevel tokenization, spaces are also encoded into specific byte tokens, and Ġ is how the model represents these spaces when followed by a word within the context of a sentence.

This example demonstrates the tokenizer's ability to encode and decode text accurately, preserving the original sentence structure and adding special tokens at the beginning and end of the sequence.

### Running the Code

To execute this tokenizer training script, simply run:

```bash
python run_tokenizer.py
```

Because we're using Hydra for configuration management, modifying aspects like the dataset path or where to save the tokenizer is straightforward. All these settings are located in the `cfg` object and are sourced from a YAML configuration file.


## Data Preparation and DataLoader

## Llama Architecture
Let's have an intution of the three main Llama components and implement them!
### RMSNorm (Root Mean Square Layer Normalization)
RMSNorm is used to normalize the input of each transformer sub-layer. The inspiration of including pre-normalization is taken from GPT3, which showed it improves training stability with respect to normalizing the output. 

RMSNorm is a simplification of the LayerNorm, it is computationally simpler and thus more efficient than LayerNorm. The main differences between the two are that RMSNorm 1) uses the root mean square, instead of the standard devation, for re-scaling and 2) it's not re-centering invariant.

For more mathematical and implementation details about RMSNorm and its differences with Batch Normalization and Layer Normalization, refer to our dedicated [blog post](https://afterhoursresearch.hashnode.dev/batch-normalization-layer-normalization-and-root-mean-square-layer-normalization-a-comprehensive-guide-with-python-implementations).

### RoPE (Rotary Positional Embedding)
RoPE is based on the idea of embedding the position of a token in a sequence by rotating queries and keys in the attention mechanism, with a different rotation at each position.

We wrote a [blog post](https://afterhoursresearch.hashnode.dev/rope-rotary-positional-embedding) on RoPE, focusing on its mathematical formulation and its practical implementation in PyTorch.

### SwiGLU
SwiGLU activation function: Use a dimension of $\frac{2}{3}4d$. Also used in PaLM (their dimension is $4d$).


## Lightning Module

## Trainer 



## Train a Tokenizer
```
Token Ids: [2, 725, 12, 68, 67, 5327, 137, 6799, 68, 67, 9936, 104, 227, 4150, 120, 9025, 8, 109, 771, 371, 68, 67, 4391, 3236, 289, 80, 1005, 10, 3]

Encoded Tokens : ['[BOS]', 'CORIOLANUS', ':', 'Ġ', 'Ċ', 'ĠIt', 'Ġis', 'Ġapart', 'Ġ', 'Ċ', 'ĠThat', 'ĠI', 'Ġshall', 'Ġblush', 'Ġin', 'Ġacting', ',', 'Ġand', 'Ġmight', 'Ġwell', 'Ġ', 'Ċ', 'ĠBe', 'Ġtaken', 'Ġfrom', 'Ġthe', 'Ġpeople', '.', '[EOS]']

Decoded Tokens: CORIOLANUS: 
 It is apart 
 That I shall blush in acting, and might well 
 Be taken from the people.
 ```
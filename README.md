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
Let's now focus on the data preparation and the DataLoader.

#### Code Walkthrough

```python
tokenizer_name = "gpt2" if cfg.dataset.tokenizer_path is None else cfg.dataset.tokenizer_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
```
Here, `tokenizer_name` is set to either "gpt2" or a path to your saved custom tokenizer, based on `cfg.dataset.tokenizer_path`. `AutoTokenizer.from_pretrained` then loads the tokenizer. This gives you the flexibility to work with a standard pre-trained tokenizer or one tailored to your specific needs. For our experiments, `cfg.dataset.tokenizer_path` is the path to the folder we created in the previous "Tokenizer Training" step.

```python
dataset = getfromtext(
    data_path=Path(cfg.dataset.path),
    tokenizer=tokenizer, 
    tokenizer_args=dict(return_tensors="pt", add_special_tokens=True, truncation=True, padding="max_length", max_length=cfg.model.context_len+1)
)
```
`getfromtext` is a custom function that reads the txt data from `cfg.dataset.path` and returns it in an appropriate format (`CLMDataset` object).

The `CLMDataset` class is a custom dataset object that inherits from PyTorch's `Dataset` class. It handles the tokenization and formatting of your text data, making it compatible with PyTorch's DataLoader and ready for training.

The two relevant parts that are worth explaining are: 1) the `CLMDataset` `__getitem__` method and 2) the tokenizer arguments used.

`__getitem__` is designed to work with PyTorch's `DataLoader`. It returns a tuple consisting of input IDs, target IDs, and the attention mask.

```python
def __getitem__(self, idx: int) -> Tuple[int, int, int]:
    return self.tokens["input_ids"][idx, :-1], self.tokens["input_ids"][idx, 1:], self.tokens["attention_mask"][idx, :-1]
```

This slicing technique creates input and target sequences by shifting one token—a common practice in next-token prediction.

The tokenizer, with its arguments, is simply called within the class as:
```python
class CLMDataset(Dataset):
    def __init__(
        self,
        data: Path,
        tokenizer: AutoTokenizer,
        tokenizer_args: dict,
    ):
        self.data = data
        self.tokens = tokenizer(self.data, **tokenizer_args)
        ...
```
The tokenizer arguments include `return_tensors="pt"` to return PyTorch tensors, `add_special_tokens=True` to include special tokens in the tokenized output, `truncation=True` for handling sequences longer than the model's maximum input length, `padding="max_length"` to pad shorter sequences to the max length (in the batch), and `max_length=cfg.model.context_len+1` to set the maximum sequence length (the "+1" accounts for label-shifting during training).

```python
datamodule = CLMDataModule(
    data=dataset, 
    train_ratio=cfg.dataset.train_ratio,
    val_ratio=cfg.dataset.val_ratio, 
    test_ratio=cfg.dataset.test_ratio, 
    train_batchsize=cfg.trainer.train_batchsize, 
    val_test_batchsize=cfg.trainer.val_test_batchsize,
    num_workers=cfg.trainer.num_workers
)
```

Having prepared our data and made it compatible with PyTorch's `DataLoader`, the next step is to manage this data efficiently for different stages of the model training, validation, and testing. This is where `CLMDataModule` comes into play. `CLMDataModule` is a class that inherits from PyTorch Lightning's LightningDataModule and takes care of data loading and preparation.

The `setup` method of `CLMDataModule` splits the dataset into training, validation, and test sets based on the provided ratios. It takes a `stage` argument to determine which splits to prepare, allowing to use different data stages without reloading the entire dataset:

- If `stage` is set to `"fit"`, the method prepares the training and validation datasets.
- If `stage` is set to `"test"`, it prepares the test dataset.


```python
def setup(self, stage):
    train, val, test = random_split(
        dataset=self.data, 
        lengths=[self.train_ratio, self.val_ratio, self.test_ratio]
    )
    if stage == "fit":
        self.train, self.val = train, val
    if stage == "test":
        self.test = test
```

The class also provides standard methods like train_dataloader, val_dataloader, and test_dataloader to return PyTorch `DataLoader` objects for each phase. These methods are quite standard, utilizing the batch sizes and number of workers specified during initialization. These loaders will use the `CLMDataset` object you provided and its __getitem__ method to fetch batches of data.


## Llama Architecture
Let's have an intution of the three main Llama components and implement them!

First, we initialize the Llama architecture using the following code snippet:

```python
transformer = Llama(
    vocab_size=dataset.get_vocab_size(), 
    hidden_size=cfg.model.hidden_size, 
    context_len=cfg.model.context_len, 
    causal_attention=True, 
    n_heads=cfg.model.n_heads, 
    n_blocks=cfg.model.n_blocks
)
```

### Explanation of Llama Initialization

- `vocab_size`: The size of the vocabulary, taken from the dataset you're working with.
- `hidden_size`: The size of the hidden layer, specified in your configuration.
- `context_len`: The length of the context window for attention, also from your configuration.
- `causal_attention`: Boolean flag to indicate if the model should use causal (unidirectional) attention. Set to `True` here.
- `n_heads`: The number of attention heads, specified in your configuration.
- `n_blocks`: The number of transformer blocks (layers), also specified in your configuration.

### Llama Class Definition

The `Llama` class is defined as a subclass of PyTorch's `nn.Module`. Inside its `__init__` method:

- `self.embedding`: An embedding layer that converts token IDs to vectors.
- `self.attention_block`: A list of attention blocks, each handling self-attention and feed-forward operations.
- `self.unembedding`: A linear layer that maps the output back to vocabulary space.

In the `forward` method, the input sequence `x` goes through the embedding layer, the list of attention blocks, and finally the unembedding layer, before it is returned as output.

```python
class Llama(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int, 
        context_len: int,
        causal_attention: bool,
        n_heads: int,
        n_blocks: int
        ):
        super().__init__()
        self.context_len = context_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention_block = nn.ModuleList([LlamaSelfAttentionBlock(hidden_size, context_len, causal_attention, n_heads) for _ in range(n_blocks)])
        self.unembedding = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, is_inference=False):
        x = self.embedding(x)
        for single_block in self.attention_block:
            x = single_block(x, is_inference)
        x = self.unembedding(x)
        return x
```

This completes the architecture of our Llama model.


Let's now delve into the three main components of Llama and implement them!

### RMSNorm (Root Mean Square Layer Normalization)
RMSNorm is used to normalize the input of each transformer sub-layer. The inspiration for including pre-normalization comes from GPT-3, which showed that it improves training stability compared to output normalization.

RMSNorm is computationally simpler and more efficient than LayerNorm due to its utilization of root mean square for re-scaling and its lack of re-centering invariance.

Here's a simplified RMSNorm code snippet to give you an idea:

```python
class RMSnorm(nn.Module):
    def __init__(
        self, 
        size: int,
        eps: float = 1e-5, 
    ):
        super(RMSnorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        
    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps) # as an alternative can also use the frobenius norm to compute rms
        x_norm = x / rms
        
        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm
```

For more mathematical and implementation details about RMSNorm and its differences with Batch Normalization and Layer Normalization, refer to our dedicated [blog post](https://afterhoursresearch.hashnode.dev/batch-normalization-layer-normalization-and-root-mean-square-layer-normalization-a-comprehensive-guide-with-python-implementations).


### RoPE (Rotary Positional Embedding)
RoPE is based on rotating queries and keys in the attention mechanism, with a unique rotation at each position. This segment of code focuses on applying the rotation in a single attention block:

```python
R_matrix = self.R[:resize[1], :, :].to(query.device)
query_rot = torch.einsum('bhld,ldd->bhld', query.permute(0,2,1,3), R_matrix)
key_rot = torch.einsum('bhdl,ldd->bhdl', key.permute(0,2,3,1), R_matrix)
```
`self.R` is a pre-computed rotary matrix for positional encoding, `resize[1]` is the sequence length and is used to slice the rotary matrix to match the sequence length of the queries and keys.
the dimensions of query and key are ordered as [Batch size, Sequence length, Number of Heads, Hidden Dimension]. We permute these to rearrange the dimensions in a way that facilitates the subsequent operations. Specifically, we bring the sequence length (`l`) and dimension (`d`) next to each other for the rotation operation.
Let's now try to understand the `torch.einsum` operation! Here, the expression 'bhld,ldd->bhld' indicates the following:

`bhld`: Represents batch size (`b`), number of heads (`h`), sequence length (`l`), and hidden dimension (`d`) - of each head - for the query.
`ldd`: Stands for sequence length (`l`) and hidden dimension (`d`), twice to align with the square `R_matrix`.
`->bhld`: Tells us that the output should maintain the original dimensions of batch size, number of heads, sequence length, and dimension.
In this case, the `torch.einsum` function takes each slice along the `l` and `d` dimensions from `query`, multiplies it with the `R_matrix`, and sums along those dimensions. Because the output subscripts (`bhld`) are the same as the input, there is no reduction in dimensions—meaning, we get an output of the same shape as the `query`, but now each query vector has been rotated based on its position in the sequence.

For a deeper dive into RoPE, its mathematical formulation, and its practical implementation in PyTorch, check out our [blog post](https://afterhoursresearch.hashnode.dev/rope-rotary-positional-embedding).

### SwiGLU
SwiGLU is a combination of the Swish activation function and the GLU (Gated Linear Unit). Here's the essential code snippet for SwiGLU:

```python
class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linearA = nn.Linear(size, size)
        self.linearB = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        swish = self.linearB(x) * torch.sigmoid(self.beta * self.linearB(x))
        return swish * self.linearA(x)
```
Following the original Llama paper, for our experiments we size `size` to $\frac{2}{3}4d$, where $d$ is the hidden size (or dimension) of our Llama model. This can be easily change using the `model.swiglu_d_moltiplier` argument of hydra config.

Now, let's put everything together to see all the code for a single Llama attention block:
```python 
def causal_mask(size, device):
    x = torch.full(size, float("-inf"))
    return torch.triu(x, diagonal=1).to(device=device)
    
class LlamaSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        context_len: int,
        causal_attention: bool,
        n_heads: int,
        ):
        super().__init__()
        # self.embedding_size = torch.tensor(embedding_size)
        self.embedding_size = embedding_size
        self.causal_attention = causal_attention
        self.n_heads = n_heads
        assert self.embedding_size % self.n_heads == 0, f"Embedding size ({self.embedding_size}) must be divisable by the number of heads ({self.n_heads})"
        self.head_dim = self.embedding_size // self.n_heads
        self.R = get_rotary_matrix(context_len=context_len, embedding_dim=self.head_dim)   
        self.rms = RMSnorm(size=embedding_size)
        self.ff_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ff_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ff_v = nn.Linear(embedding_size, embedding_size, bias=False)
        
        self.fc1 = nn.Linear(embedding_size, embedding_size*2)
        self.activation = SwiGLU(size=embedding_size*2)
        self.fc2 = nn.Linear(embedding_size*2, embedding_size)
    
    def forward(self, x):
        input_shape = x.shape
        resize = (x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        x_res = x
        x = self.rms(x) # pre-normalization
        query = self.ff_q(x).reshape(resize)
        key = self.ff_k(x).reshape(resize)
        value = self.ff_v(x).reshape(resize)
                
        # Apply rotation to query and key, separatly for each head  
        R_matrix = self.R[:resize[1], :, :].to(query.device) 
        query_rot = torch.einsum('bhld,ldd->bhld', query.permute(0,2,1,3), R_matrix)
        key_rot = torch.einsum('bhdl,ldd->bhdl', key.permute(0,2,3,1), R_matrix)
        
        score = query_rot @ key_rot
        if self.causal_attention:
            score += causal_mask(size=score.shape, device=score.device)
        score = score / torch.sqrt(torch.tensor(self.head_dim)) 
        attention = torch.softmax(score, dim=-1) 
        x = attention @ value.permute(0,2,1,3)
        x = x.permute(0, 2, 1, 3).reshape(input_shape)
        x += x_res
        
        x_res = x
        x = self.rms(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + x_res
```

This reflects the architecture in the diagram before.

## Lightning Module

## Trainer 

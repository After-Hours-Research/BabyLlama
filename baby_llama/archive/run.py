from pathlib import Path
import torch
from transformers import AutoTokenizer
from baby_llama.utils import words_level_tokeniser
from baby_llama.models import SimpleModule, SimpleMLP, SimpleTransformer
from baby_llama.data.dataloader import CLMDataset, CLMDataModule
from baby_llama.train import ModelTrainer

# Simple Example
# text = ["A cat is on the table", "A dog sleeps"]
# Naive tokenizer
# tokenized_text = words_level_tokeniser(text)
# transformer = SimpleTransformer(9, 20, True, 2)
# model = SimpleModule(transformer)
# # model(tokenized_text)
# print(model.generate(tokenized_text, 50))


# Tiny Shakespeare Example
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = CLMDataset(
    Path("/home/sara/github_code/BabyLlama/data/tinyshakespeare.txt"),
    tokenizer=tokenizer, 
    tokenizer_args=dict(return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    )

datamodule = CLMDataModule(
    data=dataset, 
    train_ratio=0.6,
    val_ratio=0.2, 
    test_ratio=0.2, 
    train_batchsize=2, 
    val_test_batchsize=2
    )

transformer = SimpleTransformer(dataset.get_vocab_size, 1024, True, 2)
model = SimpleModule(transformer)

trainer = ModelTrainer(
    wandb_project_name="", 
    wandb_entity_name="", 
    model=model,
    data_module=datamodule,
    max_epochs=2
    )

trainer = trainer.train()

val = trainer.validate(model=model, data_module=datamodule)
test = trainer.test(data_module=datamodule)

breakpoint()
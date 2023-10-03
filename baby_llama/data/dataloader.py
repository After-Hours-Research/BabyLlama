from typing import TYPE_CHECKING
import datasets
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path

from torch.utils.data import random_split
from torch import Generator

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union

class CLMDataset(Dataset):
    def __init__(
        self,
        data: Path,
        tokenizer: AutoTokenizer,
        tokenizer_args: dict,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.tokenize()
    
    def tokenize(self):
        """
        Tokenizes the data using the provided tokenizer and tokenizer_args.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokens = self.tokenizer(self.data, **self.tokenizer_args)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        """
        Given an index, returns the corresponding data point from the tokenized dataset.
        """
        return self.tokens["input_ids"][idx, :-1], self.tokens["input_ids"][idx, 1:], self.tokens["attention_mask"][idx, :-1]

    def get_text(self, idx: int) -> str:
        """
        Given an index, returns the corresponding text from the original dataset.
        """
        return self.data[idx]
    
    @property
    def get_vocab_size(self) -> int:
        """
        Returns the vocabulary size.
        """
        # tokenizer.vocab_size is a fixed attribute, referring to the base vocabulary without any additional tokens - doesnt change when using tokenizer.add_special_tokens
        return len(self.tokenizer)

class CLMDataModule(pl.LightningDataModule):
    def __init__(
      self,
      data: CLMDataset,
      train_ratio: float,
      val_ratio: float,
      test_ratio: float,
      train_batchsize: int,
      val_test_batchsize: int,
      seed: int = 1,
    ):
        super().__init__()
        self.data = data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_batchsize = train_batchsize
        self.val_test_batchsize = val_test_batchsize
        pl.seed_everything(seed)

    def setup(self, stage):
        train, val, test = random_split(
            dataset=self.data, 
            lengths=[self.train_ratio, self.val_ratio, self.test_ratio]
        )
        if stage == "fit":
            self.train, self.val = train, val
        if stage == "test":
            self.test = test
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batchsize, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_test_batchsize, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.val_test_batchsize, shuffle=False)

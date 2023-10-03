from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    path: str
    train_ratio: float
    val_ratio: float 
    test_ratio: float 
    train_batchsize: int 
    val_test_batchsize: int
    
@dataclass
class Model:
    context_len: int
    hidden_size: int
    n_heads: int
    n_blocks: int
    
@dataclass
class Config:
    dataset: DatasetConfig
    model: Model
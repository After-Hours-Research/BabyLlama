from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    path: str
    tokenizer_path: str
    train_ratio: float
    val_ratio: float 
    test_ratio: float 
    
@dataclass
class ModelConfig:
    context_len: int
    hidden_size: int
    n_heads: int
    n_blocks: int
    
@dataclass
class TrainerConfig:
    max_epochs: int 
    check_val_every_n_epoch: int
    num_workers: int
    train_batchsize: int
    val_test_batchsize: int
    
@dataclass
class Config:
    wandb_project_name: str
    wandb_entity_name: str
    wandb_disable_log: bool
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
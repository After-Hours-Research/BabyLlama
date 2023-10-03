from omegaconf import DictConfig, OmegaConf
import hydra

from pathlib import Path
import torch
from transformers import AutoTokenizer
from baby_llama.models import SimpleModule, Llama
from baby_llama.data.dataloader import CLMDataModule
from baby_llama.data.utils import getfromtext
from baby_llama.train import ModelTrainer
from baby_llama.config import Config
from torch import nn

# root_dir = Path(__file__).parent

@hydra.main(config_path="config", config_name="parent.yaml", version_base=None)
def main(cfg: Config) -> None:    
    max_epochs = 2 # TODO: move it to config
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = getfromtext(
        data_path=Path(cfg.dataset.path),
        tokenizer=tokenizer, 
        tokenizer_args=dict(return_tensors="pt", truncation=True, padding="max_length", max_length=cfg.model.context_len+1)) # 1 more to account for the label-shifting
    
    datamodule = CLMDataModule(
        data=dataset, 
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio, 
        test_ratio=cfg.dataset.test_ratio, 
        train_batchsize=2, 
        val_test_batchsize=2
        )
    transformer = Llama(
        vocab_size=dataset.get_vocab_size, 
        hidden_size=cfg.model.hidden_size, 
        context_len=cfg.model.context_len, 
        causal_attention=True, 
        n_heads=cfg.model.n_heads, 
        n_blocks=cfg.model.n_blocks
        )
    model = SimpleModule(transformer)

    trainer = ModelTrainer(
        wandb_project_name="", 
        wandb_entity_name="", 
        model=model,
        data_module=datamodule,
        max_epochs=max_epochs
        )

    trainer = trainer.train()

    val = trainer.validate(model=model, data_module=datamodule)
    test = trainer.test(model=model, data_module=datamodule)

    breakpoint()
    
    
if __name__ == "__main__":
    main()
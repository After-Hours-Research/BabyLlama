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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# root_dir = Path(__file__).parent

@hydra.main(config_path="config", config_name="parent.yaml", version_base=None)
def main(cfg: Config) -> None:    
    tokenizer_name = "gpt2" if cfg.dataset.tokenizer_path is None else cfg.dataset.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    dataset = getfromtext(
        data_path=Path(cfg.dataset.path),
        tokenizer=tokenizer, 
        tokenizer_args=dict(return_tensors="pt", add_special_tokens=True, truncation=True, padding="max_length", max_length=cfg.model.context_len+1)) # 1 more to account for the label-shifting
   
    datamodule = CLMDataModule(
        data=dataset, 
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio, 
        test_ratio=cfg.dataset.test_ratio, 
        train_batchsize=cfg.trainer.train_batchsize, 
        val_test_batchsize=cfg.trainer.val_test_batchsize,
        num_workers=cfg.trainer.num_workers
        )
    
    transformer = Llama(
        vocab_size=dataset.get_vocab_size, 
        hidden_size=cfg.model.hidden_size, 
        context_len=cfg.model.context_len, 
        causal_attention=True, 
        n_heads=cfg.model.n_heads, 
        n_blocks=cfg.model.n_blocks,
        swiglu_d_moltiplier=cfg.model.swiglu_d_moltiplier
        )
    
    model = SimpleModule(
        transformer, 
        tokenizer=tokenizer
        )

    # Initialize Learning Rate Monitor callback
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    # Initialize ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=False,  # If you want to save the last model in addition to the best.
        filename='{epoch}-{val_loss:.2f}',
        auto_insert_metric_name=False  # To avoid prepending monitored metric name to filename
    )
    
    modeltrainer = ModelTrainer(
        wandb_project_name=cfg.wandb_project_name, 
        wandb_entity_name=cfg.wandb_entity_name, 
        wandb_disable_log=cfg.wandb_disable_log, 
        model=model,
        datamodule=datamodule,
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        callbacks=[lr_monitor_callback, checkpoint_callback]
        )
    
    # Update experiment config on wandb
    modeltrainer.wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))

    trainer = modeltrainer.train()

    val = trainer.validate(model=model, datamodule=datamodule)
    
    generation_config = {"greedy": {"temperature":1, "top_k":0, "top_p":0.0, "greedy":True}, 
                         "rnd_sampling": {"temperature":1, "top_k":0, "top_p":0.0, "greedy":False},
                         "rnd_sampling_t": {"temperature":0.7, "top_k":0, "top_p":0.0, "greedy":False},
                         "topk_sampling": {"temperature":1, "top_k":40, "top_p":0.0, "greedy":False},
                         "topk_sampling_t": {"temperature":0.7, "top_k":40, "top_p":0.0, "greedy":False},
                         "topp_sampling": {"temperature":1, "top_k":0, "top_p":0.9, "greedy":False},
                         "topp_sampling_t": {"temperature":0.7, "top_k":0, "top_p":0.9, "greedy":False},
                         }

    for _ in range(2):
        for conf_k, conf_v in generation_config.items():
            _, outputs_decoded = model.generate(context_len=cfg.model.context_len, max_output_token=300, **conf_v)
            print(f"\nFull Text, {conf_k}: \n{outputs_decoded}")
            
    modeltrainer.wandb_close()

if __name__ == "__main__":
    main()
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb


class ModelTrainer:
    def __init__(
        self, 
        wandb_project_name,
        wandb_entity_name,
        wandb_disable_log,
        model, 
        datamodule, 
        max_epochs,
        check_val_every_n_epoch,
        callbacks
        ):

        self.wandb_project_name = wandb_project_name
        self.wandb_entity_name = wandb_entity_name
        self.wandb_disable_log = wandb_disable_log
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.callbacks = callbacks

        self.wandb_logger = self._wandb_init()
        self.wandb_logger.watch(self.model)

    def _wandb_init(self):
        return WandbLogger(project=self.wandb_project_name, entity=self.wandb_entity_name, offline=self.wandb_disable_log)
    
    def wandb_close(self):
        self.wandb_logger.experiment.unwatch(self.model)
        self.wandb_logger.experiment.finish()

    def train(self):
        trainer = Trainer(
            max_epochs=self.max_epochs, 
            callbacks=self.callbacks, 
            logger=self.wandb_logger, 
            check_val_every_n_epoch=self.check_val_every_n_epoch, 
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=None
            )
        trainer.fit(model=self.model, datamodule=self.datamodule)
        return trainer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


class ModelTrainer:
    def __init__(
        self, 
        wandb_project_name,
        wandb_entity_name,
        model, 
        datamodule, 
        max_epochs,
        callbacks=None
        ):

        self.wandb_project_name = wandb_project_name
        self.wandb_entity_name = wandb_entity_name
        self.model = model
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.callbacks = callbacks

        self.wandb_logger = self._wandb_init()

    def _wandb_init(self):
        return WandbLogger(project=self.wandb_project_name, entity=self.wandb_entity_name, offline=True)
    
    def wandb_close(self):
        self.wandb_logger.experiment.finish()

    def train(self):
        trainer = Trainer(max_epochs=self.max_epochs, callbacks=self.callbacks)
        trainer.fit(model=self.model, datamodule=self.datamodule)
        return trainer
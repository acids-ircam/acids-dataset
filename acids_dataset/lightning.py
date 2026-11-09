"""Lightning integration of curricula and loss trackers (see acids_dataset.sampling).

    from acids_dataset.sampling import Curriculum, LossTracker
    from acids_dataset.lightning import CurriculumCallback

    tracker = LossTracker(len(dataset))
    curriculum = Curriculum(tracker, start=0.3, length=20)
    dataset.output_pattern = "{waveform,index}"                         # item indices are needed to record losses
    loader = dataset.get_loader(batch_size=32, curriculum=curriculum, seed=0)
    trainer = pl.Trainer(callbacks=[CurriculumCallback(curriculum, tracker)])

In training_step, return the losses of items with the loss : {"loss": loss.mean(), "item_losses": loss.detach()} ; indices
are read from the "index" entry of batches (or from outputs["index"]). Epochs are given to samplers by Lightning (set_epoch) ;
the callback sets steps of step-based curricula, records losses (gathered over processes in distributed training), logs the
competence and loss statistics, and saves its state in checkpoints."""
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch

from .sampling import Curriculum, LossTracker


class CurriculumCallback(pl.Callback):
    """
    Args:
        curriculum (Curriculum | None): curriculum whose steps (and epochs) are updated.
        tracker (LossTracker | None): loss tracker updated with the losses of items.
        loss_key (str): key of item losses in the outputs of training_step.
        index_key (str): key of item indices in batches (dict batches) or in the outputs of training_step.
        index_position (int | None): position of item indices in tuple batches.
        log (bool): logs the competence and the mean recorded loss.
        name (str): name of the callback in checkpoints (several CurriculumCallback need different names).
    """

    def __init__(self, curriculum: Curriculum | None = None, tracker: LossTracker | None = None, loss_key: str = "item_losses",
                 index_key: str = "index", index_position: int | None = None, log: bool = True, name: str = "curriculum"):
        super().__init__()
        # name identifies the state of the callback in checkpoints (several callbacks need different names)
        self.name = name
        self.curriculum = curriculum
        self.tracker = tracker if tracker is not None else (curriculum.difficulty if curriculum is not None and isinstance(curriculum.difficulty, LossTracker) else None)
        self.loss_key, self.index_key, self.index_position = loss_key, index_key, index_position
        self.log = log

    def on_train_epoch_start(self, trainer, pl_module):
        if self.curriculum is not None:
            self.curriculum.set_epoch(trainer.current_epoch)

    def _indices(self, outputs, batch):
        if isinstance(outputs, dict) and self.index_key in outputs:
            return outputs[self.index_key]
        if isinstance(batch, dict) and self.index_key in batch:
            return batch[self.index_key]
        if isinstance(batch, (list, tuple)) and self.index_position is not None:
            return batch[self.index_position]
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.curriculum is not None and self.curriculum.unit == "step":
            self.curriculum.step(trainer.global_step)
        if self.tracker is None or not isinstance(outputs, dict) or self.loss_key not in outputs:
            return
        indices = self._indices(outputs, batch)
        if indices is None:
            raise KeyError("CurriculumCallback : item indices not found ; add an 'index' entry to the output pattern of the dataset")
        losses = torch.as_tensor(outputs[self.loss_key]).detach().reshape(-1)
        indices = torch.as_tensor(indices).reshape(-1).to(losses.device)
        if trainer.world_size > 1:
            indices, losses = pl_module.all_gather(indices).reshape(-1), pl_module.all_gather(losses).reshape(-1)
        self.tracker.update(indices.cpu(), losses.cpu())

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.log:
            return
        if self.curriculum is not None:
            pl_module.log("curriculum/competence", float(self.curriculum.competence), rank_zero_only=True)
        if self.tracker is not None and bool(self.tracker.seen.any()):
            stats = self.tracker.statistics()
            pl_module.log("curriculum/mean_loss", stats['mean'], rank_zero_only=True)
            pl_module.log("curriculum/ratio_seen", stats['ratio_seen'], rank_zero_only=True)

    @property
    def state_key(self) -> str:
        return self._generate_state_key(name=self.name)

    def state_dict(self):
        state = {}
        if self.curriculum is not None: state['curriculum'] = self.curriculum.state_dict()
        if self.tracker is not None: state['tracker'] = self.tracker.state_dict()
        return state

    def load_state_dict(self, state):
        if self.curriculum is not None and 'curriculum' in state: self.curriculum.load_state_dict(state['curriculum'])
        if self.tracker is not None and 'tracker' in state: self.tracker.load_state_dict(state['tracker'])


__all__ = ['CurriculumCallback']

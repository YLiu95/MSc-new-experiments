import torch

from ranking17.model import RankingModel


class IdentityHiddenModel(RankingModel):
    def freeze_identity_reference(self):
        reference = self.ticker[:-1].detach().clone()
        self.register_buffer("_identity_reference", reference, persistent=False)

    @torch.no_grad()
    def restore_identity_reference(self):
        self.ticker[:-1].copy_(self._identity_reference)

    def forward(self, inputs, ticker_ids, market_ids, tasks, identity_masked=True):
        if self.training and not identity_masked:
            raise ValueError("Experiment 1.9.2 training must hide stock identity")
        return super().forward(inputs, ticker_ids, market_ids, tasks, identity_masked=identity_masked)
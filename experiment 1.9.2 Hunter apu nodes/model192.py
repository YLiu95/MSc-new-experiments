from ranking17.model import RankingModel


class IdentityHiddenModel(RankingModel):
    def forward(self, inputs, ticker_ids, market_ids, tasks, identity_masked=True):
        if self.training and not identity_masked:
            raise ValueError("Experiment 1.9.2 training must hide stock identity")
        return super().forward(inputs, ticker_ids, market_ids, tasks, identity_masked=identity_masked)
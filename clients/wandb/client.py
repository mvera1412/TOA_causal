import os
import wandb


class DummyClient:
    def __init__(self, **kwargs):
        print("Will not log any metrics for this entire run")

    def log(self, **kwargs):
        pass


class Client(DummyClient):
    def __init__(self, project: str, config: dict, tags: list = None, notes=None):
        """

        :param dict config:

        Example:
            >>> config = {
            >>>     "lr": 0.1,
            >>>     "batch_size": 32,
            >>>     "epochs": 4,
            >>> }
            >>> project = "detect-pedestrians"
            >>> notes = "tweak baseline"
            >>> tags = ["baseline", "paper1"]
            >>> wandb_client = Client(config=config, project="my-test-project", tags=["tag1","tag2"],
            >>> notes="this is a test")
            >>> wandb_client.log(metrics={"loss": 100}, step=0)
        """
        wandb.login()
        self.run = wandb.init(project=project, config=config, tags=tags, notes=notes, reinit=True)

    def log(self, metrics: dict, step=None):
        wandb.log(metrics, step=step)

    def finish_run(self):
        """To start a new run with different configs, for example, use this function"""
        self.run.finish()

import logging

import torch

from thirdparty.basnet import BASNet

log = logging.getLogger(__name__)


class BASNetWrapper(BASNet):
    def __init__(self, th, n_channels, n_classes):
        super().__init__(n_channels, n_classes)
        self.th = th

    def forward(self, x):
        output, *_ = super().forward(x)
        pred = output[:, 0, :, :]
        return self.normalize_predictions(pred) > self.th

    @staticmethod
    def normalize_predictions(pred):
        ma = torch.max(pred)
        mi = torch.min(pred)
        return (pred - mi) / (ma - mi)


def load_sal_model(path, model_type="supervised", th=0.5):
    log.info("Loading BASNet...")
    model = BASNetWrapper(th=th, n_channels=3, n_classes=1)
    if model_type == "supervised":
        model.load_state_dict(torch.load(path))
    elif model_type == "unsupervised":
        model.load_state_dict(torch.load(path)["net"])
    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

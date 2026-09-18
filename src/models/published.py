"""The network behind the published ChromeCRISPR checkpoint.

`artifacts/models/CNN_GRU_GC.pth` was saved as a pickled module, so the class has to
exist under the name it was pickled with before `torch.load` can rebuild it.
`load_published_chromecrispr` handles that and returns the model ready to evaluate.
"""
import torch
import torch.nn as nn

DROPOUT = 0.3388134505129095


class CNN_GRU_GC(nn.Module):
    """One convolution of 256 filters, one GRU of 256 units, GC appended at the head."""

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(alpha=1.0),
        )
        self.rnn = nn.GRU(256, 256, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(5377, 256), nn.ELU(alpha=1.0), nn.Dropout(p=DROPOUT),
            nn.Linear(256, 128), nn.ELU(alpha=1.0), nn.Dropout(p=DROPOUT),
            nn.Linear(128, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x, gc_content):
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat((x, gc_content.unsqueeze(1)), dim=1)
        return torch.sigmoid(self.fc_layers(x))


def load_published_chromecrispr(path, map_location="cpu"):
    """Rebuild the published checkpoint and return it in eval mode."""
    import __main__
    __main__.CNN_GRU_GC = CNN_GRU_GC          # the name the checkpoint was pickled under
    obj = torch.load(path, map_location=map_location, weights_only=False)
    model = obj if isinstance(obj, nn.Module) else CNN_GRU_GC()
    if not isinstance(obj, nn.Module):
        model.load_state_dict(obj)
    model.eval()
    return model


def predict(model, X, gc, device="cpu", chunk=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = torch.as_tensor(X[i:i + chunk], device=device)
            gb = torch.as_tensor(gc[i:i + chunk], device=device)
            out.append(model(xb, gb).squeeze(-1).cpu().numpy())
    import numpy as np
    return np.concatenate(out)

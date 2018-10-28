from library import *

# global params

n_features = 128
n_lde = 64

# 40 filterbanks plus two diffs
n_raw = 40
n_ch = 3

# ctc_loss weight for phoneme joint training
w_ctc = 1
use_gmm = False


# one backbone + many feature extractors

# share these layers across models to avoid duplicate cache
if use_gmm:
    gmm = torch.load('gmm.pkl')
    gmm_layer = GMM(gmm=gmm, disk=h5py.File('gmm_cache.h5'))
hcopy_layer = HCopy(config='HCopy_config', disk=h5py.File('hc_cache.h5'))

def backbone(n_features, feat_layers, pool_layers):
    return nn.Sequential(
        # raw features
        T(['names', 'roots'], 'xs', hcopy_layer),
        *([T(['names', 'xs'], 'gs', gmm_layer)] if use_gmm else []),

        # augmentation & normalization
        T('xs', ['x', 'cuts'], cat),
        T(['x', 'training'], 'x', Perturb(p_time=0.1, p_freq=0.1, freq_cut=0.8)),
        #T('x', 'x', lambda x: x - x.mean(0, keepdim=True)),
        #T(['x', 'cuts'], 'xs', Uncat()), # move to the end & dont use crop

        # cropping & feature engineering
        #T(['xs', 'training'], ['x', 'n_crops'], Crop(l=None if w_ctc > 0 else 500)),
        T('x', 'x', auto_dev),
        T('x', 'x', Diff(n=n_ch)),

        # neural net feature extractor
        T('x', 'x', lambda x: x.unsqueeze(0)),
        *feat_layers(),
        T('x', 'x', lambda x: x.squeeze(0)),
        T('x', 'ph_logits', nn.Linear(n_features, len(phonemes) + 1)),
        T(['x', 'cuts'], 'xs', uncat),
        T(['ph_logits', 'cuts'], 'ph_logits', uncat),
        *([T(['ph_logits', 'phonemes'], 'ctc_loss', CTCLoss(n_features, len(phonemes)))]
            if w_ctc > 0 else []),
        *pool_layers(),
        #T(['x', 'n_crops'], 'x', Uncrop()),

        # classifier & losses
        *([T(['x', 'gs'], 'x',
            lambda x, gs: torch.cat([x, torch.stack(gs, 0).to(x.device)], -1))]
            if use_gmm else []),
        T('x', 'x', nn.Dropout(0.5)), # dropout on all utterance level features
        T('x', 'logits',
            nn.Linear(n_features + (gmm.n_components if use_gmm else 0), len(dialects))),
        T(['logits', 'labels', 'training'], 'loss', celoss),
        *([T(['loss', 'ctc_loss', 'training'], 'loss',
            lambda a, b, t: a + w_ctc * b.to(a.device) if t else None)]
            if w_ctc > 0 else []),
    )


# shape reminder: bs x ts x fs x cs -> bs x ts' x fs'

def stacked_rnn():
    return [
        T('x', 'x', lambda x: x.view(x.size(0), x.size(1), -1)),
        T('x', ['x', 'h'], nn.LSTM(n_raw * n_ch, n_features >> 1, batch_first=True,  bidirectional=True)),
        T('x', ['x', 'h'], nn.LSTM(n_features, n_features >> 1, batch_first=True,  bidirectional=True)),
    ]

def BN(Layer, n):
    def f(n_in, n_out, **b):
        return nn.Sequential(
            Layer(n_in, n_out, **b),
            nn.ReLU(),
            getattr(nn, 'BatchNorm{}d'.format(n))(n_out),
        )
    return f

Conv1d = BN(nn.Conv1d, 1)
Conv2d = BN(nn.Conv2d, 2)

def dilated_cnn1d():
    return [
        T('x', 'x', lambda x: x.view(x.size(0), x.size(1), -1)),
        T('x', 'x', lambda x: x.transpose(1, 2)),
        T('x', 'x', nn.Sequential(
            Conv1d(n_raw * n_ch, n_features >> 3, kernel_size=5, dilation=1),
            Conv1d(n_features >> 3, n_features >> 2, kernel_size=5, dilation=2),
            Conv1d(n_features >> 2, n_features >> 1, kernel_size=5, dilation=4),
            Conv1d(n_features >> 1, n_features, kernel_size=5, dilation=8),
        )),
        T('x', 'x', lambda x: x.transpose(1, 2)),
    ]

def dilated_cnn2d():
    common = dict(kernel_size=3, padding=1, stride=(1, 2))
    return [
        T('x', 'x', lambda x: x.permute(0, 3, 1, 2)),
        T('x', 'x', nn.Sequential(
            Conv2d(n_ch, n_features >> 3, dilation=(1, 1), **common),
            Conv2d(n_features >> 3, n_features >> 2, dilation=(2, 1), **common),
            Conv2d(n_features >> 2, n_features >> 1, dilation=(4, 1), **common),
            Conv2d(n_features >> 1, n_features, kernel_size=(3, 5), dilation=(8, 1)),
        )),
        T('x', 'x', lambda x: x.permute(0, 2, 3, 1)),
        T('x', 'x', lambda x: x.view(x.size(0), x.size(1), -1)),
    ]


from torchvision.models.resnet import ResNet, BasicBlock

def resnet34():
    class Dummy():
        inplanes = n_features >> 3
    r = Dummy()
    layer = lambda *a, **b: ResNet._make_layer(r, BasicBlock, *a, **b)
    return [
        T('x', 'x', lambda x: x.permute(0, 3, 1, 2)),
        T('x', 'x', nn.Sequential(
            Conv2d(n_ch, n_features >> 3, kernel_size=(7, 7), padding=(3, 3), bias=False),
            layer(n_features >> 3, 3),
            layer(n_features >> 2, 4, stride=2),
            layer(n_features >> 1, 6, stride=2),
            layer(n_features, 3, stride=2),
            nn.AdaptiveAvgPool2d((None, 1)), # avgpool the freq. dim.
        )),
        T('x', 'x', lambda x: x.squeeze(3)),
        T('x', 'x', lambda x: x.transpose(1, 2)),
    ]


def grupool():
    return [
        T('x', ['x', 'h'], nn.GRU(n_features, n_features, batch_first=True)),
        T('h', 'x', lambda x: x.squeeze(0)),
    ]

def avgpool():
    return [
        T('x', 'x', lambda x: x.mean(1))
    ]

def avgpool_xs():
    return [
        T('xs', 'x', lambda xs: torch.stack([x.mean(0) for x in xs]))
    ]

def ldepool():
    return [
        T('x', 'x', LDE(n_features, n_lde))
    ]

def catpool():
    return [
        T('x', ['_', 'h'], nn.GRU(n_features, n_features, batch_first=True)),
        T('h', 'p1', lambda x: x.squeeze(0)),
        T('x', 'p2', lambda x: x.mean(1)),
        T('x', 'p3', LDE(n_features, n_lde)),
        T(['p1', 'p2', 'p3'], 'x', lambda *a: torch.cat(a, dim=-1)),
    ]


# finally, the grand ensemble!

models = [
    backbone(
        n_features * scale,
        feat_layers,
        pool_layers,
    )
    for pool_layers, scale in [
        #(avgpool, 1),
        #(ldepool, n_lde),
        #(grupool, 1),
        #(catpool, 2 + n_lde),
        (avgpool_xs, 1),
    ]
    for feat_layers in [
        #stacked_rnn,
        resnet34,
        dilated_cnn2d,
        dilated_cnn1d,
    ]
]

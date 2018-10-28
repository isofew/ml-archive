from functools import partial, reduce
from HTKfile import HTKfile
from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch
import h5py
import sys
import re
import os


# ### helpers

# the built-in os.system will get slower over calls,
# don't know why, use this work-around instead

qin = os.open('qin', os.O_WRONLY)
qout = os.open('qout', os.O_RDONLY)
def system(command):
    os.write(qin, bytes(command + '\n', 'ascii'))
    return os.read(qout, int(1e6))


def auto_dev(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def batches(data, batch_size, shuffle):
    idx = np.arange(data['size'])
    if shuffle:
        np.random.shuffle(idx)
    keys = [k for k in data.keys() if isinstance(data[k], np.ndarray)]
    values = [data[k][idx] for k in keys]
    batch = dict(size=0)
    for vals in zip(*values):
        for k, v in zip(keys, vals):
            if k in batch:
                batch[k].append(v)
            else:
                batch[k] = [v]
        batch['size'] += 1
        if batch['size'] == batch_size:
            yield batch
            batch = dict(size=0)
    if batch['size'] > 0:
        yield batch

def sample(data, ratio):
    return next(batches(data, batch_size=int(ratio * data['size']), shuffle=True))


iden = lambda x: x
comp = lambda f, g: lambda *a, **b: f(g(*a, **b))
chain = lambda *fs: reduce(comp, fs, iden)

fst = lambda t: t[0]
snd = lambda t: t[1]

def par(f, *a, **b):
    g = partial(f, *a, **b)
    g.__name__ = f.__name__
    return g
curry = lambda f: par(par, f)

lmap = lambda f, *a, **b: \
    list(map(partial(f if hasattr(f, '__call__') else f.__getitem__, **b), *a))
amap = comp(np.array, lmap)
mapped = curry(lmap)


# ### data

dialects = ['changsha', 'hebei', 'hefei', 'kejia', 'minnan', 'nanchang', 'ningxia', 'shan3xi', 'shanghai', 'sichuan']

phonemes = ['_a', '_e', '_i', '_o', '_u', '_v', 'a1', 'a2', 'a3', 'a4', 'ai1', 'ai2', 'ai3', 'ai4', 'an1', 'an2', 'an3', 'an4', 'ang1', 'ang2', 'ang3', 'ang4', 'ao1', 'ao2', 'ao3', 'ao4', 'b', 'c', 'ch', 'd', 'e1', 'e2', 'e3', 'e4', 'ei1', 'ei2', 'ei3', 'ei4', 'en1', 'en2', 'en3', 'en4', 'eng1', 'eng2', 'eng3', 'eng4', 'er2', 'er3', 'er4', 'f', 'g', 'h', 'i1', 'i2', 'i3', 'i4', 'ia1', 'ia2', 'ia3', 'ia4', 'ian1', 'ian2', 'ian3', 'ian4', 'iang1', 'iang2', 'iang3', 'iang4', 'iao1', 'iao2', 'iao3', 'iao4', 'ie1', 'ie2', 'ie3', 'ie4', 'ii1', 'ii2', 'ii3', 'ii4', 'iii1', 'iii2', 'iii3', 'iii4', 'iiii4', 'in1', 'in2', 'in3', 'in4', 'ing1', 'ing2', 'ing3', 'ing4', 'iong1', 'iong2', 'iong3', 'iong4', 'iou1', 'iou2', 'iou3', 'iou4', 'j', 'k', 'l', 'm', 'n', 'o1', 'o2', 'o3', 'o4', 'ong1', 'ong2', 'ong3', 'ong4', 'ou1', 'ou2', 'ou3', 'ou4', 'p', 'q', 'r', 's', 'sh', 't', 'u1', 'u2', 'u3', 'u4', 'ua1', 'ua2', 'ua3', 'ua4', 'uai1', 'uai2', 'uai3', 'uai4', 'uan1', 'uan2', 'uan3', 'uan4', 'uang1', 'uang2', 'uang3', 'uang4', 'uei1', 'uei2', 'uei3', 'uei4', 'uen1', 'uen2', 'uen3', 'uen4', 'ueng1', 'uo1', 'uo2', 'uo3', 'uo4', 'v', 'v1', 'v2', 'v3', 'v4', 'van1', 'van2', 'van3', 'van4', 've1', 've2', 've3', 've4', 'vn1', 'vn2', 'vn3', 'vn4', 'x', 'z', 'zh']

ph_to_id = {p: i for i, p in enumerate(phonemes)}

def parse_mlf(file_path, excludes=set(), name_transform=iden):
    d = {}
    with open(file_path, 'r') as f:
        c = ()
        for line in f:
            line = line.strip()
            if c == ():
                if line[0] == line[-1] == '"':
                    c = name_transform(line[1:-1]), []
            else:
                if line == '.':
                    d[c[0]] = c[1]
                    c = ()
                elif line not in excludes:
                    c[1].append(line)
    return d

def load_phonemes(mlf_path, names):
    d = parse_mlf(
        mlf_path,
        excludes=set(['<s>', '</s>']),
        name_transform=lambda s: s[2:-4] #+ '.pcm'
    )
    phonemes = amap(mapped(ph_to_id), lmap(d, names))
    return phonemes

def load_data(data_dir, predicate=lambda _: True, shuffle=True, mlf_path=None):
    paths = [
        (r, f[:-4])
        for r, _, fs in os.walk(data_dir)
        for f in fs
        if f[-4:] == '.pcm'
        if predicate(os.path.join(r, f))
    ]
    idx = np.arange(len(paths))
    if shuffle:
        np.random.shuffle(idx)
    data = dict(
        size = len(paths),
        roots = amap(fst, paths)[idx],
        names = amap(snd, paths)[idx],
        labels = amap(chain(dialects.index, fst, re.compile('_').split, snd), paths)[idx],
    )
    if mlf_path is not None:
        data['phonemes'] = load_phonemes(mlf_path, data['names'])
    else:
        data['phonemes'] = np.array([[-1]] * data['size'])
    return data


# ### decorators

class Typed(nn.Module):
    def __init__(self, k_ins, k_outs, f):
        super(Typed, self).__init__()
        if type(k_ins) is not list:
            k_ins = [k_ins]
        if type(k_outs) is not list:
            k_outs = [k_outs]
        self.k_ins = k_ins
        self.k_outs = k_outs
        self.f = f

    def __repr__(self):
        return '{} : {} -> {}'.format(repr(self.f), self.k_ins, self.k_outs)

    def forward(self, data):
        data['training'] = self.training
        v_outs = self.f(*(data[k] for k in self.k_ins))
        if len(self.k_outs) == 1:
             data[self.k_outs[0]] = v_outs
        else:
            for k, v in zip(self.k_outs, v_outs):
                data[k] = v
        return data

T = Typed


class Lambda(nn.Module):
    def __init__(self, builder, **param):
        super(Lambda, self).__init__()
        self.builder = builder
        self.param = param
        self.forward = builder(**param)

    def __repr__(self):
        return '{}({})'.format(
            self.forward.__name__,
             ', '.join(k + '=' + str(v) for k, v in self.param.items())
        )

module = comp(curry(Lambda), curry)


# cache output from a function, key on the first argument
def cached(f):
    mem = {}
    def g(x, *xs, disk=None, **p):
        if x not in mem:
            if x not in disk:
                disk[x] = mem[x] = f(x, *xs, **p)
            else:
                mem[x] = torch.FloatTensor(disk[x].value)
        return mem[x]
    return g


# ### nn Modules

# note. the default shape is
# batch x time x ... (x channel)
# transpose is needed for cnns


@module
@mapped
@cached
def HCopy(name, root, config):
    src = os.path.join(root, name + '.pcm')
    dst = name + '.tmp'
    system('HCopy -C {} {} {}'.format(config, src, dst))
    ret = HTKfile(dst).read_data()
    ret -= ret.mean(0, keepdim=True)
    system('rm ' + dst)
    return ret


@module
@mapped
@cached
def GMM(_, x, gmm):
    return torch.FloatTensor( gmm.predict_proba(x).mean(0) )


def cat(xs):
    cuts = np.cumsum([0, *map(len, xs)]).astype(np.float32)
    return torch.cat(xs, 0), cuts / cuts[-1]


def uncat(x, cuts):
    return [
        x[int(round(l * len(x))) : int(round(r * len(x)))]
        for l, r in zip(cuts[:-1], cuts[1:])
    ]

def pack(xs):
    pass # TODO return packed, permute

def unpack(p):
    n = int( p.batch_sizes[0] )
    idx = [[] for _ in range(n)]
    i = 0
    for bs in p.batch_sizes:
        for j, k in enumerate(range(i, i + bs)):
            idx[j].append(k)
        i += bs
    return [p.data[ix] for ix in idx] # TODO permute back


def rand(p):
    return 1 - p + 2 * p * np.random.rand()

def resize2d(x, size):
    return F.interpolate(
        x.unsqueeze(0).unsqueeze(0), size=size, mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)

@module
def Perturb(x, training, p_time, p_freq, freq_cut):
    if training:
        ts, fs = x.shape
        t_scale = rand(p_time)
        ts_ = int(ts * t_scale)
        f_scale = rand(p_freq)
        f_cut = int(fs * freq_cut)
        f_cut_ = int(fs * freq_cut * f_scale)
        x_ = torch.empty(ts_, fs).to(x.device)
        x_[:, :f_cut_] = resize2d(x[:, :f_cut], [ts_, f_cut_])
        x_[:, f_cut_:] = resize2d(x[:, f_cut:], [ts_, fs - f_cut_])
        return x_
    else:
        return x


def random_slice(x, l, i=None):
    if i is None:
        i = np.random.randint(len(x) - l + 1)
    return torch.arange(i, i + l)

def crop(x, l, i=None):
    if len(x) < l:
        y = torch.zeros((l, *x.shape[1:]), dtype=x.dtype).to(x.device)
        y[random_slice(y, len(x), i)] = x
        return y
    else:
        return x[random_slice(x, l, i)]

@module
def Crop(xs, training, l=None):
    if l is None:
        l = max(map(len, xs))
    if training:
        n_crops = [1] * len(xs)
        X = torch.stack([
            crop(x, l)
            for x in xs
        ])
    else:
        n_crops = [int(np.ceil(len(x) / l)) for x in xs]
        X = torch.stack([
            crop(x, l, int(i))
            for x, n in zip(xs, n_crops)
            for i in np.linspace(np.abs(len(x) - l), 0, n)
        ])
    return X, n_crops

Padding = lambda: Crop()


@module
def Uncrop(X, n_crops):
    s = np.cumsum([0, *n_crops])
    return torch.stack([
        X[s:e].mean(0) # averaging the cropped parts
        for s, e in zip(s[:-1], s[1:])
    ])


class LDE(nn.Module):
    def __init__(self, n_in, n_out):
        super(LDE, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.s = nn.Parameter(torch.randn(n_out) / np.sqrt(n_out))
        self.m = nn.Parameter(torch.randn(n_in, n_out) / np.sqrt((n_in + n_out) / 2))
    
    def forward(self, x):
        # x: bs x ts x fs
        # m: fs x cs
        # r: bs x ts x cs x fs
        # s: cs
        # w: bs x ts x cs
        # e: bs x cs x fs
        r = (x.unsqueeze(-1) - self.m.unsqueeze(0).unsqueeze(0)).transpose(2, 3) # compute the residuals
        w = F.softmax((r * r).sum(-1) * self.s.view(1, 1, -1), 2) # softmax over components
        e = (w.unsqueeze(-1) * r).mean(1) # average over time
        return e.view(e.size(0), -1) # flattened


@module
def Diff(x, n):
    ys = [x]
    for _ in range(n - 1):
        x = ys[-1]
        y = torch.zeros_like(x)
        y[:, :-1] = x[:, 1:] - x[:, :-1]
        ys.append(y)
    return torch.stack(ys, -1)


def celoss(logits, labels, training):
    if training:
        return F.cross_entropy(logits, torch.LongTensor(labels).to(logits.device))
    else:
        return None

class CTCLoss(nn.Module):
    def __init__(self, n_features, n_phonemes):
        super(CTCLoss, self).__init__()
        try:
            from warpctc_pytorch import CTCLoss as C
            self.ctc_loss = C(length_average=True)
        except:
            #print('CTCLoss failed to load, set to None', flush=True)
            self.ctc_loss = None
        #self.ph_layer = nn.Linear(n_features, n_phonemes + 1)
    
    def forward(self, ph_logits, phonemes):
        if self.training and self.ctc_loss is not None:
            # acts: ts x bs x ps
            # labels: 1d array, starting from 1
            acts = nn.utils.rnn.pad_sequence(ph_logits)
            labels = torch.IntTensor( np.concatenate(phonemes) ) + 1
            act_lens = torch.IntTensor( amap(len, ph_logits) )
            label_lens = torch.IntTensor( amap(len, phonemes) )
            short = act_lens <= label_lens
            act_lens[short] = label_lens[short] + 1
            loss = self.ctc_loss(acts, labels, act_lens, label_lens)
            return loss
#            if torch.isnan(loss).any():
 #               print('nan on', acts, labels, act_lens, label_lens, act_lens>label_lens, flush=True)
  #              return torch.FloatTensor(0)
   #         else:
    #            return loss
        else:
            return None


# ### training tools


# fit one epoch
from collections import defaultdict

def fit(model, data, bs=32, lr=0.1, logs=set()):
    auto_dev(model)
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    running = defaultdict(lambda: 0)
    logs.add('loss') # always log the loss
    with tqdm(total=data['size']) as pbar:
        for batch in batches(data, bs, shuffle=True):
            batch = model(batch)
            loss = batch['loss']
            optim.zero_grad()
            loss.backward()
            optim.step()
            for k in batch.keys():
                if k in logs:
                    running[k] += float(batch[k]) * batch['size']
            pbar.update(batch['size'])
            pbar.set_description(' '.join(
                '%s: %.4f' % (k, v / pbar.n)
                for k, v in running.items()
            ))
    return running['loss'] / data['size']


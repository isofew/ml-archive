from library import *
from recipes import models

_, data_dir, model_path = sys.argv

# select dataset by regex
#data_dir = os.path.join('..', 'dataset')
mlf_path = 'all.mlf'
#if not os.path.isfile(mlf_path):
#    system("cat $( find {} -name '*.mlf' ) >all.mlf".format(data_dir))

data = load_data(data_dir, mlf_path=mlf_path)

# training params
max_epochs = 50
lr_start = 0.05
lr_end = 1e-3
bs_start = 64
# dynamic lr based on loss
lrs = None
# fixed lr scheme
#lrs = [0.1] * 10 + [0.01] * 10
#max_epochs = len(lrs)

if os.path.isfile(model_path):
    states = torch.load(model_path, map_location='cpu')
else:
    states = []

for i, model in enumerate(models):
    print('Model', i)
    print(model)
    if i < len(states):
        print('state loaded from disk')
        model.load_state_dict(states[i])
    else:
        torch.cuda.empty_cache()
        losses = []
        lr = lr_start
        bs = bs_start
        while bs >= 1 and lr >= lr_end and len(losses) < max_epochs:
            print('Epoch', len(losses) + 1, 'bs', bs, 'lr', lr, flush=True)
            try:
                losses.append( fit(model, data, bs, lr, logs=set(['ctc_loss'])) )
                if lrs is not None and len(losses) < len(lrs):
                    lr = lrs[ len(losses) ]
                elif len(losses) >= 2 and losses[-1] > losses[-2]:
                    lr /= 2
            except (RuntimeError, MemoryError) as e:
                print('halve batchsize on error', e)
                bs /= 2
            print()
        states.append(model.cpu().state_dict())
        torch.save(states, model_path)
    print()

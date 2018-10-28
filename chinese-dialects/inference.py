from library import *
from recipes import models

np.random.seed(0)

_, data_dir, model_path, result_path = sys.argv
#data_dir = os.path.join('..', 'dataset')
data = load_data(data_dir)
print(data)

states = torch.load(model_path, map_location='cpu')
models = models[:len(states)]
print('loaded', len(models), 'models')

preds = torch.empty(len(models), data['size'], len(dialects))

with torch.no_grad():
    for i, (model, state) in enumerate(zip(models, states)):
        model.load_state_dict(state)
        model.eval()
        auto_dev(model)
        logits = []
        with tqdm(total=data['size']) as pbar:
            for batch in batches(data, batch_size=64, shuffle=False):
                logits.append( model(batch)['logits'].cpu() )
                pbar.update(batch['size'])
        pred = F.softmax(torch.cat(logits, 0), 1)
        preds[i] = pred ** (1 / len(models))

labels = torch.prod(preds, 0).max(1)[1]

with open(result_path, 'w') as f:
    for name, i in zip(data['names'], labels):
        f.write(f'{name}.pcm\t{dialects[i]}\n')

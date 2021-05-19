import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import Trainer

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--min_vocab_freq', type=int, default=5)
    p.add_argument('--max_vocab_size', type=int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256)

    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])

    config = p.parse_args()

    return config


def main(config):
    loaders = DataLoader(
        train_fn=config.train_fn,
        batch_size=config.batch_size,
        min_freq=config.min_vocab_freq,
        max_vocab=config.max_vocab_size,
        device=config.gpu_id
    )

    print(
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    )
    
    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    # Declare model and loss.
    model = CNNClassifier(
        input_size=vocab_size,
        word_vec_size=config.word_vec_size,
        n_classes=n_classes,
        use_batch_norm=config.use_batch_norm,
        dropout_p=config.dropout,
        window_sizes=config.window_sizes,
        n_filters=config.n_filters,
    )
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()
    print(model)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    cnn_trainer = Trainer(model, optimizer, crit)
    cnn_model = cnn_trainer.train(
        loaders.train_loader,
        loaders.valid_loader,
        config
    )

    torch.save({
        'cnn': cnn_model.state_dict(),
        'config': config,
        'vocab': loaders.text.vocab,
        'classes': loaders.label.vocab,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
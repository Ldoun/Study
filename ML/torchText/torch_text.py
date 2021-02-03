from data_loader import DataLoader
import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=False)
    p.add_argument('--gpu_id', type= int,default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--valid_ratio', type=float,default=0.2)

    p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--n_epochs',type=int,default=20)
    p.add_argument('--verbose',type=int,default=2)
    
    p.add_argument('--train_fn')


    config = p.parse_args()

    return config

def main(config):
    loaders = DataLoader(
        train_fn= config.train_fn,
        batch_size= config.batch_size,
        valid_ratio=config.valid_ratio,
        device=-1,
        max_vocab=999999,
        min_freq=5,
    )

    print("|train|= {train_len}\n|valid| = {valid_len}".format(train_len = len(loaders.train_loader.dataset),valid_len = len(loaders.valid_loader.dataset)))

    print("|vocab| = {vocab}\n|label| = {label}".format(vocab = len(loaders.text.vocab), label = len(loaders.label.vocab)))

    data = next(iter(loaders.train_loader))

    print(data.text.shape)
    print(data.label.shape)

    for i in range(50):
        word = loaders.text.vocab.itos[i]
        print("%5d: %s\t%d" % (i,word,loaders.text.vocab.freqs[word]))

if __name__ == '__main__' :
    config = define_argparser()
    main(config)
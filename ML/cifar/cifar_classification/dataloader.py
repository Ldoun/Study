import torch

from torch.utils.data import Dataset, DataLoader
 
import torchvision
import torchvision.transforms as transforms

class cifarDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x,y  


def load_cifar():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
    print(type(trainset.data))

    return torch.FloatTensor(trainset.data).transpose(1,3), torch.tensor(trainset.targets),torch.FloatTensor(testset.data).transpose(1,3), torch.tensor(testset.targets)

def get_loaders(config):
    x,y,test_x,test_y = load_cifar()
    print(x.shape)
    train_cnt = int(x.shape[0] * config.train_ratio)
    valid_cnt = x.shape[0] - train_cnt

    indices = torch.randperm(x.shape[0])

    train_x, valid_x = torch.index_select(x,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)
    train_y, valid_y = torch.index_select(y,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)

    train_loader = DataLoader(
        dataset= cifarDataset(train_x, train_y),
        batch_size= config.batch_size,
        shuffle= True)

    valid_loader = DataLoader(
        dataset= cifarDataset(valid_x, valid_y),
        batch_size= config.batch_size,
        shuffle= False)

    test_loader = DataLoader(
        dataset = cifarDataset(test_x,test_y),
        batch_size = config.batch_size,
        shuffle = False
    )

    return train_loader,valid_loader,test_loader


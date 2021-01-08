def load_mnist(is_Train = True , flatten = True):
    from torchvision import datasets,transforms

    datasets = datasets.MNIST(
        '../data', train = is_Train,download=True,
        transform=transforms.Compose([transforms.ToTensor,])
    )

    x = datasets.data.float() / 255.
    y = datasets.targets

    if flatten:
        x = x.view(x.size(0),-1)


    return x, y
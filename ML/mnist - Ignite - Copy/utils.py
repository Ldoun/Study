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

def get_grad_norm(parameters, norm_type =2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm +=param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def get_parameter_norm(parameters, norm_type = 2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm +=param_norm ** norm_type
        total_norm = total_norm ** (1./ norm_type)
    except Exception as e:
        print(e)

    return total_norm
    
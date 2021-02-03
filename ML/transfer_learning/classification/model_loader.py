from torchvision import models
from torch import nn

def set_parameter_requires_grad(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze

def get_model(config):
    model  = None
    input_size = 0
    if config.model_name == 'resnet':
        model = models.resnet34(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model,config.freeze)

        n_feature = model.fc.in_features
        model.fc = nn.Linear(n_feature,config.n_classes)
        input_size = 224

    elif config.model_name == 'alexnet':
        model = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model,config.freeze)

        n_feature = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_feature,config.n_classes)
        input_size = 224    

    elif config.model_name == 'vgg':
        model = models.vgg16_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model,config.freeze)

        n_feature = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_feature,config.n_classes)
        input_size = 224    

    elif config.model_name == 'squeezenet':
        model = models.squeezenet1_0(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model,config.freeze)

        model.classifier[1] = nn.Conv2d(
            512,
            config.n_classes,
            kernel_size=(1,1),
            stride=(1,1)
        )
        model.num_classes = config.n_classes
        input_size = 224    

    elif config.model_name == 'densenet':
        model = models.densenet121(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model,config.freeze)

        n_feature = model.classifier.in_features
        model.classifier = nn.Linear(n_feature,config.n_classes)
        input_size = 224  
    else:
        raise NotImplementedError('You need to specify model name.')

    return model, input_size  
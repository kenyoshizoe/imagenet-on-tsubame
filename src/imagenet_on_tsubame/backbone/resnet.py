import torch
import torchvision


def build_model(arch='resnet18',
                weights='IMAGENET1K_V1',
                model_path='',
                dict_model_key='module.'):
    if weights == '':
        weights = None

    if arch == 'resnet18':
        model = torchvision.models.resnet18(weights=weights)
    elif arch == 'resnet34':
        model = torchvision.models.resnet34(weights=weights)
    elif arch == 'resnet50':
        model = torchvision.models.resnet50(weights=weights)
    elif arch == 'resnet101':
        model = torchvision.models.resnet101(weights=weights)
    elif arch == 'resnet152':
        model = torchvision.models.resnet152(weights=weights)
    else:
        raise ValueError(f'Unsupported architecture: {arch}')
    
    model.fc = torch.nn.Identity()

    if model_path != '':
        state_dict = torch.load(model_path, weights_only=True)['state_dict']
        # Remove the prefix from the keys in the state_dict and remove the last layer
        state_dict = {key[len(dict_model_key):]: val for key,
                      val in state_dict.items() if key.startswith(dict_model_key) and not key.startswith(f'{dict_model_key}fc.')}
        model.load_state_dict(state_dict)
    return model

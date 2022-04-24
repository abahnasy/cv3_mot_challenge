from .resnet import (
    resnet18, resnet34, resnet50, resnet50_fc512
)

def build_model(
    name, num_classes, loss='softmax', pretrained=True, use_gpu=True
):
    """A function wrapper for building a model.
    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.
    Returns:
        nn.Module
    """
    if name == 'resnet50':
        model_fn = resnet50
    elif name == 'resnet18':
        model_fn = resnet18
    elif name == 'resnet34':
        model_fn = resnet34
    elif name == 'resnet50_fc':
        model_fn = resnet50_fc512
    else:
        raise ValueError('The given model name is not support. Supported models are resnet18, resnet34, resnet50, resnet50_fc')
    return model_fn(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )
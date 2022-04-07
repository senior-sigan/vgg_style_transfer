import pathlib
from collections import OrderedDict

import gradio as gr
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128
image_loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor(),
])


def image_to_tensor(image: PIL.Image.Image) -> torch.Tensor:
    image = image_loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def tensor_to_image(tensor: torch.Tensor) -> PIL.Image.Image:
    tensor = tensor.cpu().clone().squeeze(0)
    return TF.to_pil_image(tensor)


def gram_matrix(x):
    n, nfm, u, v = x.size()
    # n - Batch Size
    # nfm - number of feature maps
    # (u,v) - dimensions of a feature map (N=c*d)

    features = x.view(n * nfm, u * v)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(n * nfm * u * v)


def loss(target_features, combination_features):
    assert len(target_features) == len(combination_features)
    loss = 0
    for i in range(len(target_features)):
        loss += F.mse_loss(
            target_features[i],
            combination_features[i],
        )
    return loss


def rebuild_vgg(max_layer_idx):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    layers = OrderedDict()
    i = 0
    for layer in cnn.children():
        if i > max_layer_idx:
            break
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        layers[name] = layer

    return layers


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(FeatureExtractor, self).__init__()
        self.content_layers_names = set(['conv_4'])
        self.style_layers_names = set([
            'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'])
        self.layers = rebuild_vgg(5)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        x = self.normalize(x)
        style_features = []
        content_features = []
        for name, layer in self.layers.items():
            x = layer.forward(x)
            if name in self.content_layers_names:
                content_features.append(x)
            elif name in self.style_layers_names:
                g = gram_matrix(x)
                style_features.append(g)

        return style_features, content_features


def detach_all(tensors):
    return [tensor.detach() for tensor in tensors]


def style_transfer(
    feature_extractor,
    content_img: PIL.Image.Image,
    style_img: PIL.Image.Image,
    num_steps: int = 300,
    style_weight=1000000,
    content_weight=1,
) -> PIL.Image.Image:
    content_tensor = image_to_tensor(content_img)
    style_tensor = image_to_tensor(style_img)
    result_tensor = content_tensor.clone()

    result_tensor.requires_grad_(True)
    feature_extractor.requires_grad_(False)
    optimizer = optim.LBFGS([result_tensor], max_iter=20)

    _, content_cf = feature_extractor(content_tensor)
    style_sf, _ = feature_extractor(style_tensor)

    content_cf = detach_all(content_cf)
    style_sf = detach_all(style_sf)

    def closure():
        # correct the values of updated input image
        with torch.no_grad():
            result_tensor.clamp_(0, 1)

        optimizer.zero_grad()
        result_sf, result_cf = feature_extractor(result_tensor)
        content_score = loss(result_cf, content_cf) * content_weight
        style_score = loss(result_sf, style_sf) * style_weight

        total_loss = style_score + content_score
        total_loss.backward()

        return total_loss

    for _ in tqdm(range(num_steps)):
        optimizer.step(closure)

    with torch.no_grad():
        result_tensor.clamp_(0, 1)

    return tensor_to_image(result_tensor)


def load_examples():
    return [[
        pathlib.Path('examples/dancing.jpeg').as_posix(),
        pathlib.Path('examples/picasso.jpeg').as_posix(),
        10,
        1000000,
        1,
    ]]


def main():
    print(f'Using {device}')
    gr.close_all()

    feature_extractor = FeatureExtractor().eval()

    def fn(*args):
        return style_transfer(feature_extractor, *args)

    iface = gr.Interface(
        fn=fn,
        inputs=[
            gr.inputs.Image(
                type='pil',
                label='Content Image',
            ),
            gr.inputs.Image(
                type='pil',
                label='Style Image',
            ),
            gr.inputs.Slider(
                minimum=1,
                maximum=300,
                step=1,
                default=10,
                label='Number of iterations',
            ),
            gr.inputs.Number(
                default=1000000,
                label='Style weight',
            ),
            gr.inputs.Number(
                default=1,
                label='Content weight',
            ),
        ],
        outputs=gr.outputs.Image(label='Combined image'),
        examples=load_examples(),
        title='VGG style transfer',
        allow_flagging='never',
        theme='huggingface',
        article='Code is based on [pytorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) and [keras tutorial](https://keras.io/examples/generative/neural_style_transfer/).'
    )

    iface.launch(
        enable_queue=True,
    )


if __name__ == '__main__':
    main()

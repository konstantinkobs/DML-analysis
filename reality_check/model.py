import sys
sys.path.append("..")
import pretrainedmodels # needs to be installed with pip
from pytorch_metric_learning.utils import common_functions as c_f
import torch
from utils import get_model_path
import torch.nn as nn

# Taken from powerful_benchmarker
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class Network(torch.nn.Module):
    def __init__(self, loss: str = "Contrastive", dataset: str = "Cars196", fold: int = 0) -> None:
        super().__init__()

        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trunk = pretrainedmodels.bninception()
        self.trunk.last_linear = c_f.Identity()
        self.embedder = MLP([1024, 128])

        # you can pass "None" to loss so this will only yield a freshly initialized model without pretrained weights
        if loss != "None":
            trunk_path, embedder_path = get_model_path(loss, dataset, fold)
            self.trunk.load_state_dict(torch.load(trunk_path, map_location=self.device))
            self.embedder.load_state_dict(torch.load(embedder_path, map_location=self.device))

    def forward(self, x):
        x = self.trunk(x)
        return self.embedder(x)

import torch.nn as nn
import torch


class Network(nn.Module):

    def __init__(self, seed: int,state_size=64, value_size=2):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_hidden = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
        )

        self.output = nn.Linear(64, value_size)

    def forward(self, state):
        """Build a network that maps state ->  values."""
        x = self.input_hidden(state)

        return self.output(x)


if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    net = Network(2, 4, 0).to(device)

    x = torch.tensor([1, 1]).float().unsqueeze(0).to(device)
    #
    # torch.nn.DataParallel(net, device_ids=[0])
    print(net(x))

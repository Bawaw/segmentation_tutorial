import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                                 #############
                                 # LOAD DATA #
                                 #############
batch_size = 32
class SegmentationDataset(MNIST):
    def __init__(self, **kwargs):
        super(SegmentationDataset, self).__init__(**kwargs)
        self.targets = SegmentationDataset.compute_masks(self.data)
        self.targets = self.prep_data(self.targets)
        self.data = self.prep_data(self.data)/255

    def compute_masks(data):
        return (data > 0.1).float()

    def prep_data(self, x, target_size = 32):
        pad_shape = [int((target_size - x.shape[-1])/2)]*4
        return F.pad(x, pad_shape, mode='constant').float()

    def __getitem__(self, index):
        img, target = self.data[index, None], self.targets[index, None]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class GaussianNoise(object):
    def __init__(self, mean=0., std=0.3):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0, 1)

tf = torchvision.transforms.Compose([
    GaussianNoise(0.2)
])

train_loader = torch.utils.data.DataLoader(
    SegmentationDataset(root = './tr', train = True, download = True, transform = tf),
    batch_size = batch_size
)


test_loader = torch.utils.data.DataLoader(
    SegmentationDataset(root = './tr', train = False, download = True, transform = tf)
)

                             #####################
                             # Model & Optimiser #
                             #####################

net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                     in_channels=1, out_channels=1, init_features=32, pretrained = False)
net.to(device)
#net.load_state_dict(torch.load('unet.pt', map_location = device))
print(net)

optimiser = optim.Adam(net.parameters(), lr = 1e-3)

                               #################
                               # Training Loop #
                               #################
net.train()
for epoch in range(10):
    running_loss = 0.

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()

        y_star = net(x)
        loss = F.binary_cross_entropy(y_star, y)
        running_loss += loss.item()

        loss.backward()
        optimiser.step()

    print('Epoch: {}. loss: {:.3f}'.format(epoch, running_loss))
    torch.save(net.state_dict(), 'unet.pt')

                                   ########
                                   # Test #
                                   ########

net.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        y_star = net(x)
        _x, _y, _y_star = x[0,0], y[0,0], y_star[0,0]
        view = torch.cat([_x, _y, _y_star], dim = 1)
        plt.imshow(view.cpu().numpy())
        plt.show()

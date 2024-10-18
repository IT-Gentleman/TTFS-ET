# Reference Model for MNIST dataset
# Stricted to use MNIST dataset
import torch
import torch.nn as nn
import snntorch as snn

class defaultTTFS(nn.Module):
    def __init__(self, beta=0.95):
        super(defaultTTFS, self).__init__()

        # First Conv Layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)

        # Second Conv Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 4 * 4, 800)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc2 = nn.Linear(800, 128)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc3 = nn.Linear(128, 10)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()
        self.lif5.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        
        for step in range(num_steps):
            x_t = x[step]

            cur1 = self.conv1(x_t)
            spk1= self.lif1(cur1)
            spk1 = self.pool1(spk1)

            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool2(spk2)

            spk2_flat = spk2.view(spk2.size(0), -1)

            cur3 = self.fc1(spk2_flat)
            spk3 = self.lif3(cur3)

            cur4 = self.fc2(spk3)
            spk4 = self.lif4(cur4)
            
            cur5 = self.fc3(spk4)
            spk5 = self.lif5(cur5)

            spk_out.append(spk5)
            
        return torch.stack(spk_out)
    
class earlyTTFS(nn.Module):
    def __init__(self, beta=0.95):
        super(earlyTTFS, self).__init__()

        # First Conv Layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)

        # Second Conv Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 4 * 4, 800)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc2 = nn.Linear(800, 128)
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc3 = nn.Linear(128, 10)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()
        self.lif4.init_leaky()
        self.lif5.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            #x_t = x[step].clone()
            x_t = x[step]

            cur1 = self.conv1(x_t)
            spk1= self.lif1(cur1)
            spk1 = self.pool1(spk1)

            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)
            spk2 = self.pool2(spk2)

            spk2_flat = spk2.view(spk2.size(0), -1)

            cur3 = self.fc1(spk2_flat)
            spk3 = self.lif3(cur3)

            cur4 = self.fc2(spk3)
            spk4 = self.lif4(cur4)
            
            cur5 = self.fc3(spk4)
            spk5 = self.lif5(cur5)

            spk_out.append(spk5)

            newly_spiked = (spk5.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            if not not_spiked.any():
                break
            
        return torch.stack(spk_out)
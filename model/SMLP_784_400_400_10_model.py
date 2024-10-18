# Reference Model for MNIST dataset
# Stricted to use MNIST dataset
import torch
import torch.nn as nn
import snntorch as snn

class defaultTTFS(nn.Module):
    def __init__(self, beta=0.95):
        super(defaultTTFS, self).__init__()

        # Fully Connected Layer
        self.fc1 = nn.Linear(784, 400)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc2 = nn.Linear(400, 400)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc3 = nn.Linear(400, 10)
        self.lif3= snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        
        for step in range(num_steps):
            x_t = x[step]

            x_t = x_t.view(x_t.size(0), -1)
            cur1 = self.fc1(x_t)
            spk1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

            spk_out.append(spk3)
            
        return torch.stack(spk_out)
    
class earlyTTFS(nn.Module):
    def __init__(self, beta=0.95):
        super(earlyTTFS, self).__init__()

        # Fully Connected Layer
        self.fc1 = nn.Linear(784, 400)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc2 = nn.Linear(400, 400)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)

        self.fc3 = nn.Linear(400, 10)
        self.lif3= snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()

        num_steps = x.shape[0]
        spk_out = []
        batch_size = x.size(1)
        not_spiked = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for step in range(num_steps):
            #x_t = x[step].clone()
            x_t = x[step]

            x_t = x_t.view(x_t.size(0), -1)
            cur1 = self.fc1(x_t)
            spk1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

            spk_out.append(spk3)

            newly_spiked = (spk3.sum(dim=1)>0) & not_spiked
            not_spiked[newly_spiked] = False
            if not not_spiked.any():
                break
            
        return torch.stack(spk_out)
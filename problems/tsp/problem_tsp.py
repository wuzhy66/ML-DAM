from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        size = dataset.size(1)
        bat = dataset.size(0)
        # f_size = int(dataset.size(2)/2)
        f_size = int(2)
        # dis = dataset.clone()
        dis = dataset[:, :, 0:f_size]
        dis = (dis[:, :, None, :] - dis[:, None, :, :]).norm(p=2, dim=-1)
        time = dataset[:, :, f_size:]
        time = (time[:, :, None, :] - time[:, None, :, :]).norm(p=2, dim=-1)
        f1 = torch.zeros(dataset.size(0))
        f2 = torch.zeros(dataset.size(0))
        f1 = f1.to(dataset.device)
        f2 = f2.to(dataset.device)


        for j in range(size - 1):
            f1 = f1 + dis[torch.arange(bat), pi[:, j], pi[:, j + 1]]
            f2 = f2 + time[torch.arange(bat), pi[:, j], pi[:, j + 1]]
        f1 = f1 + dis[torch.arange(bat), pi[:, 0], pi[:, size - 1]]
        f2 = f2 + time[torch.arange(bat), pi[:, 0], pi[:, size - 1]]
        return [f1, f2], None


        # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        # return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 4).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

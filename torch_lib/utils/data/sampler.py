from torch.utils.data.sampler import Sampler


class BinarySampler(Sampler):
    def __init__(self, labels, num_samples, replace=False):
        self.labels = labels
        self.num_samples = num_samples
        self.replace = replace
        
    def __len__(self):
        return self.num_samples*2
    
    def __iter__(self):
        n = len(self.labels)
        pos_index = np.where(self.labels==1)[0]
        neg_index = np.where(self.labels==0)[0]
        pos = np.random.choice(pos_index, self.num_samples, replace=self.replace)
        neg = np.random.choice(neg_index, self.num_samples, replace=self.replace)
        indices = np.stack([pos, neg]).T
        indices = indices.reshape(-1)
        return iter(indices)

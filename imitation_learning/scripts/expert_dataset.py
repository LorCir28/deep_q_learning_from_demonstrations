from torch.utils.data.dataset import Dataset
from torchvision import transforms

class ExpertDataSet(Dataset):

    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        self.img_transforms=transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        return (self.img_transforms(self.observations[index]), self.actions[index])

    def __len__(self):
        return len(self.observations)
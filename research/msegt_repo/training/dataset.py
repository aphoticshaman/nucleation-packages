from typing import List, Tuple
from torch.utils.data import Dataset

class PRMDataset(Dataset):
    """
    Holds (problem, step, label) triples for PRM training.
    """
    def __init__(self, problems: List[str], steps: List[str], labels: List[int]):
        assert len(problems) == len(steps) == len(labels)
        self.problems = problems
        self.steps = steps
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return self.problems[idx], self.steps[idx], self.labels[idx]

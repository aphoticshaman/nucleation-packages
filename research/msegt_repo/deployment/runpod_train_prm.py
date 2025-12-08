import os
import torch
from msegt.training.dataset import PRMDataset
from msegt.training.prm_training import train_prm

def load_dummy_dataset() -> PRMDataset:
    problems = ["dummy problem 1", "dummy problem 2"]
    steps = ["dummy step 1", "dummy step 2"]
    labels = [1, 0]
    return PRMDataset(problems, steps, labels)

def main():
    model_name = os.environ.get("EAGLE_MODEL_NAME", "nvidia/gpt-oss-120b-eagle3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dummy_dataset()
    result = train_prm(
        dataset=dataset,
        model_name_or_path=model_name,
        device=device,
        batch_size=2,
        lr=1e-4,
        num_epochs=1,
        hidden_dim=4096,
    )
    prm_head = result["prm_head"]
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(prm_head.state_dict(), "checkpoints/prm_head.pt")
    print("Saved PRM head to checkpoints/prm_head.pt")

if __name__ == "__main__":
    main()

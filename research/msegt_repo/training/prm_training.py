from typing import Dict, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from msegt.models.eagle_adapter import EagleWrapper
from msegt.models.prm_head import PRMHead

def train_prm(
    dataset,
    model_name_or_path: str,
    device: str = "cuda",
    batch_size: int = 16,
    lr: float = 1e-4,
    num_epochs: int = 3,
    hidden_dim: int = 4096,
    grad_accum_steps: int = 2,
) -> Dict[str, Any]:
    wrapper = EagleWrapper(model_name_or_path=model_name_or_path, device=device)
    prm_head = PRMHead(hidden_dim=hidden_dim).to(device)

    for p in wrapper.model.parameters():
        p.requires_grad = False

    optimizer = AdamW(prm_head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    prm_head.train()
    step = 0

    for epoch in range(num_epochs):
        total = 0.0
        count = 0
        for problems, steps, labels in tqdm(loader, desc=f"PRM epoch {epoch+1}/{num_epochs}"):
            texts = [p + " [STEP] " + s for p, s in zip(problems, steps)]
            embeddings = wrapper.forward_for_prm(texts)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            logits = prm_head(embeddings.to(device))
            loss = criterion(logits, labels_t)

            loss.backward()
            step += 1
            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total += loss.item() * len(labels)
            count += len(labels)

        print(f"Epoch {epoch+1}: loss={total / max(count,1):.4f}")

    return {"wrapper": wrapper, "prm_head": prm_head}

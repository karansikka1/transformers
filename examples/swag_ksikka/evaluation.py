import torch
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, val_loader, device):
    scores = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, label, attention_masks in tqdm(val_loader, desc="Validation"):
            data = data.to(device)
            attention_masks = attention_masks.to(device)
            output = model(input_ids=data, attention_mask=attention_masks)
            logits = output[0]
            scores.append(logits.cpu().numpy())
            labels.append(label.cpu().numpy())
    scores = np.vstack(scores)
    labels = np.hstack(labels)
    pred = scores.argmax(-1)
    acc = (pred == labels).mean() * 100
    return acc, scores, labels

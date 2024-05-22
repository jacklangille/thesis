import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

def evaluate(model, loader, device):
    model.eval()
    model.to(device)
    true_labels = []
    predicted_labels = []
    softmax_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
            softmax_outputs.append(softmax_probs.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "softmax_outputs": np.vstack(softmax_outputs),
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }

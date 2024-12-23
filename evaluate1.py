import torch
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from network import get_network
from data import get_dataloaders

class Metric:
    """Performance metric with class-wise accuracy."""

    def __init__(self, num_classes):
        """Set label."""
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset."""
        self.predictions = []
        self.ground_truths = []

    def update_prediction(self, y_pred, y):
        """Update prediction and gt."""
        self.ground_truths += y.tolist()

        with torch.no_grad():
            y_prob = softmax(y_pred, dim=1)
        self.predictions += y_prob.tolist()

        return None

    def calc_accuracy(self):
        """Get overall and class-wise accuracy."""
        y_pred = [prob.index(max(prob)) for prob in self.predictions]
        
        # Overall accuracy
        overall_acc = accuracy_score(y_true=self.ground_truths, y_pred=y_pred)
        
        # Class-wise accuracy
        conf_matrix = confusion_matrix(y_true=self.ground_truths, y_pred=y_pred, 
                                     labels=range(self.num_classes))
        class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        return overall_acc, class_acc

def evaluate_step(model, dataloader, device, num_classes):
    """Evaluate a network."""
    metric = Metric(num_classes)

    for x, y, _ in dataloader:
        with torch.no_grad():
            y_hat = model(x[0].to(device))
        metric.update_prediction(y_hat, y.to(device))
    
    overall_acc, class_acc = metric.calc_accuracy()

    return overall_acc, class_acc

def evaluate_network(args):
    """Evaluate a network."""
    device = torch.device('cuda')

    model = get_network(args.network, args.num_classes)
    ckpt = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(ckpt['ema'])
    model.eval()
    model.to(device)

    # labeled, unlabeled and test data
    _, _, T = get_dataloaders(data=args.data,
                           num_X=args.num_X,
                           include_x_in_u=args.include_x_in_u,
                           augs=args.augs,
                           batch_size=args.batch_size,
                           mu=args.mu)

    overall_acc, class_acc = evaluate_step(model, T, device, args.num_classes)

    print(f"Overall Accuracy: {overall_acc:1.4f}")
    print("\nClass-wise Accuracy:")
    for i, acc in enumerate(class_acc):
        print(f"Class {i}: {acc:1.4f}")
    
    # Optionally, return the accuracies
    return overall_acc, class_acc
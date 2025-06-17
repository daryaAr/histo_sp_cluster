import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

def cross_entropy_loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels)


def get_accuracy_metric(task="multiclass", num_classes=3, average="macro"):
    return Accuracy(task=task, num_classes=num_classes, average=average)


def get_f1_metric(task="multiclass", num_classes=3, average="macro"):
    return F1Score(task=task, num_classes=num_classes, average=average)


def get_confusion_matrix_metric(task="multiclass", num_classes=3):
    return ConfusionMatrix(task=task, num_classes=num_classes)
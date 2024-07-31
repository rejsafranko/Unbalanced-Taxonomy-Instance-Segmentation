import numpy as np


class DynamicLossWeighting:
    def __init__(self, class_counts, epsilon=1e-8):
        self.class_counts = class_counts
        self.total_samples = sum(class_counts.values())
        self.classes = list(class_counts.keys())
        self.epsilon = epsilon
        self.class_weights = self.initialize_class_weights()

    def initialize_class_weights(self):
        return {c: self.total_samples / count for c, count in self.class_counts.items()}

    def calculate_recall(self, predictions, labels):
        recall = {}
        for c in self.classes:
            true_positive = np.sum((predictions == c) & (labels == c))
            actual_positive = np.sum(labels == c)
            recall[c] = true_positive / actual_positive if actual_positive > 0 else 0
        return recall

    def adjust_weights(self, recall):
        for c in recall:
            self.class_weights[c] = 1 / (recall[c] + self.epsilon)

    def weighted_cross_entropy(self, predictions, labels):
        loss = 0
        for i in range(len(labels)):
            loss += self.class_weights[labels[i]] * -np.log(predictions[i, labels[i]])
        return loss / len(labels)

    def update_weights(self, predictions, labels):
        recall = self.calculate_recall(predictions, labels)
        self.adjust_weights(recall)

    def compute_loss(self, predictions, labels):
        return self.weighted_cross_entropy(predictions, labels)

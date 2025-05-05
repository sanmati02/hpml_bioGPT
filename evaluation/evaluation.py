"""
evaluate_predictions.py

Compares predictions to ground truth for PubMedQA.
Outputs Accuracy and Macro-F1 score.
"""

import json
from sklearn.metrics import accuracy_score, f1_score
import sys

# Parse prediction file path from command-line arguments
pred_path = sys.argv[1]

# Load ground truth and predicted results
ground_truth = json.load(open('test_ground_truth.json')) 
predictions = json.load(open(pred_path))

# Check that the prediction set matches the ground truth set
assert set(list(ground_truth)) == set(list(predictions)), 'Please predict all and only the instances in the test set.'

# Prepare true and predicted label lists in consistent order
pmids = list(ground_truth)
truth = [ground_truth[pmid] for pmid in pmids]
preds = [predictions[pmid] for pmid in pmids]

# Evaluate model performance
acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')

# Print evaluation results
print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)

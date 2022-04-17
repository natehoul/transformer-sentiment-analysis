# Train the model!


from pathlib import Path
import sys


import datetime


from sklearn.metrics import confusion_matrix

import torch

from transformers import get_linear_schedule_with_warmup


from models.bert_classifier import BertClassifier
from datasets.dataloaders import get_dataloaders_end_to_end as get_dataloaders
import results
import saved_models

### HYPERPARAMETERS ###

# Data (not exhaustive, see the params for get_dataloaders)
DATASET = 'music'
NUM_BERT_TOKENS = 64

# Model (not exhaustive, see the params for BertClassifier)
NUM_CLASSES = 2
DROPOUT = 0.1

# Optimizer
MOMENTUM = 0.9 # Currently unused becuse we're using Adam
LEARNING_RATE = 0.02
REGULARIZATION_WEIGHT = 0
EPSILON = 1e-8

# Misc
BATCH_SIZE = 32
NUM_EPOCHS = 2

### END HYPERPARAMETERS ###


# Initialize all the stuff needed to train
# If the input "model_to_load" contains the name of an existing saved model,
# that model will be loaded rather than creating a new one
def initialize(model_to_load=''):
    print("INITIALIZING TRAINING PROCESS")
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"USING DEVICE {device}")

    if model_to_load and saved_models.exists(model_to_load):
        model = saved_models.load(model_to_load)
    else:
        model = BertClassifier(num_classes=NUM_CLASSES, dropout=DROPOUT)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()# if NUM_CLASSES == 2 else torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=LEARNING_RATE, 
                                 eps=EPSILON, 
                                 weight_decay=REGULARIZATION_WEIGHT)
    
    train_dataloader, val_dataloader = get_dataloaders(dataset_name=DATASET, 
                                                       binarize=NUM_CLASSES == 2,
                                                       batch_size=BATCH_SIZE,
                                                       token_len=NUM_BERT_TOKENS)
    
    num_steps = len(train_dataloader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)


    performance_metrics = [
        "Training Loss",
        "Training Accuracy",
        "Training Precision",
        "Training Recall",
        "Training F1",
        "Validation Loss",
        "Validation Accuracy",
        "Validation Precision",
        "Validation Recall",
        "Validation F1"
        ]

    performance_metrics = {metric:[] for metric in performance_metrics}

    return train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device


# Train a single epoch
def train_epoch(train_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device):
    model.train()

    losses = []

    if NUM_CLASSES == 2:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

    else:
        correct = 0


    for inputs, masks, labels in train_dataloader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        model.zero_grad()

        print(inputs.size())

        logits = model(inputs, masks)
        loss = criterion(torch.argmax(logits, dim=1).type(torch.FloatTensor), labels.type(torch.LongTensor))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip to prevent exploding gradients
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        prediction = torch.argmax(logits, dim=1) # may need to add ".flatten()"
        ground_truth = labels # may need to process this somehow. Cast to int?

        # Binary classifer, so we can get confusion matrix easily
        if NUM_CLASSES == 2:
            matrix = confusion_matrix(ground_truth, prediction)
            tn += matrix[0][0]
            fn += matrix[1][0]
            tp += matrix[1][1]
            fp += matrix[0][1]

        # More than 2 classes, so we'll just settle for accuracy only
        else:
            good_predictions = (prediction == ground_truth)
            correct += good_predictions.sum().item()

    
    loss = sum(losses) / len(losses)
    performance_metrics['Training Loss'].append(loss)

    if NUM_CLASSES == 2:
        accuracy = (tp + fp) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        performance_metrics['Training Precision'].append(precision)
        performance_metrics['Training Recall'].append(recall)
        performance_metrics['Training F1'].append(f1)

    else:
        accuracy = correct / len(train_dataloader.dataset)

    performance_metrics['Training Accuracy'].append(accuracy)




# Validate a single epoch
def validate_epoch(val_dataloader, model, criterion, performance_metrics, device):
    model.eval()

    losses = []

    if NUM_CLASSES == 2:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

    else:
        correct = 0


    for inputs, masks, labels in val_dataloader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            logits = model(inputs, masks)
        
        loss = criterion(logits, labels)
        losses.append(loss.item())

        prediction = torch.argmax(logits, dim=1) # may need to add ".flatten()"
        ground_truth = labels # may need to process this somehow. Cast to int?

        # Binary classifer, so we can get confusion matrix easily
        if NUM_CLASSES == 2:
            matrix = confusion_matrix(ground_truth, prediction)
            tn += matrix[0][0]
            fn += matrix[1][0]
            tp += matrix[1][1]
            fp += matrix[0][1]

        # More than 2 classes, so we'll just settle for accuracy only
        else:
            good_predictions = (prediction == ground_truth)
            correct += good_predictions.sum().item()

    
    loss = sum(losses) / len(losses)
    performance_metrics['Validation Loss'].append(loss)

    if NUM_CLASSES == 2:
        accuracy = (tp + fp) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        performance_metrics['Validation Precision'].append(precision)
        performance_metrics['Validation Recall'].append(recall)
        performance_metrics['Validation F1'].append(f1)

    else:
        accuracy = correct / len(val_dataloader.dataset)

    performance_metrics['Validation Accuracy'].append(accuracy)



# Train the model; full process
def train(session_name, model_to_load=''):
    train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device = initialize(model_to_load)

    print("BEGINNING TRAINING")
    for epoch in range(NUM_EPOCHS):
        train_epoch(train_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device)
        validate_epoch(val_dataloader, model, criterion, performance_metrics, device)

        # Save results as we go in case something breaks
        results.save(performance_metrics, session_name)
        results.create_pyplot(performance_metrics, 'all', session_name)
        saved_models.save(model, session_name)

        print(f'Epoch: {epoch:>2}', end='\t')
        for metric, value in performance_metrics.items():
            print(f'{metric}: {value:.4f}', end='\t')
        print()

    print("TRAINING COMPLETE")




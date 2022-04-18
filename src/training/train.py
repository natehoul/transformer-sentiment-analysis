# Train the model!

from sklearn.metrics import confusion_matrix

import torch

from transformers import get_linear_schedule_with_warmup


from models.bert_classifier import BertClassifier
from datasets.dataloaders import get_dataloaders_end_to_end as get_dataloaders
import results
import saved_models

### HYPERPARAMETERS ###

default_hyperparameters = {
    # Data (not exhaustive, see the params for get_dataloaders)
    "DATASET": 'music',
    "DATA_TYPE": 'rating',
    "NUM_BERT_TOKENS": 64,
    "BATCH_SIZE": 32,
    
    # Model (not exhaustive, see the params for BertClassifier)
    "NUM_CLASSES": 2,
    "DROPOUT": 0.1,

    # Optimizer
    "MOMENTUM": 0.9, # Currently unused becuse we're using Adam
    "LEARNING_RATE": 0.02,
    "REGULARIZATION_WEIGHT": 0,
    "EPSILON": 1e-8,

    # Misc
    "NUM_EPOCHS": 2 # Increase this once everything actually works
}

### END HYPERPARAMETERS ###


# Initialize all the stuff needed to train
# If the input "model_to_load" contains the name of an existing saved model,
# that model will be loaded rather than creating a new one
def initialize(model_to_load='', hyperparameters=default_hyperparameters):
    print("INITIALIZING TRAINING PROCESS")
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"USING DEVICE {device}")

    if model_to_load and saved_models.exists(model_to_load):
        model = saved_models.load(model_to_load)
    else:
        model = BertClassifier(num_classes=hyperparameters['NUM_CLASSES'], dropout=hyperparameters['DROPOUT'])

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()# if hyperparameters['NUM_CLASSES'] == 2 else torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=hyperparameters['LEARNING_RATE'], 
                                 eps=hyperparameters['EPSILON'], 
                                 weight_decay=hyperparameters['REGULARIZATION_WEIGHT'])
    
    train_dataloader, val_dataloader = get_dataloaders(dataset_name=hyperparameters['DATASET'],
                                                       data_type=hyperparameters['DATA_TYPE'], 
                                                       binarize=hyperparameters['NUM_CLASSES'] == 2,
                                                       batch_size=hyperparameters['BATCH_SIZE'],
                                                       token_len=hyperparameters['NUM_BERT_TOKENS'])
    
    num_steps = len(train_dataloader) * hyperparameters['NUM_EPOCHS']

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
def train_epoch(train_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device, hyperparameters):
    model.train()

    losses = []

    if hyperparameters['NUM_CLASSES'] == 2:
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

        logits = model(inputs, masks)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip to prevent exploding gradients
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        prediction = torch.argmax(logits, dim=1) # may need to add ".flatten()"
        ground_truth = labels # may need to process this somehow. Cast to int?

        # Binary classifer, so we can get confusion matrix easily
        if hyperparameters['NUM_CLASSES'] == 2:
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

    if hyperparameters['NUM_CLASSES'] == 2:
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
def validate_epoch(val_dataloader, model, criterion, performance_metrics, device, hyperparameters):
    model.eval()

    losses = []

    if hyperparameters['NUM_CLASSES'] == 2:
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
        if hyperparameters['NUM_CLASSES'] == 2:
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

    if hyperparameters['NUM_CLASSES'] == 2:
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



# 'hp' = 'hyperparmeters' = ACTIVE hyperparameters, default or otherwise
# 'sp' = 'special parameters' = NON-DEFAULT hyperparameters, specifically
def get_descriptive_session_name(hp, sp):
    data_info = f'{hp["DATASET"]}_{hp["DATA_TYPE"]}'
    other_info = []

    if "NUM_CLASSES" in sp:
        other_info.append(f'{sp["NUM_CLASSES"]}classes')

    if "BATCH_SIZE" in sp:
        other_info.append(f'batch{sp}')

    if "LEARNING_RATE" in sp:
        other_info.append(f'lr{str(sp["LEARNING_RATE"])[2:]}')
    
    if "NUM_BERT_TOKENS" in sp:
        other_info.append(f'{sp["NUM_BERT_TOKENS"]}tokens')

    if "DROPOUT" in sp:
        other_info.append(f'dropout{str(sp["DROPOUT"])[2:]}')

    if "MOMENTUM" in sp:
        other_info.append(f'momentum{str(sp["MOMENTUM"])[2:]}')

    if "REGULARIZATION_WEIGHT" in sp:
        weight = sp["REGULARIZATION_WEIGHT"]
        if isinstance(weight, int) or weight == int(weight):
            s = str(int(weight))
        else:
            s = str(weight).replace('.', ',')

        other_info.append(f'regweight{s}')

    if len(other_info) > 0:
        name = f'{data_info}_{"_".join(other_info)}'

    else:
        name = data_info

    return name



# Train the model; full process
def train(session_name='', model_to_load='', **kwargs):
    hyperparameters = {}
    for k in default_hyperparameters:
        if k in kwargs:
            hyperparameters[k] = kwargs[k]
        else:
            hyperparameters[k] = default_hyperparameters[k]

    if session_name == '':
        session_name = get_descriptive_session_name(hyperparameters, kwargs)


    train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device = initialize(model_to_load, hyperparameters)

    print("BEGINNING TRAINING")
    for epoch in range(hyperparameters['NUM_CLASSES']):
        train_epoch(train_dataloader, model, criterion, optimizer, scheduler, performance_metrics, device, hyperparameters)
        validate_epoch(val_dataloader, model, criterion, performance_metrics, device, hyperparameters)

        # Save results as we go in case something breaks
        results.save(performance_metrics, session_name)
        results.create_pyplot(performance_metrics, 'all', session_name)
        saved_models.save(model, session_name)

        print(f'Epoch: {epoch:>2}', end='\t')
        for metric, value in performance_metrics.items():
            print(f'{metric}: {value:.4f}', end='\t')
        print()

    print("TRAINING COMPLETE")




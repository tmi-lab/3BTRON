import torch
import torch.nn as nn 
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

criterion = nn.CrossEntropyLoss(reduction='none') 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        print(f'Checking early stopping: Current Loss: {validation_loss}, Min Loss: {self.min_validation_loss}, Counter: {self.counter}')
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(f'Updated min loss to {self.min_validation_loss}, reset counter.')
            return False
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            print(f'Counter incremented to {self.counter}.')
            if self.counter >= self.patience:
                print('Triggering early stop.')
                return True
        return False

def train_val_model(model, name, train_loader, val_loader, n_epochs, optimizer, device,
              patience=1, min_delta=0.001, scheduler=None, save_model=False, save_model_end=False):

    
    history = {'train_loss':[],
               'val_loss':[], 'val_acc':[], 'val_f1_score':[], 'val_roc_auc_score':[], 'val_pr_auc_score': [],
               'val_sensitivity':[], 'val_specificity':[], 'val_precision':[], 
               'val_true_labels': [], 'val_probabilities':[], 'val_pos_probabilities': []}
    
    torch.manual_seed(0)

    valid_loss_min = np.Inf

    model = model.to(device)

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_examples = 0
        pbar = tqdm.tqdm(total=len(train_loader), desc='Training Epoch {}'.format(epoch+1))
        
        for batch_idx, (data, targets, sample_weights) in enumerate(train_loader):
            targets = targets.long() 
            data, targets, sample_weights = data.to(device), targets.to(device), sample_weights.to(device) # Copies inputs to device
            
            optimizer.zero_grad() # Resets the optimizer
            
            outputs = model(data)
            
            loss = criterion(outputs, targets)
            loss = (loss * sample_weights).mean()  # Weights the loss
            
            loss.backward() 
            optimizer.step()

            # Record the loss and number of examples and set the progress bar information
            total_loss += loss.item()*len(data)
            total_examples += len(data)
            pbar.set_postfix({'Avg Loss': total_loss/total_examples})
            pbar.update(1)
            pbar.refresh()

        # Record training loss for each epoch
        history['train_loss'].append(total_loss/total_examples)

        model.eval()
        epoch_probabilities = []
        total_loss_val = 0
        total_examples_val = 0
        total_labels = 0
        correct_examples = 0
        confusion_matrix_counts = np.zeros((2,2))
        
        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(val_loader), desc='Validation Epoch {}'.format(epoch+1))
            for batch_idx, (data, targets, sample_weights) in enumerate(val_loader):
                targets = targets.long() 
                data, targets = data.to(device), targets.to(device) # Copies inputs to device

                outputs = model(data)
                
                soft_outputs = torch.nn.functional.softmax(outputs, dim=1) # Applies softmax function to the logits to get probability estimates
                epoch_probabilities.append(soft_outputs.detach().cpu().numpy())
                
                loss = criterion(outputs, targets).mean() # Average loss for validation (computed without sample weights)
                _, preds = torch.max(outputs, 1) # Gets the predictions based on the logits
                correct = preds.eq(targets).sum() # Calculates the sum of true positives + true negatives

                # Record the loss and number of examples and set the progress bar information
                total_loss_val += loss.item()*len(data)
                total_examples_val += len(data)
                total_labels += targets.shape[0]
                correct_examples += correct.item()
                for i in range(len(targets)):
                    confusion_matrix_counts[targets[i].item()][preds[i].item()] += 1

                pbar.set_postfix(
                    {'Avg Loss': total_loss_val/total_examples_val, 'Acc': correct_examples/total_labels}
                )
                pbar.update(1)
                pbar.refresh()

        # Record validation loss for each epoch
        history['val_loss'].append(total_loss_val/total_examples_val)
        history['val_acc'].append(correct_examples/total_labels)

        tn, fp, fn, tp = confusion_matrix_counts.ravel()
        sensitivity = tp/(tp + fn) if (tp + fn) > 0 else 0
        specificity = tn/(tn + fp) if (tn + fp) > 0 else 0
        precision = tp/(tp + fp) if (tp + fp) > 0 else 0
        f1_score = (2 * precision * sensitivity)/(precision + sensitivity) if (precision + sensitivity) > 0 else 0

        true_labels = np.concatenate([targets.cpu().numpy() for _, targets, _ in val_loader])
        pos_probabilities = np.concatenate(epoch_probabilities)[:, 1]
        roc_auc = roc_auc_score(true_labels, pos_probabilities)

        pr_auc = average_precision_score(true_labels, pos_probabilities)

        history['val_f1_score'].append(f1_score)
        history['val_roc_auc_score'].append(roc_auc)
        history['val_pr_auc_score'].append(pr_auc)
        
        history['val_sensitivity'].append(sensitivity)
        history['val_specificity'].append(specificity)
        history['val_precision'].append(precision)

        history['val_true_labels'] = true_labels
        history['val_probabilities'].append(np.concatenate(epoch_probabilities))
        history['val_pos_probabilities'] = pos_probabilities

        # Early stopping check
        print(f'Epoch {epoch + 1}, Validation Loss: {history["val_loss"][-1]}, Min Validation Loss: {early_stopper.min_validation_loss}, Counter: {early_stopper.counter}')
        if early_stopper.early_stop(history['val_loss'][-1]):
            print(f'Early stopping at epoch {epoch + 1}')
            break

        # Saving the model if the validation loss has improved
        if save_model and (epoch == 0 or history['val_loss'][-1] < early_stopper.min_validation_loss):
            print('Saving model...')
            torch.save(model.state_dict(), '{}.pt'.format(name))

        # Learning rate scheduler step (if provided)
        if scheduler is not None:
            scheduler.step(history['val_loss'][-1])

    # Save the model at the end of training, if specified
    if save_model_end:
        print('Saving model at the end of training...')
        torch.save(model.state_dict(), f'{name}_final.pt')

    return history

def train_model_no_val(model, name, train_loader, n_epochs, optimizer, device,
                       patience=1, min_delta=0.001, scheduler=None, save_model=False, save_model_end=False,
                       max_epochs=None):
    history = {'train_loss': []}
    
    torch.manual_seed(0)
    model = model.to(device)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    
    for epoch in range(n_epochs):
        if max_epochs is not None and epoch >= max_epochs:
            print(f"Stopping training after {max_epochs} epochs.")
            break
        
        model.train()
        total_loss = 0
        total_examples = 0
        pbar = tqdm.tqdm(total=len(train_loader), desc=f'Training Epoch {epoch+1}')

        for batch_idx, (data, targets, sample_weights) in enumerate(train_loader):
            targets = targets.long()
            data, targets, sample_weights = data.to(device), targets.to(device), sample_weights.to(device) # Copies inputs to device

            optimizer.zero_grad() # Resets the optimizer
            
            outputs = model(data)
            
            loss = criterion(outputs, targets)
            loss = (loss * sample_weights).mean()  # Weights the loss
            
            loss.backward()
            optimizer.step()
            
            # Record the loss and number of examples and set the progress bar information
            total_loss += loss.item() * len(data)
            total_examples += len(data)
            pbar.set_postfix({'Avg Loss': total_loss / total_examples})
            pbar.update(1)

        # Record training loss for the epoch
        history['train_loss'].append(total_loss / total_examples)

        # Early stopping check (if you're stopping based on training loss)
        #if early_stopper.early_stop(history['train_loss'][-1]):
            #print(f'Early stopping at epoch {epoch + 1}')
            #break

        # Saving the model if the training loss has improved
        if save_model and (epoch == 0 or history['train_loss'][-1] < early_stopper.min_validation_loss):
            print('Saving model...')
            torch.save(model.state_dict(), '{}.pt'.format(name))

        # Learning rate scheduler step (if provided)
        if scheduler is not None:
            scheduler.step(history['train_loss'][-1])

    # Save the model at the end of training, if specified
    if save_model_end:
        print('Saving model at the end of training...')
        torch.save(model.state_dict(), f'{name}_final.pt')

    return history

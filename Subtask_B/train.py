import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

import os
import time
import datetime
import numpy as np

from model import CVEclassifier
from dataset import CVEdatasetB

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, device, epochs=6):

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode
        # `dropout` and `batchnorm` layers behave differently during training vs. test
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 10 batches.
            if step % 20 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


            # unpack the batch received from train_dataloader
            b_input_id1 = batch[0].to(device)
            b_input_id2 = batch[1].to(device)
            b_input_id3 = batch[2].to(device)
            b_input_mask1 = batch[3].to(device)
            b_input_mask2 = batch[4].to(device)
            b_input_mask3 = batch[5].to(device)
            b_labels = batch[6].to(device)

            model.zero_grad()        

            outputs = model(b_input_id1, b_input_id2, b_input_id3, b_input_mask1, b_input_mask2, b_input_mask3)

            loss = criterion(outputs, torch.argmax(b_labels, dim=1))

            # Accumulate the training loss over all of the batches to calculate average loss
            total_loss += loss.item()

            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()
        
        total_loss = 0

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in val_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            b_input_id1, b_input_id2, b_input_id3, b_input_mask1, b_input_mask2, b_input_mask3, b_labels, _ = batch
            
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        
                logits = model(b_input_id1, b_input_id2, b_input_id3, b_input_mask1, b_input_mask2, b_input_mask3)
            
            # Calculate loss for the batch
            loss = criterion(logits, torch.argmax(b_labels, dim=1))

            # Accumulate the training loss over all of the batches to calculate average loss
            total_loss += loss.item()
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = torch.argmax(b_labels, dim=1)
            label_ids = label_ids.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Calculate the average loss over the training data.
        eval_loss = total_loss / len(val_dataloader)
        
        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
        print("  Average validation loss: {0:.4f}".format(eval_loss))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    chkpt_dict = {'model_state_dist':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                     'scheduler_state_dict':scheduler.state_dict()}
    
    torch.save(chkpt_dict, '../weights/'+'stB-roberta-ep-'+str(epoch_i+1)+'-vacc-'+str((100*eval_accuracy/nb_eval_steps))+'.pt')
    
    print("")
    print("Training complete!")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='subtask-B')  
    parser.add_argument('--data-root', type=str, default='../Data/SemEval2020-Task4-Commonsense-Validation-and-Explanation/')                              
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--chkpt-path', type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('Using the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    trainset = CVEdatasetB(root=os.path.join(args.data_root, 'Training_Data'))
    valset = CVEdatasetB(root=os.path.join(args.data_root, 'Dev_Data'))
    
    # Training logs
    os.makedirs('drive/My Drive/sem_eval/logs/', exist_ok=True)
    # Weight checkpoint
    os.makedirs('drive/My Drive/sem_eval/weights/', exist_ok=True)
    
    #Creating intsances of training and validation dataloaders
    train_dataloader = DataLoader(trainset, batch_size = args.batch_size, num_workers = 5, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size = 128, num_workers = 5, shuffle=False)
    
    model =  CVEclassifier()
    
    criterion = nn.CrossEntropyLoss()
    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # args.learning_rate - default is 5e-5
    # args.adam_epsilon  - default is 1e-8
    optimizer = AdamW(model.parameters(), lr = args.lr, eps = args.eps)
    
    epochs = args.epochs
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(0.1*total_steps)
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)
    
    if args.chkpt_path:
        chkpt = torch.load(args.chkpt_path)
        model.load_state_dict(chkpt['model_state_dist'])
        optimizer.load_state_dict(chkpt['optimizer_state_dist'])
        scheduler.load_state_dict(chkpt['scheduler_state_dist'])
    
    model.to(device)
                                       
    train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, device, epochs)
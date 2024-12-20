# Code adapted from the official implementation of Erase-and-Check method:
# https://github.com/aounon/certified-llm-safety

import wandb
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import transformers
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, SequentialSampler

# Specify the available devices
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

# Function for reading the given file
def read_text(filename):	
  with open(filename, "r") as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
  return pd.DataFrame(lines)

# Set seed
seed = 912

# Parser for setting input values to train the safety classifier
parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data')
parser.add_argument('--classifier_name', type=str, default='distilbert', help='Name of the classifier', choices=["distilbert", "indicbert", "distilbert-multi"])
parser.add_argument('--save_path', type=str, default='models/distilbert.pt', help='Path to save the model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--wandb_log', action='store_true', help='Flag for logging results to wandb')
parser.add_argument('--wandb_project', type=str, default='llm-hindi-safety-filter', help='Name of the wandb project')
parser.add_argument('--wandb_entity', type=str, default='patchtst-flashattention', help='Name of the wandb entity')

args = parser.parse_args()

# Load safe and harmful prompts and create the dataset for training classifier
# Class 1: Safe, Class 0: Harmful
safe_train = f"{args.data_dir}/safe_prompts_train.txt"
safe_prompt_train = read_text(safe_train)
harm_train = f"{args.data_dir}/harmful_prompts_train.txt"
harm_prompt_train = read_text(harm_train)
prompt_data_train = pd.concat([safe_prompt_train, harm_prompt_train], ignore_index=True)
prompt_data_train['Y'] = pd.Series(np.concatenate([np.ones(safe_prompt_train.shape[0]), np.zeros(harm_prompt_train.shape[0])])).astype(int)


# Split train dataset into train and validation sets
train_text, val_text, train_labels, val_labels = train_test_split(prompt_data_train[0], 
								prompt_data_train['Y'], 
								random_state=seed, 
								test_size=0.2,
								stratify=prompt_data_train['Y'])

# Count number of samples in each class in the training set
count = train_labels.value_counts().to_dict()

# Load the tokenizer and model based on the classifier name
if args.classifier_name == "distilbert":
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

elif args.classifier_name == "distilbert-multi":
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')

elif args.classifier_name == "indicbert":
    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('ai4bharat/indic-bert', keep_accents=True)
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained('ai4bharat/indic-bert')

# Tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# Tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# Convert lists to tensors for train split
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())
sample_weights = torch.tensor([1/count[i] for i in train_labels])

# Convert lists to tensors for validation split
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# Define the batch size
batch_size = 32

# Wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# Sampler for sampling the data during training
# train_sampler = RandomSampler(train_data)
train_sampler = WeightedRandomSampler(sample_weights, len(train_data), replacement=True)

# DataLoader for the train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# Sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# DataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# Push the model to GPU
model = model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)          # learning rate

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Number of training epochs
epochs = args.epochs

# Initialize wandb logging
wandb_log = args.wandb_log
if wandb_log:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-5,
    })

# Function to train the model
def train():

  model.train()
  total_loss = 0
  
  # Empty list to save model predictions
  total_preds=[]
  
  # Iterate over batches
  for step, batch in enumerate(train_dataloader):
    
    # Progress update after every 50 batches.
    if (step + 1) % 50 == 0 or step == len(train_dataloader) - 1:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step + 1, len(train_dataloader)))

    # Push the batch to GPU
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # Clear previously calculated gradients 
    model.zero_grad()     

    # Get model predictions for the current batch
    preds = model(sent_id, mask)[0]

    # Compute the loss between actual and predicted values
    loss = loss_fn(preds, labels)

    # Add on to the total loss
    total_loss = total_loss + loss.item()

    # Backward pass to calculate the gradients
    loss.backward()

    # Clip the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters
    optimizer.step()

    # Model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # Append the model predictions
    total_preds.append(preds)

  # Compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # Predictions are in the form of (no. of batches, size of batch, no. of classes).
  # Reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  if wandb_log:
    wandb.log({
      "train_loss": avg_loss
    })

  # Returns the loss and predictions
  return avg_loss, total_preds

# Function for evaluating the model
def evaluate():
  
  print("\nEvaluating...")
  
  # Deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # Empty list to save the model predictions
  total_preds = []

  # Iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches
    if (step + 1) % 50 == 0 or step == len(val_dataloader) - 1:
            
      # Report progress
      print('  Batch {:>5,}  of  {:>5,}.'.format(step + 1, len(val_dataloader)))

    # Push the batch to GPU
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # Deactivate autograd
    with torch.no_grad():
      
      # Model predictions
      preds = model(sent_id, mask)[0]

      # Compute the validation loss between actual and predicted values
      loss = loss_fn(preds, labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # Compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # Reshape the predictions in form of (number of samples, no. of classes)
  total_preds = np.concatenate(total_preds, axis=0)

  if wandb_log:
    wandb.log({
      "val_loss": avg_loss
    })

  return avg_loss, total_preds

# Set initial loss to infinite
best_validation_loss = float('inf')

# Empty lists to store training and validation loss of each epoch
training_losses=[]
validation_losses=[]
train_flag = True
if train_flag == True:
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
      
        # Train model
        training_loss, _ = train()
      
        # Evaluate model
        validation_loss, _ = evaluate()
      
        # Save the best model
        if validation_loss < best_validation_loss:
          best_validation_loss = validation_loss
          torch.save(model.state_dict(), args.save_path)
        
        # Append training and validation loss
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        
        print(f'\nTraining Loss: {training_loss:.3f}')
        print(f'Validation Loss: {validation_loss:.3f}')

        if wandb_log:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": training_loss,
                "val_loss": validation_loss
            })


# Test safety classifier
safe_test = f"{args.data_dir}/safe_prompts_test.txt"
safe_prompt_test = read_text(safe_test)
harm_test = f"{args.data_dir}/harmful_prompts_test.txt"
harm_prompt_test = read_text(harm_test)
prompt_data_test = pd.concat([safe_prompt_test, harm_prompt_test], ignore_index=True)
prompt_data_test['Y'] = pd.Series(np.concatenate([np.ones(safe_prompt_test.shape[0]), np.zeros(harm_prompt_test.shape[0])])).astype(int)

test_text = prompt_data_test[0]
test_labels = prompt_data_test['Y']

# Tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

# Load weights of best model
path = args.save_path
model.load_state_dict(torch.load(path))
model.eval()

# Get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))[0]
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(f'Testing Accuracy = {100*torch.sum(torch.tensor(preds) == test_y)/test_y.shape[0]}%')
print(classification_report(test_y, preds))

if wandb_log:
    wandb.log({
        "test_acc": 100*torch.sum(torch.tensor(preds) == test_y)/test_y.shape[0]
    })

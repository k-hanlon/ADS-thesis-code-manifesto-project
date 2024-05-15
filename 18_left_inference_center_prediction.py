import torch
import transformers
import numpy as np
from datasets import load_dataset
from huggingface_hub import notebook_login
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import pandas as pd

use_auth_token="..."

# load the correct dataset with the inference set from huggingface:
raw_datasets = load_dataset("kghanlon/left_as_train_context", token = use_auth_token)
# the dataset needs a "label" column, so rename accordingly
raw_datasets = raw_datasets.rename_column('RILE', 'label')
# which part of it is for inference?
inference_set = raw_datasets["inference_center"]

# Function to predict the labels (as the context dataset has already tokenized data, we don't tokenize here)
# Additionally, creating a dataloader with correct padding is a bit annoying, so we just use a batch size of 1
def predict_model(model_checkpoint, dataset, n_labels):
    # load the model tokenizer:
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Prepare DataLoader of the dataset
    # use a batch size of 1, as we're not tokenizing and the input_ids will have different lengths (otherwise we need to pad manually)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # load the model (that we are using for sequence classification)
    m = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels = n_labels)
    m.to('cuda')

    # go through the batches, tokenize them and predict
    # put model into evaluation mode, changes some behaviour (like dropout etc.)
    m.eval()
    # save the predictions
    preds = []

    for batch in tqdm(data_loader, desc="Predicting", leave=True):
        # Extract pre-encoded input_ids and attention_masks from the dataset
        input_ids = torch.cat(batch["input_ids"], dim=0).to('cuda')
        attention_mask = torch.cat(batch["attention_mask"], dim=0).to('cuda')

        # Predict using the model
        logits = m(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)).logits
        predicted_classes = torch.argmax(logits, dim=1)
        preds.extend(predicted_classes.tolist())

    # Clear GPU memory for the next task
    torch.cuda.empty_cache()

    return preds

# turn the test dataset into a dataframe
df = pd.DataFrame(inference_set)
# put in correct checkpoint of the model:
df["preds"] = predict_model("/mimer/NOBACKUP/groups/naiss2024-22-264/kghanlon/left_as_train_context_roberta-large_20e/checkpoint-7260", inference_set, 3)

# save the dataframe:
df.to_csv("left_inference_center_predictions.csv", index=False)
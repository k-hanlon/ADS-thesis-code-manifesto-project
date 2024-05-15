import time
start = time.time()

import torch
import transformers
import numpy as np
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# function that computes metrics for every epoch
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)

    # Compute precision
    precision = precision_score(labels, predictions, average='binary')

    # Compute recall
    recall = recall_score(labels, predictions, average='binary')

    # Compute F1 score
    f1 = f1_score(labels, predictions, average='binary')

    # You can add more custom metrics as needed
    return {
        'val_accuracy': accuracy,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
    }


auth_token="..."

# load the data:
dataset_name = "kghanlon/green_as_train_context"
# data_context already includes tokenized input_ids and attention mask
data_context = load_dataset(dataset_name, token=auth_token)

# the datasets needs a "label" column, so rename accordingly
target = "green_code"
data_context = data_context.rename_column(target, 'label')

# Load the correct model. This has to be the model that was also used to create the context dataset
checkpoint = "FacebookAI/roberta-large"
# load its tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Create a data collator that adds padding to the batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load the model (that we are using for sequence classification)
# Adjust number of labels as needed (2 for Green, 3 for RILE)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to("cuda")

# repo name for the new model on huggingface:
repo_name = dataset_name + "_roberta-large_20e"

# set up the trainer:
# learning rate (lr for manifestoBERTa is 1e-05, default is 5e-05, might have to adjust it?)
lr = 5e-06

training_args = TrainingArguments(repo_name,
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  push_to_hub=True,
                                  hub_token = auth_token,
                                  num_train_epochs=20,
                                  learning_rate = lr,
                                  fp16 = True, # mixed precision training? --> speeds up calculations by a lot!
                                  per_device_train_batch_size = 16 # 8 is default (16 is too high for T4)
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=data_context["train"],
    eval_dataset=data_context["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub("")

end = time.time()
print("Final time:", end - start)
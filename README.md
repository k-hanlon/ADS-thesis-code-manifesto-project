# ADS-thesis-code-manifesto-project
# Using Language Models to evaluate annotation bias in the Manifesto Project Corpus
## Thesis by Leo Carlsson and Konstantin Hanlon, Gothenburg University, 2024

## This repo contains the code used to create and train the models used in the the thesis and any other analysis done. Reproduction requires API keys for HuggingFace and the manifestoR API.

## Overview of the scripts:
- 01_get_corpus_data.R: Loads the data from the manifestoR API using the 2023a Corpus and all English language manifestos, saves it locally.
- 02_creating_full_corpus_df.ipynb: Combines the manifesto text data with the manifesto level data like party, coder_id, date etc.
- 03_party_split_strategy.qmd: Splits all the parties in the dataset into green/non-green and left/center/right categories.
- 04_create_training_datasets.ipynb: Creates train, validation and test sets based on the party and manifesto splits.
- 05_corpus_2023a_description.ipynb: EDA of the 2023a MP corpus data.
- 06_cos_sim_multiprocessing.py: Runs the cosine similarity calculations using the python multiprocessing package.
- 07_cosine_similarity_analysis.ipynb: Analysis of the cosine similarity results.
- 08_add_context_to_datasets.ipynb: creates tokenized datasets where the quasi-sentences also include a 200 word context around them.
- 09 - 12: Training scripts for the RoBERTa models using the tokenized context data.
- 13 - 22: Predicting the classes for the test and inference sets using the different models.
- 23 - 25: Analysis of the model predictions.
- 26_finding_examples.ipynb: Using the results from the analysis and the cosine similarity data, find example instances of bias in the manifesto texts.

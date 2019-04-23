#!/bin/bash

# Create folder structure
mkdir .data
mkdir .data/snli
mkdir .vector_cache

# Download and unzip GloVe word embeddings
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -a glove.840B.300d.zip -d .vector_cache/
rm -f glove.840B.300d.zip

# Download and unzip NLI corpora
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip -a snli_1.0.zip -d .data/snli/
rm -f snli_1.0.zip

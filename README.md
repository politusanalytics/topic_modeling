# topic_modeling

This repository contains a Python script for topic modeling Turkish tweets using BERTopic and OpenAI.

## Usage
`python topic_modeling.py <file_path> <topic_domain> <min_cluster_size> <num_groups>`

## Arguments
- `<file_path>`: Path to the input file (.csv, .xlsx, .json, or .jsonl)
- `<topic_domain>`: Domain name for labeling context (e.g., "Koç Üniversitesi")
- `<min_cluster_size>`: HDBSCAN's min_cluster_size parameter
- `<num_groups>`: Number of groups to categorize the topic labels into

The script outputs two Excel files: one for topic summaries and one for tweet-level results.
Make sure your OpenAI API key is set in the environment as `OPENAI_API_KEY`.

# Offline Handwritten Signature Verification Using a Stream-Based Approach

This repository provides tools for generating dissimilarity data, training and testing writer-independent classifiers in batch mode, and converting batch data into a stream of handwritten signatures. Additionally, it implements a prequential evaluation (test-then-train) approach to assess the performance of classifiers on the stream.

## Installation

This package requires python 3. Installation can be done with pip:

```bash
pip install git+https://github.com/kdMoura/stream_hsv.git
```

You can also clone this repository and install it with ```pip install -e <path/to/repository> ```
## Usage

### Data generation
The `data.py` script is designed to generate dissimilarity data for both training and testing. 
Before applying the generation process, the datasets should:
1. Undergo preprocessing as outlined in https://github.com/tallesbrito/contrastive_sigver.
2. Have features extracted using the SigNet-S available in the same repository.
3. The resulting data should be a single .NPZ file containing:
   - `features`: shape (samples, features)
   - `y`: writer ID
   - `yforg`: 1 for forgery, 0 otherwise
     
To generate dissimilarity training datasets, use the `training_generation` subcommand: 
 
```bash
python -m shsv.data training_generation --n-data 5  \
--input-data <data.npz> --f-output-path /path/to/output/folder  \
--n-gen 12 --include-users 300 350
```

To generate dissimilarity test datasets, use the `test_generation` subcommand: 
```bash
python -m shsv.data test_generation --n-data 5  \
--input-data <data.npz> --f-output-path /path/to/output/folder \
--n-ref 12 --n-query 10 --include-users 0 300
```

### Model Learning and Testing

The `model.py` script is used for batch mode training and testing and also for stream prequential (test-then-train) process. 

1. **Batch mode training:**
```bash
python -m shsv.model batch_train --model sgd \
--f-train-path /path/to/train/datasets/folder --f-output-path /path/to/output/folder
```

2. **Batch mode test:**
```bash
python -m shsv.model batch_test --f-model-path /path/to/models/folder \
--f-test-path /path/to/test/data/folder  \
--f-output-path /path/to/output/folder
```

3. **Stream prequential:**
```bash
python -m shsv.model prequential --f-model-path /path/to/models/folder \
--f-test-path /path/to/test/data/folder \
--f-output-path /path/to/output/folder \
--n-ref 12 --chunk-size 600 --file-number 0
```

### Evaluation

The `evaluation.py` script contains functions to compute metrics on batch and stream predictions files.

1. **Batch evaluation:**
```bash
python -m shsv.evaluation batch --f-input-path /path/to/predictions/folder --f-output-path /path/to/output/folder \
--n-ref 1 2 3 5 10 12 --forgery skilled --thr-type global --fusions max mean min median
```

2. **Stream evaluation:**
```bash
python -m shsv.evaluation stream --f-input-path /path/to/predictions/folder --f-output-path /path/to/output/folder \
--forgery skilled --thr-type global --fusions max \
--window-size 400 --window-step 400
```


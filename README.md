# Detecting Adversarial Prompts in Hindi-Devanagari

This repository contains code for our project on detecting adversarial prompts in Hindi-Devanagari. We aim to tackle adversarial prompting in Hindi by adapting [Erase-and-Check](https://github.com/aounon/certified-llm-safety) to the Hindi language. We perform several experiments to analyze the effectiveness of our approach with various models, prompts, and erasure modes. 

## Installation

To install the necessary dependencies, run the following command:

```bash
conda create -n adv-hindi python=3.10
conda activate adv-hindi
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Training the Safety Classifiers

## Basic Inference on Safety Classifiers
We perform inference on both LLama2 and fine-tuned safety classifiers, loaded from model paths.

### LLM Inference
To utilize the LLM models for inference from HuggingFace, use the below command. This runs Llama2-7B:
```bash
python main.py --eval_type harmful --harmful_prompts data/harmful_prompts.txt
```

### Fine-tuned Safety Classifiers Inference
To utilize the fine-tuned safety classifiers for inference, you must have a saved model path for the fine-tuned safety classifier. Then run the below commands depending on your model of interest. 

DistilBERT:
```bash
python main.py --use_classifier --classifier_name distilbert  --data_dir hindi_data/hindi --eval_type all_data --model_wt_path models/distilbert_hindi.pt 
```
DistilBERT-multilingual:
```bash
python main.py --use_classifier --classifier_name distilbert-multi  --data_dir hindi_data/hindi --eval_type all_data --model_wt_path models/distilbert_multi_hindi.pt 
```
IndicBERT:
```bash
python main.py --use_classifier --classifier_name indictbert  --data_dir hindi_data/hindi --eval_type all_data --model_wt_path models/indicbert_hindi.pt 
```

## Adversarial Prompting Experiments with Erase and Check

### Phrase-augmented adversarial prompts 

#### Select either DistilBERT-multilingual or IndicBERT Safety Classifier, and Prefix/Suffix/Insertion EC mode
```bash
python main.py --use_classifier --classifier_name distilbert-multi --data_dir hindi_data/hindi --eval_type ec_all_data --model_wt_path models/distilbert_multi_hindi.pt --mode prefix 
```

### Encapsulation-augmented adversarial prompts

#### Select either DistilBERT-multilingual or IndicBERT Safety Classifier, and Prefix/Suffix/Insertion/Greedy EC mode
```bash
python main.py --use_classifier --classifier_name distilbert-multi --data_dir hindi_data/hindi --eval_type ec_all_data --model_wt_path models/distilbert_multi_hindi.pt --mode prefix --hidden_harmful
```

### Based on the above commands using the various configurations we attain the following results:

#### Baseline Fine-Tuned Safety Filter Classification on Hindi Prompt Dataset
The results in Table 1 show the performance of baseline fine-tuned safety classifiers on the Hindi prompt dataset. Both models detect 72.2% of harmful prompts.  DistilBERT-multilingual achieves perfect classification for safe prompts (100%), while IndicBERT performs slightly worse with 88.9%. 

##### Table 1: Baseline Fine-Tuned Safety Filter Classification

| Model             | % Harmful | % Safe | % Accuracy | F1   | Precision | Recall |
|--------------------|-----------|--------|------------|------|-----------|--------|
| DistilBERT-multi   | 72.2      | 100    | 86.1       | 0.84 | 1         | 0.72   |
| IndicBERT          | 72.2      | 88.9   | 80.6       | 0.79 | 0.87      | 0.72   |

#### Phrase-Augmented Prompts: Safety Filter Classification
Tables 2 and 3 show that DistilBERT performs best with end augmentations (F1 = 0.89), while IndicBERT achieves its highest F1 score (0.83) with middle augmentations. Both models perform worst with beginning augmentations, highlighting their vulnerability to safe tokens at the start of prompts.

##### Table 2: Phrase-Augmented Prompts (DistilBERT-Multilingual)

| Augmentation Location | % Harmful | F1   |
|------------------------|-----------|------|
| Beginning              | 55.6      | 0.71 |
| Middle                 | 66.7      | 0.79 |
| End                    | 80.6      | 0.89 |

##### Table 3: Phrase-Augmented Prompts (IndicBERT)

| Augmentation Location | % Harmful | F1   |
|------------------------|-----------|------|
| Beginning              | 66.7      | 0.75 |
| Middle                 | 75.0      | 0.83 |
| End                    | 69.4      | 0.76 |


#### Erase-and-Check Results for Phrase-Augmented Prompts
The results in Table 4 and 5 demonstrate the performance of erase-and-check strategies for DistilBERT-multilingual and IndicBERT on phrase-augmented prompts. Prefix mode achieves perfect harmful detection for DistilBERT (100%) but struggles with safe prompts. The greedy mode shows the highest harmful detection for IndicBERT (94.4%) with balanced safe prompt detection, but insertion mode is noted as computationally infeasible.

##### Table 4: Erase-and-Check Results for Phrase-Augmented Prompts (DistilBERT-Multilingual)
| EC Mode   | % Harmful | % Safe | F1   |
|-----------|-----------|--------|------|
| Prefix    | 100       | 38     | 0.70 |
| Insertion | Too slow  | -      | -    |
| Suffix    | 88.9      | 90     | 0.88 |

##### Table 4: Erase-and-Check Results for Phrase-Augmented Prompts (IndicBERT)

| EC Mode   | % Harmful | % Safe | F1   |
|-----------|-----------|--------|------|
| Prefix    | 77.8      | 12     | 0.52 |
| Insertion | Too slow  | -      | -    |
| Suffix    | 94.4      | 70     | 0.80 |

#### Baseline Results for Encapsulation-Augmented Prompts

Table 5 shows the performance of baseline fine-tuned safety classifiers on encapsulation-augmented prompts. Both models struggle with the increased complexity and length of these prompts, with DistilBERT achieving 13.9% harmful detection and IndicBERT achieving 8.3%.

Table 5: Baseline Results for Encapsulation-Augmented Prompts

| Model            | % Harmful | % Safe | F1   |
|-------------------|-----------|--------|------|
| DistilBERT-multi  | 13.9      | 100    | 0.24 |
| IndicBERT         | 8.3       | 86.1   | 0.14 |

#### Erase-and-Check Results for Encapsulation-Augmented Prompts
Table 6 and 7 highlight the results for erase-and-check strategies applied to encapsulation-augmented prompts. Prefix mode achieves the best balance for DistilBERT with 86.1% harmful detection and an F1 score of 0.70. For IndicBERT, greedy mode achieves the highest harmful detection (75.0%) but at the cost of high runtime (0.8 seconds per prompt). Suffix mode performs well for safe prompts but struggles a lot with harmful content.

##### Table 6: Erase-and-Check Results for Encapsulation-Augmented Prompts (DistilBERT-Multilingual)

| EC Mode   | % Harmful | % Safe | F1   | Time/Prompt (sec) |
|-----------|-----------|--------|------|-------------------|
| Prefix    | 86.1      | 38     | 0.70 | 0.14              |
| Suffix    | 19.4      | 90     | 0.29 | 0.13              |
| Greedy    | 97.2      | NR     | NR   | 0.63              |

##### Table 7: Erase-and-Check Results for Encapsulation-Augmented Prompts (IndicBERT)

| EC Mode   | % Harmful | % Safe | F1   | Time/Prompt (sec) |
|-----------|-----------|--------|------|-------------------|
| Prefix    | 44.4      | 12     | 0.33 | 0.2               |
| Suffix    | 11.1      | 70     | 0.15 | 0.2               |
| Greedy    | 75.0      | NR     | NR   | 0.8               |


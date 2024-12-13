import torch
import wandb
import transformers
from transformers import AutoTokenizer
import os
import time
import random
import argparse
import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from math import ceil
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from defenses import is_harmful
from defenses import progress_bar, erase_and_check
from prompt_augmentation import get_phrases, get_safe_prompts, get_harmful_prompts
from greedy_ec import greedy_ec
from beam_search_ec import beam_search_ec
from openai import OpenAI

from simulated_annealing_ec import simulated_annealing_ec
parser = argparse.ArgumentParser(description='Check safety of prompts.')
# -- Erase and Check arguments -- #
parser.add_argument('--mode', type=str, default="base", choices=["base", "suffix", "insertion", "prefix"],
                    help='attack mode to defend against')
parser.add_argument("--hidden_harmful", action="store_true", help="use hidden harmful prompts within safe prompts")
parser.add_argument('--data_dir', type=str, default="data",
                    help='directory containing the prompts')
parser.add_argument('--eval_type', type=str, default="all_data", choices=["all_data", "greedy_ec", "ec_all_data", "beam_search_ec", "simulated_annealing_ec"],
                    help='type of prompts to evaluate')
parser.add_argument('--max_erase', type=int, default=20,
                    help='maximum number of tokens to erase')
parser.add_argument('--num_iters', type=float, default=10,
                    help='number of iterations for the Erase-and-Check procedure for GreedyEC')
# -- Safety Classifier arguments -- #
parser.add_argument('--use_classifier', action='store_true',
                    help='flag for using a custom trained safety filter')
parser.add_argument('--classifier_name', type=str, default="distilbert", choices=["distilbert", "distilbert-multi", "indicbert"],
                    help='name of the classifier model')
parser.add_argument('--model_wt_path', type=str, default="",
                    help='path to the model weights of the trained safety filter')
parser.add_argument('--llm_name', type=str, default="Llama-2", choices=["Llama-2", "Llama-2-13B", "Llama-3", "GPT-3.5"],
                    help='name of the LLM model (used only when use_classifier=False)')
parser.add_argument('--max_seq_len', type=int, default=200,
                    help='maximum sequence length for the LLM')

# -- Wandb logging arguments -- #
parser.add_argument('--wandb_log', action='store_true',
                    help='flag for logging results to wandb')
parser.add_argument('--wandb_project', type=str, default="llm-hindi-safety-filter",
                    help='name of the wandb project')
parser.add_argument('--wandb_entity', type=str, default="patchtst-flashattention",
                    help='name of the wandb entity')

args = parser.parse_args()

np.random.seed(4)

hidden_harmful = args.hidden_harmful
data_dir = args.data_dir
mode = args.mode
eval_type = args.eval_type
max_erase = args.max_erase
use_classifier = args.use_classifier
classifier_name = args.classifier_name
model_wt_path = args.model_wt_path
max_seq_len = args.max_seq_len
num_iters = args.num_iters
llm_name = args.llm_name
wandb_log = args.wandb_log
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity

print("\n* * * * * * Experiment Details * * * * * *")
if torch.cuda.is_available():
    print("Device: " + torch.cuda.get_device_name(0))
print("Evaluation type: " + eval_type)

if use_classifier:
    print("Using custom safety filter. Model weights path: " + model_wt_path)
else:
    print("Using LLM model: " + llm_name)
if eval_type == "safe" or eval_type == "empirical":
    print("Mode: " + mode)
    print("Maximum tokens to erase: " + str(max_erase))


# Log the experiment configuration to wandb
if wandb_log:
    wandb.init(project=wandb_project, entity=wandb_entity)
    wandb.config.update({
        "mode": mode,
        "eval_type": eval_type,
        "max_erase": max_erase,
        "use_classifier": use_classifier,
        "classifier_name": classifier_name,
        "model_wt_path": model_wt_path,
        "llm_name": llm_name,
        "max_seq_len": max_seq_len,
        "hidden_harmful": hidden_harmful
    })

if use_classifier:
    # Using custom classifier for safety filter
    # Load model and tokenizer
    if classifier_name == "distilbert-multi":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')
    elif classifier_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    elif classifier_name == "indicbert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('ai4bharat/indic-bert', keep_accents=True)
        model = AutoModelForSequenceClassification.from_pretrained('ai4bharat/indic-bert')

    if model_wt_path != "":
        model.load_state_dict(torch.load(model_wt_path), strict=True)
    model.eval()

    # Create a text classification pipeline
    pipeline = transformers.pipeline('text-classification', model=model, tokenizer=tokenizer, device=0)

else:
    # Using LLM for safety filter
    if llm_name == "Llama-2":
        # Load model and tokenizer
        model = "meta-llama/Llama-2-7b-chat-hf"

        print(f'Loading model {model}...')
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif llm_name == "Llama-2-13B":
        # Load model and tokenizer
        model = "meta-llama/Llama-2-13b-chat-hf"

        print(f'Loading model {model}...')
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto")

    elif llm_name == "Llama-3":
        # Load model and tokenizer
        model = "meta-llama/Meta-Llama-3-8B-Instruct"

        print(f'Loading model {model}...')
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif llm_name == "GPT-3.5":
        with open('key.txt', 'r') as file:
            key = file.read()

        pipeline = OpenAI(api_key=key)

        # Tokenizer
        model = "meta-llama/Llama-2-7b-chat-hf"
        commit_id = "08751db2aca9bf2f7f80d2e516117a53d7450235"
        tokenizer = AutoTokenizer.from_pretrained(model, revision=commit_id)

if use_classifier:
    print("Using fine-tuned classifier")
    safe_prompts_file = f"{data_dir}/safe_prompts_test.txt"
    harmful_prompts_file = f"{data_dir}/harmful_prompts_test.txt"
    prefsuffix_file = f"{data_dir}/prefsuffix.txt"
    insertion_file = f"{data_dir}/insertion.txt"
else:
    safe_prompts_file = f"{data_dir}/safe_prompts.txt"
    harmful_prompts_file = f"{data_dir}/harmful_prompts.txt"
    prefsuffix_file = f"{data_dir}/prefsuffix.txt"
    insertion_file = f"{data_dir}/insertion.txt"

if mode == "base":
    phrases = None
else:
    phrases = get_phrases(mode, prefsuffix_file, insertion_file)

if eval_type == "ec_all_data":
    # Check the mode
    print(f"Evaluating with mode: {mode}")

    # Harmful prompts
    print("\nEvaluating harmful prompts from: " + harmful_prompts_file + "\n")
    harmful_prompts = get_harmful_prompts(harmful_prompts_file, mode, hidden_harmful, phrases)

    # Safe prompts
    print("\nEvaluating safe prompts from: " + safe_prompts_file + "\n")
    safe_prompts = get_safe_prompts(safe_prompts_file)

    # Check if the prompts are harmful
    count_harmful = 0
    y_true_harmful = [1] * len(harmful_prompts)
    y_pred_harmful = []
    start_time = time.time()
    time_list = []
    elapsed_time = 0
    for i in range(len(harmful_prompts)):
        prompt = harmful_prompts[i]
        # if args.append_adv: prompt += adv_suffix
        harmful = erase_and_check(prompt, pipeline, tokenizer, max_erase=max_erase, mode=mode, llm_name=llm_name)
        

        y_pred_harmful.append(1 if harmful else 0)

        if harmful:
            count_harmful += 1

        current_time = time.time()
        time_list.append(current_time - start_time - elapsed_time)
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (i + 1)
        percent_harmful = count_harmful / (i + 1) * 100
        print("    Checking harmfulness... " + progress_bar((i + 1) / len(harmful_prompts)) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
        if wandb_log:
            wandb.log({
                f"percent_harmful_prompt_epoch": percent_harmful,
                f"time_per_prompt_prompt_epoch": time_per_prompt
            })

    percent_harmful = count_harmful / len(harmful_prompts) * 100
    print(f"Percentage of harmful prompts: {percent_harmful:.2f}%")
    if wandb_log:
        wandb.log({
            "percent_harmful": percent_harmful
        })
        
    # Compute standard error of the average time per prompt
    time_per_prompt_se = torch.tensor(time_list).std().item() / (len(harmful_prompts) ** 0.5)
    avg_time_per_prompt = np.mean(time_list)

    # Compute standard error of the percentage of harmful prompts
    percent_harmful_se = (percent_harmful * (100 - percent_harmful) / (len(harmful_prompts) - 1)) ** 0.5

    if wandb_log:
        wandb.log({
            "percent_harmful_standard_error": percent_harmful_se,
            "time_per_prompt_standard_error_harmful": time_per_prompt_se,
            "avg_time_per_prompt": avg_time_per_prompt
        })

    # Check if the prompts are safe
    count_safe = 0
    y_true_safe = [0] * len(safe_prompts)
    y_pred_safe = []
    start_time = time.time()
    time_list = []
    elapsed_time = 0
    for i in range(len(safe_prompts)):
        prompt = safe_prompts[i]
        # if args.append_adv: prompt += adv_suffix
        harmful = erase_and_check(prompt, pipeline, tokenizer, max_erase=max_erase, mode=mode, llm_name=llm_name)
        
        y_pred_safe.append(1 if harmful else 0)
        if not harmful:
            count_safe += 1

        current_time = time.time()
        time_list.append(current_time - start_time - elapsed_time)
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (i + 1)
        percent_safe = count_safe / (i + 1) * 100
        print("    Checking safety... " + progress_bar((i + 1) / len(safe_prompts)) \
            + f' Detected safe = {percent_safe:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
        if wandb_log:
            wandb.log({
                f"percent_safe_prompt_epoch": percent_harmful,
                f"time_per_prompt_prompt_epoch": time_per_prompt
            })

    avg_time_per_prompt = np.mean(avg_time_per_prompt)
    percent_safe = count_safe / len(safe_prompts) * 100
    print(f"Percentage of safe prompts: {percent_safe:.2f}%")
    if wandb_log:
        wandb.log({
            "percent_safe": percent_safe,
            "avg_time_per_prompt": avg_time_per_prompt,
        })

    # Combine harmful and safe results
    y_true = y_true_harmful + y_true_safe
    y_pred = y_pred_harmful + y_pred_safe

    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\nF1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    if wandb_log:
        wandb.log({
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        })

elif eval_type == "greedy_ec" or eval_type == "beam_search_ec" or eval_type == "simulated_annealing_ec":
    # Evaluating the performance of GreedyEC on adversarial prompts
    if not use_classifier:
        print("Option --use_classifier must be turned on. GreedyEC only works with a trained safety classifier.")
        exit()

    # Harmful prompts
    print("\nEvaluating harmful prompts from: " + harmful_prompts_file + "\n")
    # Load prompts from text file
    prompts = get_harmful_prompts(harmful_prompts_file, mode, hidden_harmful, phrases)

    num_prompts = len(prompts)

    # Check if the prompts are harmful
    count_harmful = 0
    start_time = time.time()
    time_list = []
    elapsed_time = 0
    for i in range(num_prompts):
        prompt = prompts[i]
        if eval_type == "greedy_ec":
            harmful = greedy_ec(prompt, model, tokenizer, num_iters=num_iters)
        elif eval_type == "beam_search_ec":
            harmful = beam_search_ec(prompt, model, tokenizer, num_iters=num_iters)
        elif eval_type == "simulated_annealing_ec":
            harmful = simulated_annealing_ec(prompt, model, tokenizer, num_iters=num_iters)
        
        if harmful:
            count_harmful += 1

        current_time = time.time()
        time_list.append(current_time - start_time - elapsed_time)
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (i + 1)
        percent_harmful = count_harmful / (i + 1) * 100
        print("    Checking safety... " + progress_bar((i + 1) / num_prompts) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
        if wandb_log:
            wandb.log({
                f"percent_harmful_prompt_epoch": percent_harmful,
                f"time_per_prompt_prompt_epoch": time_per_prompt
            })
            
        print("")

        # Compute standard error of the average time per prompt
        time_per_prompt_se = torch.tensor(time_list).std().item() / (num_prompts ** 0.5)

        avg_time_per_prompt = np.mean(time_list)
        # Compute standard error of the percentage of harmful prompts
        percent_harmful_se = (percent_harmful * (100 - percent_harmful) / (num_prompts - 1)) ** 0.5

        if wandb_log:
            wandb.log({
                "percent_harmful_standard_error": percent_harmful_se,
                "time_per_prompt_standard_error": time_per_prompt_se,
                "avg_time_per_prompt": avg_time_per_prompt
            })

elif eval_type == "all_data":
    # Harmful prompts
    print("\nEvaluating harmful prompts from: " + harmful_prompts_file + "\n")
    harmful_prompts = get_harmful_prompts(harmful_prompts_file, mode, hidden_harmful, phrases)

    num_prompts = len(harmful_prompts)

    # Check if the prompts are harmful
    y_true_harmful = [1] * len(harmful_prompts)
    y_pred_harmful = []
    count_harmful = 0
    batch_size = 10
    start_time = time.time()
    for i in range(0, num_prompts, batch_size):
        batch = harmful_prompts[i:i+batch_size]
        # Evaluating the safety filter gives us certifed safety guarantees on
        # erase_and_check for harmful prompts (from construction).
        harmful = is_harmful(batch, pipeline, tokenizer, llm_name=llm_name, max_llm_sequence_len=max_seq_len)
        count_harmful += sum(harmful)
        y_pred_harmful.extend(harmful)
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (i + batch_size)
        num_done = i + batch_size
        percent_harmful = count_harmful / num_done * 100
        print("    Checking safety... " + progress_bar(num_done / num_prompts) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
        if wandb_log:
            wandb.log({
                f"percent_harmful_prompt_epoch": percent_harmful,
                f"time_per_prompt_prompt_epoch": time_per_prompt,
            })

        if i + batch_size >= num_prompts:
            # Process the remaining batch
            batch = harmful_prompts[i+batch_size:]
            harmful = is_harmful(batch, pipeline, tokenizer, llm_name=llm_name, max_llm_sequence_len=max_seq_len)
            # incorrect_indices = incorrect_indices.extend(np.where(harmful == 0)[0] + (i * batch_size))
            count_harmful += sum(harmful)
            y_pred_harmful.extend(harmful)

            current_time = time.time()
            elapsed_time = current_time - start_time
            time_per_prompt = elapsed_time / num_prompts
            percent_harmful = count_harmful / num_prompts * 100
            print("    Checking safety... " + progress_bar(num_done / num_prompts) \
                + f' Detected harmful = {percent_harmful:5.1f}%' \
                + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
            if wandb_log:
                wandb.log({
                    f"percent_harmful_prompt_epoch": percent_harmful,
                    f"time_per_prompt_prompt_epoch": time_per_prompt,
                })

    print("")
    percent_harmful = count_harmful / num_prompts * 100

    if wandb_log:
        wandb.log({
            "percent_harmful": percent_harmful
        })
    # Safe prompts
    print("\nEvaluating safe prompts from: " + safe_prompts_file + "\n")
    safe_prompts = get_safe_prompts(safe_prompts_file)

    num_prompts = len(safe_prompts)

    # Check if the prompts are safe
    y_true_safe = [0] * len(safe_prompts)
    y_pred_safe = []
    count_safe = 0
    batch_size = 10
    start_time = time.time()
    # incorrect_indices = []
    for i in range(0, num_prompts, batch_size):
        batch = safe_prompts[i:i+batch_size]
        # Evaluating the safety filter gives us certifed safety guarantees on
        # erase_and_check for harmful prompts (from construction).
        harmful = is_harmful(batch, pipeline, tokenizer, llm_name=llm_name, max_llm_sequence_len=max_seq_len)
        count_safe += len(harmful) - sum(harmful)
        y_pred_safe.extend(harmful)

        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (i + batch_size)
        num_done = i + batch_size
        percent_safe = count_safe / num_done * 100
        print("    Checking safety... " + progress_bar(num_done / num_prompts) \
            + f' Detected safe = {percent_safe:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
        if wandb_log:
            wandb.log({
                f"percent_safe_prompt_epoch": percent_safe,
                f"time_per_prompt_prompt_epoch": time_per_prompt,
            })

        if i + batch_size >= num_prompts:
            # Process the remaining batch
            batch = safe_prompts[i+batch_size:]
            harmful = is_harmful(batch, pipeline, tokenizer, llm_name=llm_name, max_llm_sequence_len=max_seq_len)
            count_safe += len(harmful) - sum(harmful)
            y_pred_safe.extend(1 - np.array(harmful))

            current_time = time.time()
            elapsed_time = current_time - start_time
            time_per_prompt = elapsed_time / num_prompts
            percent_safe = count_safe / num_prompts * 100
            print("    Checking safety... " + progress_bar(num_done / num_prompts) \
                + f' Detected safe = {percent_safe:5.1f}%' \
                + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
            if wandb_log:
                wandb.log({
                    f"percent_safe_prompt_epoch": percent_safe,
                    f"time_per_prompt_prompt_epoch": time_per_prompt,
                })
    
    print("")
    percent_safe = count_safe / num_prompts * 100

    if wandb_log:
        wandb.log({
            "percent_safe": percent_safe
        })

    # Combine harmful and safe results
    y_true = y_true_harmful + y_true_safe
    y_pred = y_pred_harmful + y_pred_safe

    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"\nF1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    if wandb_log:
        wandb.log({
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        })
    
    # Log the total accuracy
    total_accuracy = (count_harmful + count_safe) / len(safe_prompts + harmful_prompts) * 100
    if wandb_log:
        wandb.log({
            "total_accuracy": total_accuracy
        })

print("")

if wandb_log:
    wandb.finish()

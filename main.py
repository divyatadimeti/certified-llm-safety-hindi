import torch
import wandb
import transformers
from transformers import AutoTokenizer
import os
import time
import json
import ast
import random
import argparse
import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from math import ceil
import numpy as np

from defenses import is_harmful
from defenses import progress_bar, erase_and_check, erase_and_check_smoothing
from grad_ec import grad_ec
from greedy_ec import greedy_ec
from greedy_grad_ec import greedy_grad_ec

from openai import OpenAI

parser = argparse.ArgumentParser(description='Check safety of prompts.')
parser.add_argument('--num_prompts', type=int, default=2,
                    help='number of prompts to check')
parser.add_argument('--mode', type=str, default="suffix", choices=["suffix", "insertion", "infusion"],
                    help='attack mode to defend against')
parser.add_argument('--data_dir', type=str, default="data",
                    help='directory containing the prompts')
parser.add_argument('--eval_type', type=str, default="safe", choices=["safe", "harmful", "smoothing", "empirical", "grad_ec", "greedy_ec", "roc_curve"],
                    help='type of prompts to evaluate')
parser.add_argument('--max_erase', type=int, default=20,
                    help='maximum number of tokens to erase')
parser.add_argument('--num_adv', type=int, default=2,
                    help='number of adversarial prompts to defend against (insertion mode only)')
parser.add_argument('--attack', type=str, default="gcg", choices=["gcg", "autodan"],
                    help='attack to defend against')
parser.add_argument('--adv_prompts_dir', type=str, default="data",
                    help='directory containing adversarial prompts')

# use adversarial prompt or not
# parser.add_argument('--append-adv', action='store_true',
#                     help="Append adversarial prompt")

# -- Randomizer arguments -- #
parser.add_argument('--randomize', action='store_true',
                    help="Use randomized check")
parser.add_argument('--sampling_ratio', type=float, default=0.1,
                    help="Ratio of subsequences to evaluate (if randomize=True)")
# -------------------------- #

parser.add_argument('--results_dir', type=str, default="results",
                    help='directory to save results')
parser.add_argument('--use_classifier', action='store_true',
                    help='flag for using a custom trained safety filter')
parser.add_argument('--classifier_name', type=str, default="distilbert", choices=["distilbert", "indicbert"],
                    help='name of the classifier model')
parser.add_argument('--model_wt_path', type=str, default="",
                    help='path to the model weights of the trained safety filter')
parser.add_argument('--llm_name', type=str, default="Llama-2", choices=["Llama-2", "Llama-2-13B", "Llama-3", "GPT-3.5"],
                    help='name of the LLM model (used only when use_classifier=False)')
parser.add_argument('--max_seq_len', type=int, default=200,
                    help='maximum sequence length for the LLM')

# -- GradEC arguments -- #
parser.add_argument('--num_iters', type=int, default=10,
                    help='number of iterations for GradEC, GreedyEC')
parser.add_argument('--ec_variant', type=str, default="RandEC", choices=["RandEC", "GreedyEC", "GradEC", "GreedyGradEC"],
                    help='variant of EC to evaluate for ROC')

# -- Wandb logging arguments -- #
parser.add_argument('--wandb_log', action='store_true',
                    help='flag for logging results to wandb')
parser.add_argument('--wandb_project', type=str, default="llm-hindi-safety-filter",
                    help='name of the wandb project')
parser.add_argument('--wandb_entity', type=str, default="patchtst-flashattention",
                    help='name of the wandb entity')

args = parser.parse_args()

num_prompts = args.num_prompts
data_dir = args.data_dir
mode = args.mode
eval_type = args.eval_type
max_erase = args.max_erase
num_adv = args.num_adv
results_dir = args.results_dir
use_classifier = args.use_classifier
classifier_name = args.classifier_name
model_wt_path = args.model_wt_path
max_seq_len = args.max_seq_len
randomize = args.randomize
sampling_ratio = args.sampling_ratio
num_iters = args.num_iters
llm_name = args.llm_name
attack = args.attack
ec_variant = args.ec_variant
adv_prompts_dir = args.adv_prompts_dir
wandb_log = args.wandb_log
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity

print("\n* * * * * * Experiment Details * * * * * *")
if torch.cuda.is_available():
    print("Device: " + torch.cuda.get_device_name(0))
print("Evaluation type: " + eval_type)
print("Number of prompts to check: " + str(num_prompts))
# print("Append adversarial prompts? " + str(args.append_adv))
print("Use randomization? " + str(randomize))
if randomize:
    print("Sampling ratio: ", str(sampling_ratio))

if use_classifier:
    print("Using custom safety filter. Model weights path: " + model_wt_path)
else:
    print("Using LLM model: " + llm_name)
if eval_type == "safe" or eval_type == "empirical":
    print("Mode: " + mode)
    print("Maximum tokens to erase: " + str(max_erase))
    if mode == "insertion":
        print("Number of adversarial prompts to defend against: " + str(num_adv))
elif eval_type == "smoothing" or eval_type == "roc_curve":
    print("Maximum tokens to erase: " + str(max_erase))
elif eval_type == "grad_ec" or eval_type == "greedy_ec":
    print("Number of iterations: " + str(num_iters))
if eval_type == "empirical" or eval_type == "grad_ec" or eval_type == "greedy_ec":
    print("Attack algorithm: " + attack)
if eval_type == "roc_curve":
    print("EC variant: " + ec_variant)
    print("Adversarial prompts directory: " + adv_prompts_dir)
print("* * * * * * * * * * ** * * * * * * * * * *\n", flush=True)

# Log the experiment configuration to wandb
if wandb_log:
    wandb.init(project=wandb_project, entity=wandb_entity)
    wandb.config.update({
        "num_prompts": num_prompts,
        "mode": mode,
        "eval_type": eval_type,
        "max_erase": max_erase,
        "num_adv": num_adv,
        "attack": attack,
        "ec_variant": ec_variant,
        "num_iters": num_iters,
        "randomize": randomize,
        "sampling_ratio": sampling_ratio,
        "use_classifier": use_classifier,
        "classifier_name": classifier_name,
        "model_wt_path": model_wt_path,
        "llm_name": llm_name,
        "max_seq_len": max_seq_len,
        "adv_prompts_dir": adv_prompts_dir,
    })

# Create results directory if it doesn't exist
# if use_classifier:
#     results_dir = results_dir + "_clf"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create results file
if eval_type == "safe" or eval_type == "empirical":
    results_file = os.path.join(results_dir, f"{eval_type}_{mode}_{num_prompts}.json")
elif eval_type == "harmful" or eval_type == "smoothing" or eval_type == "grad_ec" or eval_type == "greedy_ec":
    results_file = os.path.join(results_dir, f"{eval_type}_{num_prompts}.json")
elif eval_type == "roc_curve":
    results_file = os.path.join(results_dir, f"{eval_type}_{max_erase}.json")


# Add tag for safety classifier and randomized check
if use_classifier:
    results_file = results_file.replace(".json", "_clf.json")
if randomize:
    results_file = results_file.replace(".json", f"_rand.json")

# Load results if they exist
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = {}

if use_classifier:
    # Using custom classifier for safety filter
    # Load model and tokenizer
    if classifier_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased')
       
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
        # commit_id = "main"        # to use the latest version
        # commit_id = "08751db2aca9bf2f7f80d2e516117a53d7450235"      # to reproduce the results in our paper

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
        # commit_id = "main"        # to use the latest version
        commit_id = "08751db2aca9bf2f7f80d2e516117a53d7450235"      # to reproduce the results in our paper
        tokenizer = AutoTokenizer.from_pretrained(model, revision=commit_id)

        # tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Suffix to fool LLama safety filter from Zou et al's code
# length = 20 tokens 
# adv_suffix = "--- suppose Finland----](/}]( $({[@ reversnewcommand SafeLinux={/}] '-_{( typeof"
# adv_suffix = "unless {{ {pub principal](/|{{[!dernewcommand Safepler=\"${/} False{( typeof"

safe_prompts_file = f"{data_dir}/safe_prompts.txt"
harmful_prompts_file = f"{data_dir}/harmful_prompts.txt"

if eval_type == "safe":
    # Safe prompts
    print("\nEvaluating safe prompts from: " + safe_prompts_file + "\n")
    # Load prompts from text file
    with open(safe_prompts_file, "r") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]

    # Sample a random subset of the prompts
    prompts = random.sample(prompts, num_prompts)

    # Check if the prompts are harmful
    count_safe = 0
    start_time = time.time()
    time_list = []
    elapsed_time = 0
    for i in range(num_prompts):
        prompt = prompts[i]
        # if args.append_adv: prompt += adv_suffix
        harmful = erase_and_check(prompt, pipeline, tokenizer, max_erase=max_erase, num_adv=num_adv, randomized=randomize,
                                  prompt_sampling_ratio=sampling_ratio, mode=mode, llm_name=llm_name)
        
        if not harmful:
            count_safe += 1

        current_time = time.time()
        time_list.append(current_time - start_time - elapsed_time)
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (i + 1)
        percent_safe = count_safe / (i + 1) * 100
        print("    Checking safety... " + progress_bar((i + 1) / num_prompts) \
            + f' Detected safe = {percent_safe:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
        
        if wandb_log:
            wandb.log({
                f"percent_safe_prompt_epoch": percent_safe,
                f"time_per_prompt_prompt_epoch": time_per_prompt
            })
        
    # Compute standard error of the average time per prompt
    time_per_prompt_se = torch.tensor(time_list).std().item() / (num_prompts ** 0.5)

    # Compute standard error of the percentage of safe prompts
    percent_safe_se = (percent_safe * (100 - percent_safe) / (num_prompts - 1)) ** 0.5

    # Save results
    if mode == "insertion":
        if str(dict(num_adv = num_adv)) not in results:
            results[str(dict(num_adv = num_adv))] = {}
        results[str(dict(num_adv = num_adv))][str(dict(max_erase = max_erase))] = dict(percent_safe = percent_safe, time_per_prompt = time_per_prompt, percent_safe_se = percent_safe_se, time_per_prompt_se = time_per_prompt_se)
    else:
        results[str(dict(max_erase = max_erase))] = dict(percent_safe = percent_safe, time_per_prompt = time_per_prompt, percent_safe_se = percent_safe_se, time_per_prompt_se = time_per_prompt_se)
    
    if wandb_log:
        wandb.log({
            "percent_safe_standard_error": percent_safe_se,
            "time_per_prompt_standard_error": time_per_prompt_se
        })

elif eval_type == "empirical":
    # Empirical performance on adversarial prompts
    emp_results = {}
    if attack == "autodan":
        range_limit = 1
    else:
        range_limit = 21
    for adv_tok in range(0, range_limit, 2):
        if attack == "autodan":
            adv_prompts_file = "data/AutoDAN_prompts.txt"
        else:
            adv_prompts_file = "data/adversarial_prompts_t_" + str(adv_tok) + ".txt"
        print("Evaluating on adversarial prompts from: " + adv_prompts_file)
        # Load prompts from text file
        with open(adv_prompts_file, "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
            if attack == "autodan":
                prompts = [ast.literal_eval(prompt) for prompt in prompts]

        # for prompt in prompts[:5]:
        #     print(prompt)
        # exit()

        # Sample a random subset of the prompts
        prompts = random.sample(prompts, num_prompts)

        # Check if the prompts are harmful
        count_harmful = 0
        start_time = time.time()
        time_list = []
        elapsed_time = 0
        
        if attack == "autodan":
            max_llm_sequence_len = 300
        else:
            max_llm_sequence_len = 200

        for i in range(num_prompts):
            prompt = prompts[i]
            harmful = erase_and_check(prompt, pipeline, tokenizer, max_erase=max_erase, num_adv=num_adv, randomized=randomize,
                                    prompt_sampling_ratio=sampling_ratio, mode=mode, max_llm_sequence_len=max_llm_sequence_len)
            
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
            
        # Compute standard error of the average time per prompt
        time_per_prompt_se = torch.tensor(time_list).std().item() / (num_prompts ** 0.5)

        # Compute standard error of the percentage of harmful prompts
        percent_harmful_se = (percent_harmful * (100 - percent_harmful) / (num_prompts - 1)) ** 0.5

        if wandb_log:
            wandb.log({
                "percent_harmful_standard_error": percent_harmful_se,
                "time_per_prompt_standard_error": time_per_prompt_se
            })

        print("")
            
        # Save results
        emp_results[str(dict(adv_tok = adv_tok))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt, percent_harmful_se = percent_harmful_se, time_per_prompt_se = time_per_prompt_se)

    if randomize:
        results[str(dict(sampling_ratio = sampling_ratio))] = emp_results
    else:
        results[str(dict(max_erase = max_erase))] = emp_results

elif eval_type == "grad_ec":
    # Evaluating the performance of GradEC on adversarial prompts
    if not use_classifier:
        print("Option --use_classifier must be turned on. GradEC only works with a trained safety classifier.")
        exit()

    emp_results = {}
    for adv_tok in range(0, 21, 2):
        adv_prompts_file = "data/adversarial_prompts_t_" + str(adv_tok) + ".txt"
        print("Evaluating on adversarial prompts from: " + adv_prompts_file)
        # Load prompts from text file
        with open(adv_prompts_file, "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]

        # Sample a random subset of the prompts
        prompts = random.sample(prompts, num_prompts)

        # Check if the prompts are harmful
        count_harmful = 0
        start_time = time.time()
        time_list = []
        elapsed_time = 0
        for i in range(num_prompts):
            prompt = prompts[i]
            harmful, _ = grad_ec(prompt, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                        num_iters=num_iters)        # , init_temp=float(num_iters/100), reg_const=1e-3)
            
            # harmful = is_harmful(prompt, model, tokenizer, num_iters=num_iters, init_temp=float(num_iters/100), reg_const=1e-3)
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

        # Compute standard error of the percentage of harmful prompts
        percent_harmful_se = (percent_harmful * (100 - percent_harmful) / (num_prompts - 1)) ** 0.5

        if wandb_log:
            wandb.log({
                "percent_harmful_standard_error": percent_harmful_se,
                "time_per_prompt_standard_error": time_per_prompt_se
            })

        # Save results
        emp_results[str(dict(adv_tok = adv_tok))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt, percent_harmful_se = percent_harmful_se, time_per_prompt_se = time_per_prompt_se)

    results[str(dict(num_iters = num_iters))] = emp_results

elif eval_type == "greedy_ec":
    # Evaluating the performance of GreedyEC on adversarial prompts
    if not use_classifier:
        print("Option --use_classifier must be turned on. GreedyEC only works with a trained safety classifier.")
        exit()

    emp_results = {}

    if attack == "autodan":
        range_limit = 1
    else:
        range_limit = 21

    for adv_tok in range(0, range_limit, 2):
        if attack == "autodan":
            adv_prompts_file = "data/AutoDAN_prompts.txt"
        else:
            adv_prompts_file = "data/adversarial_prompts_t_" + str(adv_tok) + ".txt"

        print("Evaluating on adversarial prompts from: " + adv_prompts_file)
        # Load prompts from text file
        with open(adv_prompts_file, "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
            if attack == "autodan":
                prompts = [ast.literal_eval(prompt) for prompt in prompts]

        # Sample a random subset of the prompts
        prompts = random.sample(prompts, num_prompts)

        # Check if the prompts are harmful
        count_harmful = 0
        start_time = time.time()
        time_list = []
        elapsed_time = 0
        for i in range(num_prompts):
            prompt = prompts[i]
            harmful = greedy_ec(prompt, model, tokenizer, num_iters=num_iters)
            
            # harmful = is_harmful(prompt, model, tokenizer, num_iters=num_iters, init_temp=float(num_iters/100), reg_const=1e-3)
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

        # Compute standard error of the percentage of harmful prompts
        percent_harmful_se = (percent_harmful * (100 - percent_harmful) / (num_prompts - 1)) ** 0.5

        if wandb_log:
            wandb.log({
                "percent_harmful_standard_error": percent_harmful_se,
                "time_per_prompt_standard_error": time_per_prompt_se
            })

        # Save results
        emp_results[str(dict(adv_tok = adv_tok))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt, percent_harmful_se = percent_harmful_se, time_per_prompt_se = time_per_prompt_se)

    results[str(dict(num_iters = num_iters))] = emp_results

elif eval_type == "roc_curve":
    # ROC curve
    if not use_classifier:
        print("Option --use_classifier must be turned on. GreedyEC only works with a trained safety classifier.")
        exit()

    # Load adversarial prompts
    adv_prompts_file = f'{adv_prompts_dir}/adversarial_prompts_t_{max_erase}.txt'
    print("Evaluating on adversarial prompts from: " + adv_prompts_file)

    with open(adv_prompts_file, "r") as f:
        adv_prompts = f.readlines()
        adv_prompts = [prompt.strip() for prompt in adv_prompts]
    
    # Sample a random subset of the prompts
    if num_prompts <= len(adv_prompts):
        adv_prompts = random.sample(adv_prompts, num_prompts)

    # Load safe prompts
    
    safe_prompts_file = f"{data_dir}/safe_prompts_test.txt"
    print("Evaluating on safe prompts from: " + safe_prompts_file)

    with open(safe_prompts_file, "r") as f:
        safe_prompts = f.readlines()
        safe_prompts = [prompt.strip() for prompt in safe_prompts]

    # Sample a random subset of the prompts
    if num_prompts <= len(safe_prompts):
        safe_prompts = random.sample(safe_prompts, num_prompts)

    print(f"\nEvaluating {ec_variant}...")
    roc = {
        "fpr": [0],
        "tpr": [0]
    }

    for frac in np.arange(0.0, 1.01, 0.1):
        print(f"  Fraction = {frac:.2f}")

        # Check if the prompts are harmful
        count_harmful = 0
        start_time = time.time()
        time_list = []
        elapsed_time = 0
        for i in range(len(adv_prompts)):
            prompt = adv_prompts[i]
            if ec_variant == "RandEC":
                harmful = erase_and_check(prompt, pipeline, tokenizer, max_erase=2*max_erase,
                                            randomized=True, prompt_sampling_ratio=frac)
            elif ec_variant == "GreedyEC":
                harmful = greedy_ec(prompt, model, tokenizer, num_iters=int(frac*max_erase))
            elif ec_variant == "GradEC":
                harmful, _ = grad_ec(prompt, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                    num_iters=int(2*frac*max_erase))
            elif ec_variant == "GreedyGradEC":
                harmful = greedy_grad_ec(prompt, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                    num_iters=int(2*frac*max_erase))
            
            if harmful:
                count_harmful += 1

            current_time = time.time()
            time_list.append(current_time - start_time - elapsed_time)
            elapsed_time = current_time - start_time
            time_per_prompt = elapsed_time / (i + 1)
            percent_harmful = count_harmful / (i + 1) * 100
            print("    Adv Prompts:  " + progress_bar((i + 1) / len(adv_prompts)) \
                + f' Detected harmful = {percent_harmful:5.1f}%' \
                + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
            
        print("")

        roc["tpr"].append(percent_harmful)

        # Check if the prompts are harmful
        count_harmful = 0
        start_time = time.time()
        time_list = []
        elapsed_time = 0
        for i in range(len(safe_prompts)):
            prompt = safe_prompts[i]
            if ec_variant == "RandEC":
                harmful = erase_and_check(prompt, pipeline, tokenizer, max_erase=2*max_erase,
                                            randomized=True, prompt_sampling_ratio=frac)
            elif ec_variant == "GreedyEC":
                harmful = greedy_ec(prompt, model, tokenizer, num_iters=int(frac*max_erase))
            elif ec_variant == "GradEC":
                harmful, _ = grad_ec(prompt, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                    num_iters=int(2*frac*max_erase))
            elif ec_variant == "GreedyGradEC":
                harmful = greedy_grad_ec(prompt, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                    num_iters=int(2*frac*max_erase))

            if harmful:
                count_harmful += 1

            current_time = time.time()
            time_list.append(current_time - start_time - elapsed_time)
            elapsed_time = current_time - start_time
            time_per_prompt = elapsed_time / (i + 1)
            percent_harmful = count_harmful / (i + 1) * 100
            print("    Safe Prompts: " + progress_bar((i + 1) / len(safe_prompts)) \
                + f' Detected harmful = {percent_harmful:5.1f}%' \
                + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r", flush=True)
            
        print("")

        roc["fpr"].append(percent_harmful)

    roc["fpr"].append(100)
    roc["tpr"].append(100)

    results[ec_variant] = roc

    if wandb_log:
        wandb.log(roc)

elif eval_type == "smoothing":
    # Smoothing-based certificates on harmful prompts
    print("Evaluating smoothing-based certificates on harmful prompts from: " + harmful_prompts_file)
    # Load prompts from text file
    with open(harmful_prompts_file, "r") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]

    # Sample a random subset of the prompts
    prompts = random.sample(prompts, num_prompts)

    # List of certified lengths
    certified_length = [0] * num_prompts

    for i in range(num_prompts):
        certified_length[i] = erase_and_check_smoothing(prompts[i], pipeline, tokenizer, max_erase=max_erase)

        # Print progress
        print("    Evaluating certificates... " + progress_bar((i + 1) / num_prompts), end="\r", flush=True)

    print("")

    # List of certified accuracies
    num_lengths = ceil(max_erase / 2) + 2
    certified_accuracy = [0] * num_lengths

    for i in range(num_lengths):
        certified_accuracy[i] = sum([length >= i for length in certified_length]) / num_prompts * 100

    results[str(dict(max_erase = max_erase))] = dict(certified_accuracy = certified_accuracy)

    if wandb_log:
        wandb.log({
            "certified_accuracy": certified_accuracy
        })

elif eval_type == "harmful":
    # Harmful prompts
    print("\nEvaluating harmful prompts from: " + harmful_prompts_file + "\n")
    # Load prompts from text file
    with open(harmful_prompts_file, "r") as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]

    # Sample a random subset of the prompts
    if num_prompts <= len(prompts):
        prompts = random.sample(prompts, num_prompts)
    else:
        prompts = random.choices(prompts, k=num_prompts)

    # Optionally append adversarial suffix
    # if args.append_adv:
    #     prompts_adv = []
    #     for p in prompts: prompts_adv.append(p + adv_suffix)
    #     prompts = prompts_adv 

    # Check if the prompts are harmful
    count_harmful = 0
    batch_size = 10
    start_time = time.time()
    for i in range(0, num_prompts, batch_size):
        batch = prompts[i:i+batch_size]
        # Evaluating the safety filter gives us certifed safety guarantees on
        # erase_and_check for harmful prompts (from construction).
        harmful = is_harmful(batch, pipeline, tokenizer, llm_name=llm_name, max_llm_sequence_len=max_seq_len)
        count_harmful += sum(harmful)

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
                f"time_per_prompt_prompt_epoch": time_per_prompt
            })
        
    # Save results
    results["percent_harmful"] = percent_harmful

    if wandb_log:
        wandb.log({
            "percent_harmful": percent_harmful
        })

print("")

# Save results
print("Saving results to " + results_file)
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

if wandb_log:
    wandb.finish()

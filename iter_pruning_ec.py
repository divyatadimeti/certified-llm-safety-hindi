import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse, time, json, os

from defenses import progress_bar

def iterative_pruning_ec(prompt, model, tokenizer, num_iters=20, threshold=0.5, output_subsequence=False):
    '''
    Implements an iterative pruning version of the erase-and-check procedure.
    At each step, the algorithm removes the token with the least impact on the harmful class score.
    '''

    device = model.device
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    # Create a copy of tokens for gradient calculation
    tokens_float = tokens.clone().float().detach()
    tokens_float.requires_grad = True

    for _ in range(num_iters):
        if tokens.shape[1] < 3:
            if output_subsequence:
                return False, prompt
            return False

        # Calculate gradients
        model_output = model(tokens)
        loss = -model_output[0][0, 0]  # Negative of harmful class score
        loss.backward()

        # Find token with smallest gradient magnitude
        token_gradients = tokens_float.grad.abs().sum(dim=-1).squeeze()
        min_grad_index = token_gradients[1:-1].argmin().item() + 1  # Skip [CLS] and [SEP]

        # Erase token with smallest gradient
        tokens = torch.cat((tokens[:, :min_grad_index], tokens[:, min_grad_index+1:]), dim=1)
        tokens_float = torch.cat((tokens_float[:, :min_grad_index], tokens_float[:, min_grad_index+1:]), dim=1)

        # Zero gradients for the next iteration
        tokens_float.grad.zero_()

        # Check if harmful
        model_output = model(tokens)
        output_softmax = torch.nn.functional.softmax(model_output[0], dim=1)
        if output_softmax[0, 0].item() > threshold:
            if output_subsequence:
                return True, tokenizer.decode(tokens, skip_special_tokens=True)
            return True

    if output_subsequence:
        return False, prompt
    return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
    parser.add_argument('--prompts_file', type=str, default='data/adversarial_prompts_t_20.txt', help='File containing prompts')
    parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations')
    parser.add_argument('--model_wt_path', type=str, default='models/distilbert_suffix.pt', help='Path to model weights')
    parser.add_argument('--results_file', type=str, default='results/iterative_pruning_results.json', help='File to store results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for harmful classification')
    parser.add_argument('--output_subsequence', action='store_true', help='Output the harmful subsequence if found')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model_wt_path = args.model_wt_path
    
    model.load_state_dict(torch.load(model_wt_path, map_location=device))
    model.to(device)
    model.eval()

    prompts_file = args.prompts_file
    num_iters = args.num_iters
    results_file = args.results_file
    threshold = args.threshold
    output_subsequence = args.output_subsequence

    print('\n* * * * * Experiment Parameters * * * * *')
    print('Prompts file: ' + prompts_file)
    print('Number of iterations: ' + str(num_iters))
    print('Model weights: ' + model_wt_path)
    print('* * * * * * * * * * * * * * * * * * * * *\n')

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")
    list_of_bools = []
    start_time = time.time()

    # Open results file and load previous results JSON as a dictionary
    results_dict = {}
    # Create results file if it does not exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
    with open(results_file, 'r') as f:
        results_dict = json.load(f)

    for num_done, input_prompt in enumerate(prompts):
        decision = iterative_pruning_ec(input_prompt, model, tokenizer, num_iters, threshold, output_subsequence)
        list_of_bools.append(decision)

        percent_harmful = (sum(list_of_bools) / len(list_of_bools)) * 100.
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (num_done + 1)

        print("  Checking safety... " + progress_bar((num_done + 1) / len(prompts)) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r")
        
    print("")

    # Save results
    results_dict[str(dict(num_iters = num_iters))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
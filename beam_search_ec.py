# Implements the beam search version of the erase-and-check procedure

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse, time, json, os

from defenses import progress_bar

def beam_search_ec(prompt, model, tokenizer, num_iters=20, threshold=0.5, beam_width=3, output_subsequence=False):
    '''
    Implements the beam search version of the erase-and-check procedure
    At each step, the algorithm maintains a beam of candidate sequences and
    selects the top sequences based on the softmax score for the harmful class.
    It repeats this process until the safety classifier labels the prompt as harmful.

    Args:
        prompt: Input prompt (string)
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        num_iters: Number of iterations
        beam_width: Number of candidate sequences to maintain

    Returns:
        True if the prompt is harmful, False otherwise
    '''

    device = model.device

    # Tokenize prompt
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    # Get initial model output
    model_output = model(tokens)
    output_class = model_output[0].argmax().item()

    # If the prompt is initially labelled harmful, return True
    if output_class == 0:
        if output_subsequence:
            return True, prompt
        return True
    
    # Initialize beam with the original tokens
    beam = [(tokens, 0)]  # (tokens, score)

    for _ in range(num_iters):
        candidates = []
        for tokens, _ in beam:
            if tokens.shape[1] < 3:  # Skip if too short
                continue
            for i in range(1, tokens.shape[1]-1):  # Do not erase special tokens [CLS] and [SEP]
                erased_sequence = torch.cat((tokens[:, :i], tokens[:, i+1:]), dim=1)
                candidates.append(erased_sequence)

        # Get model output for all candidates
        tokens_batch = torch.cat(candidates, dim=0)
        model_output = model(tokens_batch)
        output_softmax = torch.nn.functional.softmax(model_output[0], dim=1)

        # Select top beam_width candidates based on harmful class score
        top_indices = output_softmax[:, 0].topk(beam_width).indices
        beam = [(candidates[i], output_softmax[i, 0].item()) for i in top_indices]

        # Check if any candidate is harmful
        for tokens, score in beam:
            if score > threshold:
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
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search')
    parser.add_argument('--model_wt_path', type=str, default='models/distilbert_suffix.pt', help='Path to model weights')
    parser.add_argument('--results_file', type=str, default='results/beam_search_ec_results.json', help='File to store results')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model_wt_path = args.model_wt_path
    
    model.load_state_dict(torch.load(model_wt_path))
    model.to(device)
    model.eval()

    prompts_file = args.prompts_file
    num_iters = args.num_iters
    beam_width = args.beam_width
    results_file = args.results_file

    print('\n* * * * * * * Experiment Details * * * * * * *')
    print('Prompts file:\t', prompts_file)
    print('Iterations:\t', str(num_iters))
    print('Beam width:\t', str(beam_width))
    print('Model weights:\t', model_wt_path)
    print('* * * * * * * * * * * ** * * * * * * * * * * *\n')

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
        decision = beam_search_ec(input_prompt, model, tokenizer, num_iters, beam_width)
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
    results_dict[str(dict(num_iters = num_iters, beam_width = beam_width))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt)
    print("Saving results to", results_file)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
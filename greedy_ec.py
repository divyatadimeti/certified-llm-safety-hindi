# Code taken from the official implementation of Erase-and-Check method:
# https://github.com/aounon/certified-llm-safety

import torch

def greedy_ec(prompt, model, tokenizer, num_iters=20, threshold=0.5, output_subsequence=False):
    '''
    Implements the greedy version of the erase-and-check procedure
    At each step, the algorithm erases a token from the current prompt
    such that the softmax score for the harmful class is maximized. It repeats
    this process until the safety classifier labels the prompt as harmful. If the
    model does not label the prompt as harmful, the algorithm declares
    the original prompt as safe.

    Args:
        prompt: Input prompt (string)
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        num_iters: Number of iterations

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
    # Class 1: Safe, Class 0: Harmful
    if output_class == 0:
        if output_subsequence:
            return True, prompt
        return True
    
    # Otherwise, erase tokens until the prompt is labelled harmful
    for _ in range(num_iters):
        # Create batch of inputs with one token erased
        tokens_batch = []
        if tokens.shape[1] < 3:     # Loop below will not work
            if output_subsequence:
                return False, prompt
            return False
        for i in range(1, tokens.shape[1]-1):       # Do not erase special tokens [CLS] and [SEP]
            erased_sequence = torch.cat((tokens[:, :i], tokens[:, i+1:]), dim=1)
            tokens_batch.append(erased_sequence)
        tokens_batch = torch.cat(tokens_batch, dim=0)

        # Get model output
        model_output = model(tokens_batch)
        output_softmax = torch.nn.functional.softmax(model_output[0], dim=1)

        # Find index that maximizes the harmful class softmax score
        max_index = output_softmax[:, 0].argmax().item()
        max_score = output_softmax[max_index, 0].item()

        # If the max score is greater than threshold, return True
        if max_score > threshold:
            if output_subsequence:
                return True, tokenizer.decode(tokens_batch[max_index], skip_special_tokens=True)
            return True
        
        # Otherwise, update tokens
        tokens = tokens_batch[max_index].unsqueeze(0)

    if output_subsequence:
        return False, prompt
    return False
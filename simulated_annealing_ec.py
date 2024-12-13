import torch
import random
import math

def simulated_annealing_ec(prompt, model, tokenizer, num_iters=20, threshold=0.5, output_subsequence=False, initial_temp=1.8, cooling_rate=0.6):
    '''
    Implements the simulated annealing version of the erase-and-check procedure.
    At each step, the algorithm erases a token from the current prompt and evaluates
    the harmfulness. It accepts changes that increase harmfulness and occasionally
    accepts less optimal changes to escape local optima.

    Args:
        prompt: Input prompt (string)
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        num_iters: Number of iterations
        initial_temp: Initial temperature for simulated annealing
        cooling_rate: Rate at which the temperature decreases

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

    # Initialize temperature
    temperature = initial_temp

    # Simulated annealing loop
    for _ in range(num_iters):
        if tokens.shape[1] < 3:  # Ensure there are enough tokens to erase
            if output_subsequence:
                return False, prompt
            return False

        # Randomly select a token to erase
        i = random.randint(1, tokens.shape[1] - 2)  # Avoid special tokens [CLS] and [SEP]
        erased_sequence = torch.cat((tokens[:, :i], tokens[:, i+1:]), dim=1)

        # Get model output for the erased sequence
        model_output = model(erased_sequence)
        output_softmax = torch.nn.functional.softmax(model_output[0], dim=1)
        harmful_score = output_softmax[0, 0].item()

        # Calculate acceptance probability
        current_score = torch.nn.functional.softmax(model(tokens)[0], dim=1)[0, 0].item()
        acceptance_probability = math.exp((harmful_score - current_score) / temperature)

        # Decide whether to accept the new sequence
        if harmful_score > current_score or random.random() < acceptance_probability:
            tokens = erased_sequence

        # Cool down the temperature
        temperature *= cooling_rate

        # Check if the current sequence is harmful
        if harmful_score > threshold:
            if output_subsequence:
                return True, tokenizer.decode(tokens, skip_special_tokens=True)
            return True

    if output_subsequence:
        return False, prompt
    return False
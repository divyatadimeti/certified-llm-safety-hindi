import torch

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
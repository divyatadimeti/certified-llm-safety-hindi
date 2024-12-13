import numpy as np
import random

def get_phrases(mode, prefsuffix_file, insertion_file):
    """
    Loads the phrases from the text file and returns them.

    Args:
        mode: The mode of the prompt augmentation.
        prefsuffix_file: The path to the text file containing the phrases for the prefix and suffix modes.
        insertion_file: The path to the text file containing the phrases for the insertion mode.
    Returns:
        A list of phrases.
    """
    # Check the mode
    print(f"Evaluating with mode: {mode}")
    if mode == "prefix":
        phrases_file = prefsuffix_file
    elif mode == "suffix":
        phrases_file = prefsuffix_file
    elif mode == "insertion":
        phrases_file = insertion_file
    else:
        raise ValueError("Invalid mode: " + mode)
    
    # Load phrases from text file
    with open(phrases_file, "r") as f:
        phrases = f.readlines()
        phrases = [phrase.strip() for phrase in phrases]
    return phrases

def get_safe_prompts(safe_prompts_file):
    """
    Loads the safe prompts from the text file and returns them.
    
    Args:
        safe_prompts_file: The path to the text file containing the safe prompts.
    Returns:
        A list of safe prompts.
    """
    with open(safe_prompts_file, "r") as f:
        file_prompts = f.readlines()
        safe_prompts = []
        for p in file_prompts:
            prompt = p.strip()
            safe_prompts.append(prompt)
    return safe_prompts

def get_harmful_prompts(harmful_prompts_file, mode, phrases, hidden_harmful):
    """
    Loads the harmful prompts from the text file and returns them. Applies the phrase-augmentated and encapsulation-augmented 
    mode to the prompts if specified.

    Args:
        harmful_prompts_file: The path to the text file containing the harmful prompts.
        mode: The mode of the prompt augmentation.
        phrases: The list of phrases to use for the prompt augmentation.
        hidden_harmful: Whether the harmful prompts are hidden or not.
    Returns:
        A list of harmful prompts.
    """
    # Load prompts from text file
    with open(harmful_prompts_file, "r") as f:
        file_prompts = f.readlines()
        harmful_prompts = []
        if mode != "base":
            phrase = np.random.choice(phrases)
        for p in file_prompts:
            prompt = p.strip()
            if hidden_harmful:
                safe_prompts = get_safe_prompts()
                sel_safe_prompts = random.sample(safe_prompts, 2)
                padded_prompt = sel_safe_prompts[0] + prompt + sel_safe_prompts[1]
                harmful_prompts.append(padded_prompt)
            else:
                if mode != "base":
                    if mode == "prefix":
                        prompt = phrase + prompt
                    elif mode == "suffix":
                        prompt = prompt + phrase
                    elif mode == "insertion":
                        insert_idx = np.random.randint(0, len(prompt))
                        prompt = prompt[:insert_idx] + phrase + prompt[insert_idx:]
                harmful_prompts.append(prompt)
    return harmful_prompts


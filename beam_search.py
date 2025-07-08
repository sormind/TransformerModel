from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from train import get_model, get_ds, run_validation, causal_mask

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = get_weights_file_path(config, f"19")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

def beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Perform beam search decoding on the given model to generate a sequence.

    Parameters:
    model (nn.Module): The trained model used for decoding.
    beam_size (int): The number of sequences to consider at each step.
    source (torch.Tensor): The input sequence tensor.
    source_mask (torch.Tensor): The mask for the input sequence to ignore padding tokens.
    tokenizer_src (Tokenizer): The tokenizer used for the source language.
    tokenizer_tgt (Tokenizer): The tokenizer used for the target language.
    max_len (int): The maximum length of the generated sequence.
    device (torch.device): The device (CPU or GPU) for computation.

    Returns:
    torch.Tensor: The decoded sequence tensor with the best score.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:
        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:
            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Perform greedy decoding on the given model to generate a sequence.

    Parameters:
    model (nn.Module): The trained model used for decoding.
    source (torch.Tensor): The input sequence tensor.
    source_mask (torch.Tensor): The mask for the input sequence to ignore padding tokens.
    tokenizer_src (Tokenizer): The tokenizer used for the source language.
    tokenizer_tgt (Tokenizer): The tokenizer used for the target language.
    max_len (int): The maximum length of the generated sequence.
    device (torch.device): The device (CPU or GPU) for computation.

    Returns:
    torch.Tensor: The decoded sequence tensor.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
    """
    Run validation on the given model using the specified dataset.

    Parameters:
    model (nn.Module): The trained model to be validated.
    validation_ds (DataLoader): The validation dataset loader.
    tokenizer_src (Tokenizer): The tokenizer used for the source language.
    tokenizer_tgt (Tokenizer): The tokenizer used for the target language.
    max_len (int): The maximum length of the generated sequence.
    device (torch.device): The device (CPU or GPU) for computation.
    print_msg (function): The function used to print messages to the console.
    num_examples (int, optional): The number of examples to validate. Defaults to 2.

    Returns:
    None
    """
    model.eval()
    count = 0

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out_greedy = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len,
                                             device)
            model_out_beam = beam_search_decode(model, 3, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt,
                                                max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_beam = tokenizer_tgt.decode(model_out_beam.detach().cpu().numpy())
            model_out_text_greedy = tokenizer_tgt.decode(model_out_greedy.detach().cpu().numpy())

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>20}{source_text}")
            print_msg(f"{f'TARGET: ':>20}{target_text}")
            print_msg(f"{f'PREDICTED GREEDY: ':>20}{model_out_text_greedy}")
            print_msg(f"{f'PREDICTED BEAM: ':>20}{model_out_text_beam}")

            if count == num_examples:
                print_msg('-' * console_width)
                break


run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, 20, device, print_msg=print, num_examples=2)


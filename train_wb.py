from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb
import torchmetrics


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """
    Function to decode the model's output using a greedy approach.

    Args:
    - model: The Transformer model.
    - source: The input sequence from the source language.
    - source_mask: The attention mask for the source sequence.
    - tokenizer_src: Tokenizer for the source language.
    - tokenizer_tgt: Tokenizer for the target language.
    - max_len: The maximum length for the target sequence.
    - device: The device to run the model on.

    Returns:
    - The decoded output sequence.
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Compute the encoder output once and reuse it for each decoding step
    encoder_output = model.encode(source, source_mask)
    # Start the decoder input with the SOS token
    decoder_input = (
        torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    )

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Create mask for the target sequence
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(
            device
        )

        # Obtain output from the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Determine the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2
):
    """
    Perform validation and log metrics using wandb.

    Args:
    - model: The Transformer model.
    - validation_ds: Validation dataset.
    - tokenizer_src: Tokenizer for the source language.
    - tokenizer_tgt: Tokenizer for the target language.
    - max_len: The maximum length for the target sequence.
    - device: The device to run the model on.
    - print_msg: Function to print messages.
    - global_step: Global step count.
    - num_examples: Number of examples to validate.
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # Attempt to obtain the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # Default console width if unable to obtain
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # Ensure batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Display source, target, and predicted text
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    # Evaluate metrics for model performance
    # Calculate the character error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({"validation/cer": cer, "global_step": global_step})

    # Calculate the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({"validation/wer": wer, "global_step": global_step})

    # Calculate the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({"validation/BLEU": bleu, "global_step": global_step})


def get_all_sentences(ds, lang):
    """
    Generator to yield sentences from a dataset.

    Args:
    - ds: The dataset.
    - lang: Language to extract.

    Yields:
    - Sentence in the specified language.
    """
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Create or load a tokenizer for a given language.

    Args:
    - config: Configuration dictionary.
    - ds: The dataset to build the tokenizer from.
    - lang: The language for which the tokenizer is built.

    Returns:
    - tokenizer: The loaded or created tokenizer.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # Build a new tokenizer if one does not exist
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Prepare datasets for training and validation.

    Args:
    - config: Configuration dictionary.

    Returns:
    - train_dataloader: DataLoader for the training dataset.
    - val_dataloader: DataLoader for the validation dataset.
    - tokenizer_src: Tokenizer for the source language.
    - tokenizer_tgt: Tokenizer for the target language.
    """
    # Load dataset and split into train and validation sets
    ds_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )

    # Build tokenizers for source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Split dataset: 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # Create DataLoaders for training and validation datasets
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def train(config, print_msg):
    """
    Train the Transformer model.

    Args:
    - config: Configuration dictionary.
    - print_msg: Function to print messages.
    """
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Transformer model
    model = build_transformer(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
        N=config.get("num_layers", 6),
        h=config.get("num_heads", 8),
        dropout=config.get("dropout_p", 0.1),
    ).to(device)

    # Load model weights if the weights file exists
    model_weights_path = Path(get_weights_file_path(config))
    if model_weights_path.is_file():
        model.load_state_dict(torch.load(model_weights_path))
        print_msg("Model weights loaded.")
    else:
        print_msg("No pre-trained weights found.")

    # Define loss criterion
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"))

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: (
            config["d_model"] ** (-0.5) * min(
                (step + 1) ** (-0.5), (step + 1) * config["warmup_steps"] ** (-1.5)
            )
        ),
    )

    # Initialize wandb for logging
    wandb.init(project="translation-transformers", config=config)
    wandb.watch(model, log="all")

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(config["num_epochs"]):
        print_msg(f"Epoch {epoch + 1} of {config['num_epochs']}")

        # Track average loss for the epoch
        losses = []

        # Iterate through batches
        for batch in tqdm(train_dataloader):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            # Get predictions from the model
            prediction = model(
                encoder_input, encoder_mask, decoder_input, decoder_mask
            )
            prediction = prediction.permute(0, 2, 1)

            # Compute loss
            loss = loss_fn(prediction, label)
            losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log loss and global step to wandb
            wandb.log({"train/loss": loss.item(), "global_step": global_step})
            global_step += 1

        print_msg(f"Epoch {epoch + 1} loss: {sum(losses) / len(losses)}")

        # Run validation
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            print_msg,
            global_step,
        )

        # Save model weights after each epoch
        torch.save(model.state_dict(), get_weights_file_path(config))


if __name__ == "__main__":
    config = get_config()
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)

    warnings.warn = lambda *args, **kwargs: None  # Suppress warnings

    def print_msg(msg):
        print(msg)

    train(config, print_msg)


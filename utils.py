import torch
import os


def build_vocabulary(labels: list[str], save_voceb_dir, blank_char="-"):
    vocab = dict()
    index = 1
    unique_letters = set()

    # Iterate through all label in labels
    for label in labels:
        # Split label into lowercase letter
        label_letters = list(label.lower())
        unique_letters.update(label_letters)

    for letter in sorted(unique_letters):
        vocab[letter] = index
        index += 1

    vocab[blank_char] = index

    vocab_size = len(vocab)

    # Write vocabulary into a text file
    with open(os.path.join(save_voceb_dir, "ocr_vocab.txt"), "w") as f:
        for letter, index in vocab.items():
            f.write(f"{index}\t{letter}\n")

    print(f"Build vocabulary successfully. Vocab size = {vocab_size}")

    return vocab, vocab_size


def label_encoder(label, char_to_idx: dict, max_label_len, show_result=False):
    # Encode the label using the provided character-to-index mapping
    encoded_label = torch.tensor(
        [char_to_idx[char] for char in label], dtype=torch.int32
    )
    label_len = len(encoded_label)

    # Pad the encoded label to match max_label_len
    padded_label = torch.nn.functional.pad(
        encoded_label,  # Input tensor
        (0, max_label_len - label_len),  # (padding_left, padding_right)
        value=0,  # Padding value
    )
    label_len = torch.tensor(label_len, dtype=torch.int32)

    if show_result:
        print(f"Encode: From {label} to {padded_label}")

    return padded_label, label_len


def encoder(labels, char_to_idx: dict, max_label_len, show_result=False):
    encoded_result = []
    for label in labels:
        encoded_result.append(
            label_encoder(label, char_to_idx, max_label_len, show_result)
        )

    return encoded_result


def label_decoder(encoded_sequence, idx_to_char, blank_char="-", show_result=False):
    decode_label = []
    prev_char = None
    for token in encoded_sequence:
        if token != 0:  # Ignore padding
            char = idx_to_char[token]
            # Append the character if it's not a blank or the same as the previous character
            if char != blank_char:
                if char != prev_char or prev_char == blank_char:
                    decode_label.append(char)
            prev_char = char  # Update previous character

    decode_sequence = "".join(decode_label)
    if show_result:
        print(f"Decoded: From {encoded_sequence} to {decode_sequence}")

    return decode_sequence


def decode(encoded_sequences, idx_to_char, blank_char="-", show_result=False):
    decode_sequences = []
    for sequence in encoded_sequences:
        decode_sequences.append(
            label_decoder(sequence, idx_to_char, blank_char, show_result)
        )

    if show_result:
        print(f"Decoded: From {encoded_sequences} to {decode_sequences}")

    return decode_sequences

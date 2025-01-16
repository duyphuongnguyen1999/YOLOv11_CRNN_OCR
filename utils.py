import torch


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


def label_decoder(
    label: torch.tensor, idx_to_char: dict, blank_char="-", show_result=False
):
    decode_sequence = []


def decode(encoded_sequences, idx_to_char, blank_char="-", show_result=False):
    decode_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None  # To track the previous character

        for token in seq:
            if token != 0:  # Ignore padding (token = 0)
                char = idx_to_char[token]
                # Append the character if it's not a blank or the same as the previous character
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char  # Update previous character

        decode_sequences.append("".join(decoded_label))

    if show_result:
        print(f"Decoded: From {encoded_sequences} to {decode_sequences}")

    return decode_sequences

import time
import torch
import torch.nn as nn
import timm
from data_processor import DataProcessor
from config import DataProcessorConfig, RecognitionConfig


class CRNN(nn.Module):
    def __init__(
        self,
        config: RecognitionConfig,
        vocab_size,
    ):
        """
        Constructor for the CRNN model.

        :param config: Model configuration containing parameters like the number of hidden layers,
                       dropout probability, number of layers to unfreeze.
        :param vocab_size: The size of the vocabulary (output classes).
        """
        self.config = config
        super(CRNN, self).__init__()
        # Create a ResNet152 backbone with 1 input channel (grayscale image) and pretrained weights
        backbone = timm.create_model("resnet152", in_chans=1, pretrained=True)

        # Remove the last two layers (fully connected layers) from the ResNet model
        modules = list(backbone.children())[:-2]

        # Add an Adaptive Average Pooling layer to adjust the output size from ResNet
        modules.append(nn.AdaptiveAvgPool2d((1, None)))

        # Define the backbone model with the modified layers
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers of the backbone based on the configuration
        for parameter in self.backbone[-self.config.unfreeze_layers :].parameters():
            parameter.requires_grad = True

        # A fully connected layer to map ResNet features to the next space
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512),  # Map from 2048 (ResNet feature size) to 512
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(self.config.dropout_prob),  # Dropout to avoid overfitting
        )

        # GRU layer to process sequential data
        self.gru = nn.GRU(
            512,  # Input size to GRU
            self.config.hidden_size,  # Hidden state size
            self.config.n_layers,  # Number of GRU layers
            bidirectional=True,  # Bidirectional GRU (both forward and backward)
            batch_first=True,  # Batch dimension is the first dimension
            dropout=(
                self.config.dropout_prob if self.config.n_layers > 1 else 0
            ),  # Dropout if multiple GRU layers
        )

        # Layer normalization to normalize the output from the GRU
        self.layer_norm = nn.LayerNorm(
            self.config.hidden_size * 2
        )  # *2 for bidirectional GRU

        # Output layer to map from hidden state space to the vocabulary size
        self.out = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, vocab_size),  # Map to vocab size
            nn.LogSoftmax(dim=2),  # Softmax over the second dimension (for CTC loss)
        )

    # @torch.autocast(device_type="cuda")
    def forward(self, x: torch.Tensor):
        """
        Forward method for the CRNN model.

        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Model output tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Extract features using the ResNet backbone
        x = self.backbone(x)

        # Permute the dimensions to match GRU input requirements
        x = x.permute(
            0, 3, 1, 2
        )  # (batch_size, width, height, channels) -> (batch_size, channels, width, height)
        x = x.view(
            x.size(0), x.size(1), -1
        )  # Flatten the height and width into a single dimension

        # Pass through the fully connected layer
        x = self.mapSeq(x)

        # Process the sequence through the GRU layer
        x, _ = self.gru(x)

        # Apply layer normalization to the GRU output
        x = self.layer_norm(x)

        # Get the final output (logits for CTC loss)
        x = self.out(x)

        # Permute dimensions for CTC
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, vocab_size)

        return x  # Output shape: (seq_len, batch_size, vocab_size)


def evaluate(model, dataloader, criterion, device):
    # Set the model to evaluation mode (disables dropout and batch normalization)
    model.eval()

    # List to store the loss for each batch
    losses = []

    # Disable gradient computation to save memory and computation in evaluation mode
    with torch.no_grad():
        for inputs, labels, labels_len in dataloader:
            # Move inputs, labels, and labels_len to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            # # Mixed precision support
            # with torch.amp.autocast("cuda"):
            # Forward pass
            # outputs = model(inputs)

            # # Determine the length of logits
            # logits_lens = torch.full(
            #     size=(inputs.size(0),),  # Batch size
            #     fill_value=outputs.size(1),  # Output sequence length
            #     dtype=torch.long,
            #     device=device,
            # )

            # # Compute the loss
            # loss = criterion(outputs, labels, logits_lens, labels_len)

            # Forward pass
            outputs = model(inputs)

            # Determine the length of logits
            logits_lens = torch.full(
                size=(inputs.size(0),),  # Batch size
                fill_value=outputs.size(1),  # Output sequence length
                dtype=torch.long,
                device=device,
            )

            # Compute the loss
            loss = criterion(outputs, labels, logits_lens, labels_len)

            # Append the scalar loss value to the losses list
            losses.append(loss.item())

    # Compute the average loss across all batches
    loss = sum(losses) / len(losses)

    return loss


def fit(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs
):
    # Lists to store training and validation losses for each epoch
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start = time.time()  # Record the start time of the epoch

        # List to store training losses for each batch in the current epoch
        batch_train_losses = []

        # Set the model to training mode (enables dropout and batch normalization)
        model.train()

        # Use scaler for mixed precision training
        # scaler = torch.amp.GradScaler()

        # Iterate over each batch in the training data
        for idx, (inputs, labels, labels_len) in enumerate(train_loader):
            # Move inputs, labels, and labels_len to the specified device (e.g., GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            # Zero the gradients of the model's parameters before the forward-backward pass
            optimizer.zero_grad()

            # Enable autocast
            # with torch.amp.autocast("cuda"):
            #     # Perform a forward pass through the model to get predictions
            #     outputs = model(inputs)

            #     # Create a tensor for the length of the logits for each batch
            #     logits_lens = torch.full(
            #         size=(outputs.size(1),),  # Output sequence length
            #         fill_value=outputs.size(0),  # Batch size
            #         dtype=torch.long,
            #     ).to(device)

            #     # Compute the loss using the criterion (e.g., CTC loss for sequence prediction)
            #     loss = criterion(outputs, labels, logits_lens, labels_len)

            # Perform a forward pass through the model to get predictions
            outputs = model(inputs)

            # Create a tensor for the length of the logits for each batch
            logits_lens = torch.full(
                size=(outputs.size(1),),  # Output sequence length
                fill_value=outputs.size(0),  # Batch size
                dtype=torch.long,
            ).to(device)

            # Compute the loss using the criterion (e.g., CTC loss for sequence prediction)
            loss = criterion(outputs, labels, logits_lens, labels_len)

            # # Backpropagate the loss to compute gradients
            # scaler.scale(loss).backward()

            # # Clip the gradients to prevent exploding gradients (gradient clipping threshold = 5)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # # Update the model's parameters using the optimizer
            # scaler.step(optimizer)
            # scaler.update()

            # Backpropagate the loss to compute gradients
            loss.backward()

            # Clip the gradients to prevent exploding gradients (gradient clipping threshold = 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # Update the model's parameters using the optimizer
            optimizer.step()

            # Append the scalar loss value for the current batch to the list
            batch_train_losses.append(loss.item())

        # Compute the average training loss for the epoch
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        # Evaluate the model on the validation set and compute the validation loss
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Print training and validation loss for the current epoch
        print(
            f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}"
            f"\t\t Time: {time.time()- start:.2f} seconds"
        )

        # Update the learning rate scheduler
        scheduler.step()

    # Return the training and validation losses for all epochs
    return train_losses, val_losses


if __name__ == "__main__":
    # Initialize the configuration for the data processor
    data_processor_config = DataProcessorConfig()

    # Instantiate the data processor with the given configuration
    # Setting 'run_yolo_data_processor=False' assumes YOLO data processing is skipped
    processor = DataProcessor(
        config=data_processor_config, run_yolo_data_processor=False
    )

    # Initialize the configuration for the CRNN model
    recognition_config = RecognitionConfig()

    # Create an instance of the CRNN model using the recognition configuration
    # Pass the size of the vocabulary from the OCR data processor to the model
    model = CRNN(recognition_config, processor.ocr_data_processor.ocr_vocab_size)

    # Define the CTC loss function for sequence alignment
    # 'blank' is set to the index of the blank token in the vocabulary
    criterion = nn.CTCLoss(
        blank=processor.ocr_data_processor.char_to_idx_vocab["-"],  # Blank token index
        zero_infinity=True,  # Avoid infinite loss values
        reduction="mean",  # Average loss over the batch
    )

    # Set up the Adam optimizer with learning rate and weight decay from the configuration
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=recognition_config.lr,
        weight_decay=recognition_config.weight_decay,
    )

    # Define a learning rate scheduler to adjust the learning rate at specific intervals
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=recognition_config.scheduler_step_size,  # Step size for LR adjustment
        gamma=0.1,  # Factor by which the learning rate is reduced
    )

    # Train the model and track training and validation losses
    train_losses, val_losses = fit(
        model=model,  # The CRNN model
        train_loader=processor.ocr_data_processor.train_loader,  # Training data loader
        val_loader=processor.ocr_data_processor.val_loader,  # Validation data loader
        criterion=criterion,  # Loss function
        optimizer=optimizer,  # Optimizer
        scheduler=scheduler,  # Learning rate scheduler
        device=recognition_config.device,  # Device to run the computations (CPU/GPU)
        epochs=recognition_config.epochs,  # Number of training epochs
    )

    # Evaluate the model on the validation set and compute the loss
    val_loss = evaluate(
        model=model,
        dataloader=processor.ocr_data_processor.val_loader,
        criterion=criterion,
        device=recognition_config.device,
    )

    # Evaluate the model on the test set and compute the loss
    test_loss = evaluate(
        model=model,
        dataloader=processor.ocr_data_processor.test_loader,
        criterion=criterion,
        device=recognition_config.device,
    )

    # Save the trained model's weights to the specified path
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        recognition_config.save_model_path,
    )

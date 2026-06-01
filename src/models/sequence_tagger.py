"""Configurable sequence tagger used by the architecture ablations."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class CharCNNEncoder(nn.Module):
    """Extract one fixed-size morphological representation for each token."""

    def __init__(
        self,
        num_chars: int,
        char_embedding_dim: int,
        out_channels: int,
        kernel_size: int,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("CharCNN kernel_size must be odd to preserve word length")

        self.char_embedding = nn.Embedding(
            num_embeddings=num_chars,
            embedding_dim=char_embedding_dim,
            padding_idx=padding_idx,
        )

        self.conv = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``[batch, sequence, characters]`` into token features.
        """
        batch_size, seq_len, max_word_len = char_ids.shape

        x = char_ids.reshape(batch_size * seq_len, max_word_len)
        flat_mask = char_mask.reshape(batch_size * seq_len, max_word_len)

        x = self.char_embedding(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.activation(x)

        x = x.masked_fill(
            ~flat_mask.unsqueeze(1),
            torch.finfo(x.dtype).min,
        )
        x = torch.max(x, dim=2).values

        # Fully padded tokens have no valid character to pool.
        has_chars = flat_mask.any(dim=1, keepdim=True)
        x = torch.where(has_chars, x, torch.zeros_like(x))
        x = x.reshape(batch_size, seq_len, -1)

        return x


class SequenceTagger(nn.Module):
    """Compose word embeddings, optional CharCNN, BiLSTM and optional CRF.

    The flags ``use_charcnn`` and ``use_crf`` make the same implementation
    support the A0, A1 and A2 architecture ablations.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_chars: int,
        num_labels: int,
        config: dict,
        pad_token_id: int = 0,
        pad_char_id: int = 0,
    ):
        super().__init__()

        model_cfg = config.get("model", config)

        self.num_labels = num_labels
        self.use_charcnn = model_cfg.get("use_charcnn", True)
        self.use_crf = model_cfg.get("use_crf", True)

        word_embedding_dim = embedding_matrix.shape[1]
        freeze_word_embeddings = model_cfg.get("freeze_word_embeddings", False)

        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=freeze_word_embeddings,
            padding_idx=pad_token_id,
        )

        self.word_dropout = nn.Dropout(model_cfg.get("word_dropout", 0.05))

        lstm_input_dim = word_embedding_dim

        if self.use_charcnn:
            char_embedding_dim = model_cfg["char_embedding_dim"]
            char_cnn_filters = model_cfg["char_cnn_filters"]
            char_cnn_kernel_size = model_cfg["char_cnn_kernel_size"]
            char_dropout = model_cfg.get("char_dropout", 0.1)

            self.char_encoder = CharCNNEncoder(
                num_chars=num_chars,
                char_embedding_dim=char_embedding_dim,
                out_channels=char_cnn_filters,
                kernel_size=char_cnn_kernel_size,
                padding_idx=pad_char_id,
                dropout=char_dropout,
            )

            lstm_input_dim += char_cnn_filters
        else:
            self.char_encoder = None

        lstm_hidden_dim = model_cfg["lstm_hidden_dim"]
        lstm_num_layers = model_cfg.get("lstm_num_layers", 1)
        lstm_dropout = model_cfg.get("lstm_dropout", 0.0)
        bidirectional = model_cfg.get("bidirectional", True)

        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim

        self.classifier_dropout = nn.Dropout(model_cfg.get("classifier_dropout", 0.3))
        self.emission_layer = nn.Linear(lstm_output_dim, num_labels)

        self.use_auxiliary_ce_loss = model_cfg.get("use_auxiliary_ce_loss", False)
        self.auxiliary_ce_weight = model_cfg.get("auxiliary_ce_weight", 0.1)
        self.o_loss_weight = model_cfg.get("o_loss_weight", 1.0)
        self.entity_loss_weight = model_cfg.get("entity_loss_weight", 1.0)

        if self.use_crf:
            self.crf = CRF(num_tags=num_labels, batch_first=True)
        else:
            self.crf = None

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _compute_emissions(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one unnormalized label score per token.
        """
        word_repr = self.word_embedding(token_ids)
        word_repr = self.word_dropout(word_repr)

        if self.use_charcnn:
            char_repr = self.char_encoder(char_ids, char_mask)
            x = torch.cat([word_repr, char_repr], dim=-1)
        else:
            x = word_repr

        lengths = mask.sum(dim=1).detach().cpu()
        packed_x = pack_padded_sequence(
            x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.bilstm(packed_x)
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=token_ids.size(1),
        )
        lstm_out = self.classifier_dropout(lstm_out)

        emissions = self.emission_layer(lstm_out)
        return emissions

    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Compute the training loss for a padded batch.

        With CRF enabled, the main objective is negative log-likelihood. An
        optional weighted cross-entropy term can be added to emphasize entity
        emissions. Without CRF, weighted cross-entropy is the only objective.
        Label id zero is expected to represent ``O``.
        """
        emissions = self._compute_emissions(token_ids, char_ids, char_mask, mask)
        if labels is not None:
            active_logits = emissions[mask]
            active_labels = labels[mask]

            ce_losses = self.loss_fn(active_logits, active_labels)

            if self.o_loss_weight == 1.0 and self.entity_loss_weight == 1.0:
                weighted_ce_loss = ce_losses.mean()
            else:
                weights = torch.where(
                    active_labels == 0,
                    torch.full_like(active_labels, self.o_loss_weight, dtype=torch.float),
                    torch.full_like(active_labels, self.entity_loss_weight, dtype=torch.float),
                )

                weighted_ce_loss = (ce_losses * weights).mean()

            if self.use_crf:
                log_likelihood = self.crf(
                    emissions=emissions,
                    tags=labels,
                    mask=mask,
                    reduction="mean",
                )

                crf_loss = -log_likelihood

                if self.use_auxiliary_ce_loss:
                    return crf_loss + self.auxiliary_ce_weight * weighted_ce_loss

                return crf_loss

            return weighted_ce_loss

    def decode(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        char_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """Decode variable-length label sequences, excluding token padding."""
        emissions = self._compute_emissions(token_ids, char_ids, char_mask, mask)

        if self.use_crf:
            return self.crf.decode(emissions, mask=mask)

        pred_ids = torch.argmax(emissions, dim=-1)

        predictions = []
        for seq, seq_mask in zip(pred_ids.cpu(), mask.cpu()):
            predictions.append(seq[seq_mask].tolist())

        return predictions

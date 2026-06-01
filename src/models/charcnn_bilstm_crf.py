import torch
import torch.nn as nn
from torchcrf  import CRF


class CharCNNEncoder(nn.Module):
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

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: [batch, seq_len, max_word_len]

        returns:
            char_repr: [batch, seq_len, out_channels]
        """
        batch_size, seq_len, max_word_len = char_ids.shape

        x = char_ids.view(batch_size * seq_len, max_word_len)

        x = self.char_embedding(x)
        x = self.dropout(x)

        # [batch*seq, word_len, char_emb] -> [batch*seq, char_emb, word_len]
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.activation(x)

        # max pooling over characters
        x = torch.max(x, dim=2).values

        x = x.view(batch_size, seq_len, -1)

        return x


class CharCNNBiLSTMCRF(nn.Module):
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

        word_embedding_dim = embedding_matrix.shape[1]
        freeze_word_embeddings = model_cfg.get("freeze_word_embeddings", False)

        char_embedding_dim = model_cfg["char_embedding_dim"]
        char_cnn_filters = model_cfg["char_cnn_filters"]
        char_cnn_kernel_size = model_cfg["char_cnn_kernel_size"]
        char_dropout = model_cfg.get("char_dropout", 0.1)

        lstm_hidden_dim = model_cfg["lstm_hidden_dim"]
        lstm_num_layers = model_cfg.get("lstm_num_layers", 1)
        lstm_dropout = model_cfg.get("lstm_dropout", 0.0)
        bidirectional = model_cfg.get("bidirectional", True)

        classifier_dropout = model_cfg.get("classifier_dropout", 0.3)

        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=freeze_word_embeddings,
            padding_idx=pad_token_id,
        )

        self.word_dropout = nn.Dropout(model_cfg.get("word_dropout", 0.05))

        self.char_encoder = CharCNNEncoder(
            num_chars=num_chars,
            char_embedding_dim=char_embedding_dim,
            out_channels=char_cnn_filters,
            kernel_size=char_cnn_kernel_size,
            padding_idx=pad_char_id,
            dropout=char_dropout,
        )

        lstm_input_dim = word_embedding_dim + char_cnn_filters

        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim

        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.emission_layer = nn.Linear(lstm_output_dim, num_labels)

        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        token_ids: [batch, seq_len]
        char_ids: [batch, seq_len, max_word_len]
        mask: [batch, seq_len]
        labels: [batch, seq_len]

        Se labels è presente:
            ritorna loss

        Se labels è None:
            ritorna predizioni decodificate
        """
        word_repr = self.word_embedding(token_ids)
        word_repr = self.word_dropout(word_repr)

        char_repr = self.char_encoder(char_ids)

        x = torch.cat([word_repr, char_repr], dim=-1)

        lstm_out, _ = self.bilstm(x)
        lstm_out = self.classifier_dropout(lstm_out)

        emissions = self.emission_layer(lstm_out)

        if labels is not None:
            log_likelihood = self.crf(
                emissions=emissions,
                tags=labels,
                mask=mask,
                reduction="mean",
            )
            return -log_likelihood

        predictions = self.crf.decode(
            emissions=emissions,
            mask=mask,
        )

        return predictions

    def decode(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        return self.forward(
            token_ids=token_ids,
            char_ids=char_ids,
            mask=mask,
            labels=None,
        )

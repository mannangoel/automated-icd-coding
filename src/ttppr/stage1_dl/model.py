# src/ttppr/stage1_dl/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class LAAT(nn.Module):
    """
    Label-Wise Attention Network (LAAT) coupled with BioClinical-BERT encoder.
    """
    def __init__(self, model_name: str, num_classes: int, hidden_dim: int = 256):
        super(LAAT, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size
        
        # Bi-directional LSTM layer to capture sequential clinical context
        self.bilstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # LAAT Attention linear projections
        self.num_classes = num_classes
        self.attn_target = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.attn_code = nn.Linear(hidden_dim * 2, num_classes, bias=False)
        
        # Per-label classification heads
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # 1. Contextual Encoding from Transformer
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, encoder_dim]

        # 2. BiLSTM contextual representation
        h, _ = self.bilstm(sequence_output)  # [batch_size, seq_len, 2 * hidden_dim]

        # 3. Label-Wise Attention Mechanism
        # Score each token's relevance to every single ICD class label
        attn_weights = torch.softmax(self.attn_code(torch.tanh(self.attn_target(h))), dim=1) 
        # [batch_size, seq_len, num_classes] -> transpose to [batch_size, num_classes, seq_len]
        attn_weights = attn_weights.transpose(1, 2)

        # Document representation weighted individually per label
        label_repr = torch.bmm(attn_weights, h)  # [batch_size, num_classes, 2 * hidden_dim]

        # 4. Binary classification prediction per code
        logits = self.fc(label_repr)  # [batch_size, num_classes, num_classes]
        logits = torch.diagonal(logits, dim1=1, dim2=2)  # Extract per-class predictions -> [batch_size, num_classes]

        return logits
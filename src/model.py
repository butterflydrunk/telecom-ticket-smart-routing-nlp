"""
Архитектура нейросетевого классификатора текста (TextCNN) на PyTorch.
Оптимизирована для экспорта в ONNX и низколатентного инференса на CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TicketTextCNN(nn.Module):
    """
    Легковесный сверточный классификатор текстовых обращений.
    Преимущество TextCNN над тяжелыми трансформерами:
    - Время инференса < 5 мс на стандартном CPU.
    - Нетребователен к оперативной памяти (образ весит < 5 МБ).
    - Высокая точность локализации ключевых n-грамм ('обрыв кабеля', 'двойное списание').
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_filters: int = 64, num_classes: int = 4, dropout_rate: float = 0.2):
        super(TicketTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Сверточные фильтры для улавливания биграмм и триграмм
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # Переставляем оси для Conv1d: (batch_size, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Применение сверток и глобального пулинга
        feat1 = F.relu(self.conv1(embedded))
        pooled1 = F.adaptive_max_pool1d(feat1, 1).squeeze(2)
        
        feat2 = F.relu(self.conv2(embedded))
        pooled2 = F.adaptive_max_pool1d(feat2, 1).squeeze(2)
        
        # Конкатенация признаков
        combined = torch.cat([pooled1, pooled2], dim=1)
        dropped = self.dropout(combined)
        
        # Выходные логиты
        logits = self.fc(dropped)
        return logits


if __name__ == "__main__":
    model = TicketTextCNN(vocab_size=200, embed_dim=64, num_classes=4)
    dummy_input = torch.randint(0, 100, (2, 24))
    output = model(dummy_input)
    print("Архитектура TextCNN:")
    print(model)
    print(f"Выходной тензор логитов: {output.shape}")

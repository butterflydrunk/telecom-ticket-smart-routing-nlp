"""
Корпус текстовых обращений в техподдержку оператора связи и токенизатор.
Классы:
0: TECH_INCIDENT (обрывы, LOS на роутере, низкая скорость)
1: BILLING_DISPUTE (списания, оплата, баланс)
2: ROAMING_SERVICE (роуминг, зарубежные звонки, пакеты данных)
3: TARIFF_MANAGEMENT (смена тарифа, безлимит, подключение услуг)
"""

import json
import re
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

LABELS_MAP = {
    0: "TECH_INCIDENT",
    1: "BILLING_DISPUTE",
    2: "ROAMING_SERVICE",
    3: "TARIFF_MANAGEMENT"
}

ROUTING_DEPARTMENT_MAP = {
    "TECH_INCIDENT": "Линейно-технический отдел (NOC / Монтажники)",
    "BILLING_DISPUTE": "Финансовый отдел и расчетный биллинг",
    "ROAMING_SERVICE": "Отдел роуминга и международной координации",
    "TARIFF_MANAGEMENT": "Отдел по работе с клиентами и продаж"
}

RAW_CORPUS = [
    # TECH_INCIDENT
    ("Пропал интернет дома, на роутере горит красная лампочка LOS", 0),
    ("Обрыв оптического кабеля в подъезде, соединения нет", 0),
    ("Скорость интернета упала до нуля, страницы не открываются", 0),
    ("Постоянные обрывы связи каждые пять минут, пинг скачет", 0),
    ("Не горит индикатор интернета на модеме после грозы", 0),
    ("Кабель интернета перебит дверью, нужен выезд мастера", 0),
    ("Очень медленно качает, вместо 100 мегабит выдает 2", 0),
    ("Пропал доступ к сети на всех устройствах через Wi-Fi", 0),
    
    # BILLING_DISPUTE
    ("Списали абонентскую плату два раза за один месяц", 1),
    ("Пополнил баланс через онлайн-банкинг, а деньги не дошли", 1),
    ("Заблокировали интернет, хотя на балансе есть средства", 1),
    ("Пришлите детализацию звонков и списаний за прошлый период", 1),
    ("Почему сняли деньги за услугу, которую я не подключал", 1),
    ("Не проходит оплата по номеру лицевого счета через терминал", 1),
    ("Как вернуть ошибочно отправленный платеж на чужой номер", 1),
    ("Баланс ушел в минус при положительном остатке трафика", 1),

    # ROAMING_SERVICE
    ("Нахожусь за границей в Турции, телефон не ловит сеть оператора", 2),
    ("Как включить роуминг перед вылетом в командировку", 2),
    ("Списали гигантскую сумму за мобильный интернет в роуминге", 2),
    ("Какой пакет интернета выгоднее подключить для поездки в Дубай", 2),
    ("Не приходят входящие SMS с кодами подтверждения в роуминге", 2),
    ("Сим-карта заблокировалась при пересечении границы", 2),
    ("Как проверить остаток минут в международном роуминге", 2),
    ("Не удается зарегистрироваться в гостевой сети за рубежом", 2),

    # TARIFF_MANAGEMENT
    ("Хочу перевести тариф на безлимитный высокоскоростной интернет", 3),
    ("Какие условия перехода на корпоративный семейный тариф", 3),
    ("Как отключить дополнительный пакет каналов интерактивного ТВ", 3),
    ("Хочу сменить текущий тарифный план со следующего месяца", 3),
    ("Подскажите стоимость подключения статического IP-адреса", 3),
    ("Как подключить турбо-кнопку для временного увеличения скорости", 3),
    ("Нужна консультация по новым тарифам для юридических лиц", 3),
    ("Хочу заморозить договор и приостановить обслуживание на время отпуска", 3),
]


class TextTokenizer:
    """Легковесный токенизатор с фиксацией словаря для воспроизводимого инференса."""
    def __init__(self, max_vocab: int = 500, max_len: int = 24):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    @staticmethod
    def clean_text(text: str) -> List[str]:
        cleaned = re.sub(r"[^а-яa-z0-9\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 1]

    def fit(self, texts: List[str]):
        word_counts = {}
        for t in texts:
            for w in self.clean_text(t):
                word_counts[w] = word_counts.get(w, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:self.max_vocab - 2]:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, text: str) -> List[int]:
        tokens = [self.word2idx.get(w, 1) for w in self.clean_text(text)]
        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return tokens

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "max_len": self.max_len}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TextTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_len=data["max_len"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        return tok


class TicketDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: TextTokenizer):
        self.encoded = [tokenizer.encode(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

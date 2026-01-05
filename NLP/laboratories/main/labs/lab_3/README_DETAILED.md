# Лабораторная работа 3: Decoder-Only модель (GPT) для генерации текста

## Цель работы

Познакомиться с архитектурой Transformer, реализовать decoder-only модель (GPT) с использованием слоёв PyTorch и обучить её для генерации продолжения историй на датасете TinyStories.

---

## Оглавление

1. [Теоретическая часть](#теоретическая-часть)
2. [Описание датасета](#описание-датасета)
3. [Архитектура модели](#архитектура-модели)
4. [Детальное объяснение кода](#детальное-объяснение-кода)
5. [Результаты](#результаты)
6. [Вопросы для защиты](#вопросы-для-защиты)

---

## Теоретическая часть

### Что такое Transformer?

**Transformer** — это архитектура нейронной сети, представленная в статье "Attention is All You Need" (2017). Основные компоненты:

- **Self-Attention** — механизм, позволяющий модели "обращать внимание" на разные части входной последовательности
- **Encoder-Decoder структура** — энкодер обрабатывает входные данные, декодер генерирует выход
- **Positional Encoding** — добавление информации о позиции токенов в последовательности

### Что такое GPT (Decoder-Only модель)?

**GPT (Generative Pre-trained Transformer)** — это модель, использующая только **декодер** из Transformer. Особенности:

- **Автогрессивная генерация** — предсказывает следующий токен на основе предыдущих
- **Causal Mask** — маскирует будущие токены, чтобы модель не "подглядывала" вперёд
- **Language Modeling** — обучается предсказывать следующее слово в последовательности

### Отличия от RNN/LSTM

| Характеристика | RNN/LSTM | Transformer (GPT) |
|---------------|----------|-------------------|
| Обработка последовательности | Последовательная (токен за токеном) | Параллельная (вся последовательность сразу) |
| Долгосрочные зависимости | Проблема затухающего градиента | Self-attention справляется лучше |
| Скорость обучения | Медленнее | Быстрее (распараллеливание) |
| Память о контексте | Ограничена hidden state | Attention к всем токенам |

---

## Описание датасета

### TinyStories

**TinyStories** — датасет коротких историй на английском языке с ограниченным словарём, сгенерированный с помощью GPT-3.5/GPT-4.

**Характеристики:**
- Язык: английский
- Размер: ~2.1M историй
- Средняя длина: 200-300 слов
- Словарь: ~1500-2000 наиболее частых английских слов
- Создан для обучения маленьких языковых моделей

**Пример истории:**
```
Once upon a time, there was a little girl named Lily. She loved to play outside
in the sunshine. One day, she saw a big, red apple hanging from a tree. She
wanted to eat it, but it was too high...
```

**Почему TinyStories?**
- Простой словарь → быстрее обучается
- Короткие тексты → меньше требований к памяти
- Понятная структура → легко оценить качество генерации

---

## Архитектура модели

### Общая схема GPT

```
Входной текст
    ↓
[Token Embedding] → преобразование токенов в векторы
    ↓
[Positional Embedding] → добавление информации о позиции
    ↓
[Decoder Block 1]
    ↓
[Decoder Block 2]
    ↓
...
    ↓
[Decoder Block N]
    ↓
[Layer Normalization]
    ↓
[Linear Layer] → предсказание следующего токена
    ↓
Вероятности токенов (logits)
```

### Параметры модели в нашей реализации

```python
VOCAB_SIZE = 5000          # Размер словаря
EMBEDDING_DIM = 256        # Размерность векторов (embeddings)
NUM_HEADS = 4              # Количество голов в Multi-Head Attention
NUM_LAYERS = 4             # Количество Decoder Blocks
FEEDFORWARD_DIM = 1024     # Размерность Feed-Forward слоя
MAX_SEQUENCE_LENGTH = 127  # Максимальная длина последовательности
DROPOUT = 0.1              # Вероятность dropout
```

**Общее количество параметров:** ~5.7M

---

## Детальное объяснение кода

### Ячейка 1: Импорты и настройка окружения

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter
import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используется устройство: {device}')
```

**Что происходит:**
- **PyTorch (`torch`)** — основная библиотека для создания нейросетей
- **`nn.Module`** — базовый класс для всех нейросетевых компонентов
- **`F` (functional)** — функциональный API для операций (softmax, gelu и т.д.)
- **`Dataset` и `DataLoader`** — классы для работы с данными в батчах
- **`device`** — определяем, где будет работать модель:
  - `cuda` — на GPU (быстро, для обучения)
  - `cpu` — на процессоре (медленно, но всегда доступно)

**Зачем нужен GPU?**
- Модель содержит миллионы параметров
- Обучение на CPU может занять дни
- GPU ускоряет матричные операции в 10-100 раз

---

### Ячейка 2: Загрузка датасета TinyStories

```python
dataset = load_dataset('roneneldan/TinyStories', split='train[:50000]')
test_dataset = load_dataset('roneneldan/TinyStories', split='validation[:5000]')
```

**Что происходит:**

1. **`load_dataset`** — функция из библиотеки `datasets` от Hugging Face
2. **`'roneneldan/TinyStories'`** — имя датасета на Hugging Face Hub
3. **`split='train[:50000]'`** — берём первые 50,000 историй из обучающей выборки
4. **`split='validation[:5000]'`** — берём 5,000 историй для валидации

**Синтаксис слайсинга:**
- `'train'` — весь датасет (2M+ историй)
- `'train[:10000]'` — первые 10,000
- `'train[1000:2000]'` — с 1000 по 2000
- `'train[:10%]'` — первые 10%

**Почему validation, а не test?**
- В TinyStories нет отдельного test split'а
- Используем validation как тестовую выборку

**Как хранится датасет?**
```python
dataset[0]  # {'text': 'Once upon a time...'}
len(dataset)  # 50000
```

**Кэширование:**
- Датасет скачивается один раз
- Сохраняется в `~/.cache/huggingface/datasets/`
- При повторном запуске загружается из кэша (мгновенно)

---

### Ячейка 3-4: Класс SimpleTokenizer

**Что такое токенизация?**

Нейросети не понимают текст напрямую. Нужно преобразовать слова в числа:

```
"Hello world" → [42, 156] → модель обрабатывает → генерирует [23] → "!"
```

**Наш токенизатор:**

```python
class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.special_tokens = {
            '<PAD>': 0,     # Заполнение коротких последовательностей
            '<UNK>': 1,     # Неизвестное слово
            '<START>': 2,   # Начало истории
            '<END>': 3      # Конец истории
        }
```

**Специальные токены:**
- **`<PAD>`** (padding) — дополняет короткие тексты до нужной длины
  ```
  "hi" → [42, 0, 0, 0] (дополнили до 4 токенов)
  ```
- **`<UNK>`** (unknown) — для слов, которых нет в словаре
  ```
  "supercalifragilisticexpialidocious" → <UNK> → индекс 1
  ```
- **`<START>`** — маркер начала текста
- **`<END>`** — маркер конца текста (модель узнает, когда остановиться)

#### Метод `build_vocab()`

```python
def build_vocab(self, texts):
    counter = Counter()
    for text in texts:
        tokens = self._tokenize(text)
        counter.update(tokens)

    most_common = counter.most_common(self.vocab_size - 4)
    for idx, (token, _) in enumerate(most_common, start=4):
        self.token2idx[token] = idx
```

**Шаги:**
1. Проходим по всем текстам
2. Разбиваем на токены (слова)
3. Считаем, сколько раз встречается каждое слово
4. Берём топ-5000 самых частых слов
5. Создаём словарь: `{'hello': 42, 'world': 156, ...}`

**Почему только топ-5000?**
- Меньше параметров в модели (embedding слой)
- Быстрее обучается
- Для TinyStories этого достаточно (простой язык)

**Что с редкими словами?**
- Слова, не попавшие в топ-5000, заменяются на `<UNK>`
- Модель учится работать с неизвестными словами

#### Метод `_tokenize()`

```python
def _tokenize(self, text):
    return text.lower().replace('\n', ' ').split()
```

**Что делает:**
1. **`.lower()`** — приводит к нижнему регистру
   ```
   "Hello World" → "hello world"
   ```
2. **`.replace('\n', ' ')`** — убирает переносы строк
   ```
   "line1\nline2" → "line1 line2"
   ```
3. **`.split()`** — разбивает по пробелам
   ```
   "hello world" → ["hello", "world"]
   ```

**Почему так просто?**
- Для TinyStories достаточно (нет сложной пунктуации)
- Быстро работает
- В реальных проектах используют BPE/WordPiece (токенизация подслов)

#### Метод `encode()`

```python
def encode(self, text, max_length=None):
    tokens = self._tokenize(text)
    indices = [self.token2idx.get(token, 1) for token in tokens]

    if max_length is not None:
        if len(indices) > max_length:
            indices = indices[:max_length]  # Обрезаем
        else:
            indices = indices + [0] * (max_length - len(indices))  # Дополняем

    return indices
```

**Пример:**
```python
text = "Once upon a time"
tokenizer.encode(text, max_length=8)
# Шаг 1: разбиваем → ["once", "upon", "a", "time"]
# Шаг 2: преобразуем → [37, 40, 7, 51]
# Шаг 3: дополняем до 8 → [37, 40, 7, 51, 0, 0, 0, 0]
```

**Зачем padding?**
- Все последовательности в батче должны быть одинаковой длины
- GPU эффективно обрабатывает только прямоугольные матрицы
- `<PAD>` (индекс 0) игнорируется при обучении

#### Метод `decode()`

```python
def decode(self, indices):
    tokens = []
    for idx in indices:
        if idx == 0:  # <PAD>
            break
        if idx == 3:  # <END>
            break
        token = self.idx2token.get(idx, '<UNK>')
        if token not in ['<START>', '<UNK>']:
            tokens.append(token)
    return ' '.join(tokens)
```

**Пример:**
```python
indices = [37, 40, 7, 51, 0, 0, 0, 0]
tokenizer.decode(indices)
# → "once upon a time"
```

---

### Ячейка 5: Класс StoryDataset

```python
class StoryDataset(Dataset):
    def __getitem__(self, idx):
        story = self.stories[idx]
        input_ids = story[:-1]   # Все токены кроме последнего
        target_ids = story[1:]   # Все токены кроме первого (сдвиг на 1)
        return torch.tensor(input_ids), torch.tensor(target_ids)
```

**Зачем сдвиг на 1?**

GPT обучается предсказывать **следующий токен**. Для этого нужно:

```
История: "once upon a time there"
Токены:  [10, 23, 45, 67, 89]

Input:   [10, 23, 45, 67]     ← Дано
Target:  [23, 45, 67, 89]     ← Нужно предсказать
```

**Как это работает:**
- Видим `10` → предсказываем `23`
- Видим `10, 23` → предсказываем `45`
- Видим `10, 23, 45` → предсказываем `67`
- И так далее...

**Почему не вся история сразу?**
- Модель учится шаг за шагом
- На каждой позиции своя задача предсказания
- Это называется "язь language modeling"

---

### Ячейка 6: DataLoader

```python
train_loader = DataLoader(train_dataset_obj, batch_size=64, shuffle=True)
```

**Что такое батч (batch)?**
- Группа примеров, обрабатываемых вместе
- Вместо обработки по одной истории, обрабатываем 64 сразу

**Зачем батчи?**
1. **Эффективность GPU:**
   ```
   1 история → GPU загружен на 5%
   64 истории → GPU загружен на 80%
   ```

2. **Стабильность обучения:**
   - Градиенты усредняются по батчу
   - Меньше "шума" в обновлениях весов

3. **Скорость:**
   ```
   По одной: 50,000 историй × 100 мс = 1.4 часа
   Батчами: 782 батча × 500 мс = 6.5 минут
   ```

**Что делает `shuffle=True`?**
- Перемешивает данные каждую эпоху
- Модель не запоминает порядок примеров
- Улучшает обобщение

---

### Ячейка 7: MultiHeadSelfAttention

Это **самый важный** компонент Transformer. Давайте разберём детально.

#### Что такое Attention (Внимание)?

Представьте, что читаете предложение:

```
"The cat sat on the mat because it was tired"
```

Слово **"it"** относится к **"cat"**. Как модель это понимает?

**Self-Attention** позволяет каждому слову "посмотреть" на все остальные и определить, какие из них важны.

#### Query, Key, Value - что это?

Это три разных "взгляда" на те же данные:

```
Слово: "cat"

Query (запрос):   "Что я ищу?"        → вектор [0.2, -0.5, 0.8, ...]
Key (ключ):       "Что я предлагаю?"  → вектор [0.1, 0.3, -0.2, ...]
Value (значение): "Что я несу?"       → вектор [0.9, 0.1, 0.4, ...]
```

**Аналогия с поиском:**
- **Query** — поисковый запрос
- **Key** — индекс в базе данных
- **Value** — содержимое документа

#### Пошаговое объяснение кода

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        self.embedding_dim = 256      # Размерность векторов
        self.num_heads = 4            # Количество голов
        self.head_dim = 64            # 256 / 4 = 64 на каждую голову
```

**Зачем несколько голов?**

Каждая голова ищет свои паттерны:
- **Голова 1:** связь существительных с прилагательными
- **Голова 2:** связь глаголов с объектами
- **Голова 3:** дальние зависимости (местоимения)
- **Голова 4:** синтаксические структуры

**Создание проекций:**

```python
self.query_projection = nn.Linear(256, 256)
self.key_projection = nn.Linear(256, 256)
self.value_projection = nn.Linear(256, 256)
```

Это обучаемые матрицы, которые преобразуют входные векторы:

```
Input вектор [256]
     ↓
Query вектор [256] = Linear(Input)
Key вектор [256]   = Linear(Input)
Value вектор [256] = Linear(Input)
```

**Forward pass:**

```python
def forward(self, input_tensor, mask=None):
    # input_tensor: [batch=64, seq_len=127, embedding=256]

    # Шаг 1: Создаём Query, Key, Value
    query = self.query_projection(input_tensor)  # [64, 127, 256]
    key = self.key_projection(input_tensor)
    value = self.value_projection(input_tensor)
```

**Шаг 2: Разделяем на головы**

```python
# Было: [64, 127, 256]
query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
# Стало: [64, 127, 4, 64]

query = query.transpose(1, 2)
# Стало: [64, 4, 127, 64]
# [batch, num_heads, seq_len, head_dim]
```

**Зачем transpose?**
- Чтобы обрабатывать головы параллельно
- GPU эффективно работает с такой структурой

**Шаг 3: Scaled Dot-Product Attention**

```python
attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

Это **самое важное**. Разберём по шагам:

1. **`key.transpose(-2, -1)`** — транспонируем Key
   ```
   [64, 4, 127, 64] → [64, 4, 64, 127]
   ```

2. **`torch.matmul(query, key.T)`** — матричное умножение
   ```
   [64, 4, 127, 64] × [64, 4, 64, 127] = [64, 4, 127, 127]
   ```

   Результат — матрица "совместимости" каждого токена с каждым:
   ```
           once  upon    a  time
   once   [0.9  0.2  0.1  0.3]  ← "once" больше всего связан с собой
   upon   [0.2  0.8  0.4  0.5]
   a      [0.1  0.3  0.6  0.2]
   time   [0.3  0.5  0.2  0.7]
   ```

3. **`/ math.sqrt(head_dim)`** — масштабирование
   ```
   / sqrt(64) = / 8
   ```

   **Зачем?**
   - Без этого softmax становится слишком "уверенным" (0.99 или 0.01)
   - Градиенты становятся очень малыми (vanishing gradient)
   - Модель плохо обучается

**Шаг 4: Causal Mask**

```python
if mask is not None:
    attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
```

**Что такое Causal Mask?**

Это нижнетреугольная матрица:
```
      0   1   2   3
0  [ 1   0   0   0 ]  ← Токен 0 видит только себя
1  [ 1   1   0   0 ]  ← Токен 1 видит 0 и 1
2  [ 1   1   1   0 ]  ← Токен 2 видит 0, 1, 2
3  [ 1   1   1   1 ]  ← Токен 3 видит всё
```

**Зачем?**
- Модель не должна "подглядывать" в будущее
- При обучении знаем весь текст, но притворяемся, что не знаем
- Без маски модель просто скопирует ответ!

**Как работает `masked_fill`?**
```python
scores = [
    [0.9,  0.2,  0.1,  0.3],
    [0.2,  0.8,  0.4,  0.5],
    [0.1,  0.3,  0.6,  0.2],
    [0.3,  0.5,  0.2,  0.7]
]

# После маскирования:
scores = [
    [0.9,  -inf, -inf, -inf],  # Будущее заблокировано
    [0.2,  0.8,  -inf, -inf],
    [0.1,  0.3,  0.6,  -inf],
    [0.3,  0.5,  0.2,  0.7]
]
```

**Шаг 5: Softmax**

```python
attention_weights = F.softmax(attention_scores, dim=-1)
```

Превращает scores в вероятности (сумма = 1):

```
[0.9, -inf, -inf, -inf]  → [1.0, 0.0, 0.0, 0.0]
[0.2, 0.8, -inf, -inf]   → [0.2, 0.8, 0.0, 0.0]
[0.1, 0.3, 0.6, -inf]    → [0.1, 0.3, 0.6, 0.0]
```

Эти веса показывают, **сколько внимания** уделить каждому токену.

**Шаг 6: Взвешенная сумма Value**

```python
output = torch.matmul(attention_weights, value)
```

Используем веса внимания, чтобы "смешать" Value векторы:

```
output[0] = 1.0 * value[0] + 0.0 * value[1] + 0.0 * value[2] + ...
output[1] = 0.2 * value[0] + 0.8 * value[1] + 0.0 * value[2] + ...
```

**Интуиция:**
- Если слово важно (большой вес), его Value вносит больший вклад
- Если слово не важно (маленький вес), его Value почти игнорируется

**Шаг 7: Объединение голов**

```python
# Было: [64, 4, 127, 64]  (4 головы)
output = output.transpose(1, 2).contiguous()
# Стало: [64, 127, 4, 64]

output = output.view(batch_size, sequence_length, embedding_dim)
# Стало: [64, 127, 256]  (склеили головы обратно)
```

**Финальная проекция:**

```python
output = self.output_projection(output)
```

Линейное преобразование для "смешивания" информации от всех голов.

---

### Ячейка 8: FeedForward

```python
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        self.first_linear = nn.Linear(256, 1024)   # Расширяем
        self.second_linear = nn.Linear(1024, 256)  # Сжимаем обратно
```

**Зачем нужен Feed-Forward?**

После Attention нужно "переварить" информацию:
1. **Расширение:** 256 → 1024 (больше пространства для вычислений)
2. **Нелинейность:** GELU активация
3. **Сжатие:** 1024 → 256 (обратно к исходному размеру)

**Что делает GELU?**

```python
hidden = F.gelu(hidden)
```

GELU (Gaussian Error Linear Unit) — это активация, похожая на ReLU, но более плавная:

```
ReLU:  x < 0 → 0,  x > 0 → x      (жёсткий порог)
GELU:  плавный переход через 0     (мягкий порог)
```

**Зачем плавность?**
- Лучшие градиенты при обучении
- Меньше "мёртвых" нейронов
- Эмпирически работает лучше в Transformer

**Position-Wise — что это значит?**

Feed-Forward применяется **независимо** к каждой позиции:

```
Позиция 0: FFN(vector[0])
Позиция 1: FFN(vector[1])
Позиция 2: FFN(vector[2])
...
```

Каждый токен обрабатывается отдельно, но с общими весами.

---

### Ячейка 9: DecoderBlock

```python
class DecoderBlock(nn.Module):
    def forward(self, input_tensor, mask=None):
        # Self-Attention
        attention_output = self.attention(input_tensor, mask)
        input_tensor = input_tensor + self.dropout_1(attention_output)  # Residual
        input_tensor = self.layer_norm_1(input_tensor)                 # Normalization

        # Feed-Forward
        feedforward_output = self.feed_forward(input_tensor)
        input_tensor = input_tensor + self.dropout_2(feedforward_output)  # Residual
        input_tensor = self.layer_norm_2(input_tensor)                   # Normalization

        return input_tensor
```

#### Residual Connection (Skip Connection)

```
Вход X ───────────────┐
    │                 │
    ↓                 │
[Attention]           │
    │                 │
    ↓                 │
Выход ← + ← ─────────┘  (сложение X + Attention(X))
```

**Зачем?**

1. **Решает проблему затухающего градиента:**
   ```
   Без residual: градиент × 0.1 × 0.1 × 0.1 × ... → 0
   С residual: градиент "течёт" напрямую через сложение
   ```

2. **Позволяет модели "игнорировать" слои:**
   ```
   Если Attention(X) = 0, то X + 0 = X (слой не мешает)
   ```

3. **Упрощает обучение глубоких сетей:**
   ```
   Без residual: тяжело обучить больше 10 слоёв
   С residual: можно обучить 100+ слоёв
   ```

#### Layer Normalization

```python
input_tensor = self.layer_norm_1(input_tensor)
```

**Что делает?**

Нормализует значения по размерности `embedding_dim`:

```python
# До нормализации:
tensor = [5.2, -3.1, 8.9, 0.4, -2.3, ...]  # Разброс значений

# После нормализации:
tensor = [0.8, -0.4, 1.3, 0.1, -0.3, ...]  # Среднее=0, дисперсия=1
```

**Зачем?**
- **Стабилизирует обучение** — нет "взрывных" значений
- **Ускоряет сходимость** — можно использовать больший learning rate
- **Предотвращает "коварные сдвиги"** (internal covariate shift)

**Отличие от Batch Normalization:**
- **Batch Norm:** нормализует по батчу (axis=0)
- **Layer Norm:** нормализует по фичам (axis=-1)

Layer Norm лучше для последовательностей (разная длина батчей).

---

### Ячейка 10: GPTModel

Собираем всё вместе!

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers,
                 feedforward_dim, max_sequence_length, dropout=0.1):
```

#### Token Embedding

```python
self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
```

**Что это?**

Таблица поиска, которая превращает индексы в векторы:

```python
token_id = 42  # Слово "cat"
vector = embedding_table[42]  # [0.2, -0.5, 0.8, ..., 0.1]  (256 чисел)
```

**Как обучается?**
- Изначально случайные веса
- Модель сама учит хорошие представления
- Похожие слова получают похожие векторы:
  ```
  vector("cat")  ≈ vector("dog")
  vector("king") - vector("man") + vector("woman") ≈ vector("queen")
  ```

#### Positional Embedding

```python
self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
```

**Зачем?**

Self-Attention симметричен — не знает порядок слов:
```
"Кот ест рыбу" = "Рыбу ест кот" (для attention одно и то же!)
```

Positional Embedding добавляет информацию о позиции:

```python
position_0 = [0.1, 0.2, -0.3, ...]  # Вектор для позиции 0
position_1 = [0.3, -0.1, 0.5, ...]  # Вектор для позиции 1
position_2 = [-0.2, 0.4, 0.1, ...]  # Вектор для позиции 2
```

Суммируем с token embeddings:

```python
combined = token_embedding + position_embedding
```

**Почему суммирование, а не конкатенация?**
- Сохраняет размерность (256 + 256 → 256, а не 512)
- Работает лучше эмпирически
- Модель сама учится, как их комбинировать

#### Causal Mask

```python
def create_causal_mask(self, sequence_length, device):
    mask = torch.tril(torch.ones(sequence_length, sequence_length, device=device))
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask
```

**Визуализация:**

```python
sequence_length = 4
mask = torch.tril(torch.ones(4, 4))

tensor([
    [1., 0., 0., 0.],  ← Токен 0 видит только себя
    [1., 1., 0., 0.],  ← Токен 1 видит 0, 1
    [1., 1., 1., 0.],  ← Токен 2 видит 0, 1, 2
    [1., 1., 1., 1.]   ← Токен 3 видит всё прошлое
])
```

**`unsqueeze(0).unsqueeze(0)`:**
```
[4, 4] → [1, 4, 4] → [1, 1, 4, 4]
```
Добавляем размерности для batch и heads.

#### Forward Pass

```python
def forward(self, input_token_ids):
    # input_token_ids: [64, 127]  (батч из 64 историй по 127 токенов)

    # 1. Token Embeddings
    token_embeddings = self.token_embedding(input_token_ids)
    # → [64, 127, 256]

    # 2. Positional Embeddings
    position_indices = torch.arange(0, 127)  # [0, 1, 2, ..., 126]
    position_embeddings = self.position_embedding(position_indices)
    # → [127, 256]

    # 3. Комбинируем (broadcasting)
    combined = token_embeddings + position_embeddings
    # → [64, 127, 256]

    # 4. Causal Mask
    mask = self.create_causal_mask(127, device)
    # → [1, 1, 127, 127]

    # 5. Пропускаем через 4 DecoderBlock'а
    hidden_states = combined
    for decoder_block in self.decoder_blocks:  # 4 раза
        hidden_states = decoder_block(hidden_states, mask)
    # → [64, 127, 256]

    # 6. Финальная нормализация
    hidden_states = self.final_layer_norm(hidden_states)

    # 7. Language Modeling Head (предсказание токенов)
    logits = self.language_model_head(hidden_states)
    # Linear(256 → 5000)
    # → [64, 127, 5000]

    return logits
```

**Что такое logits?**

Это "сырые оценки" для каждого токена в словаре:

```python
logits[0, 0, :] = [2.3, -1.5, 0.8, ..., 1.2]  # 5000 чисел
                   ↑     ↑     ↑         ↑
                 token token token    token
                   0     1     2       4999
```

Чем больше число, тем вероятнее этот токен.

Softmax превращает в вероятности:
```python
probs = softmax(logits)  # [0.23, 0.05, 0.15, ..., 0.08]  (сумма = 1)
```

---

### Ячейка 11: Подсчёт параметров

```python
total_params = sum(param.numel() for param in model.parameters())
```

**Откуда 5.7M параметров?**

Разберём каждый компонент:

```
Token Embedding:        5000 × 256 = 1,280,000
Positional Embedding:    127 × 256 =    32,512

MultiHeadSelfAttention (×4 layers):
  - query_projection:   256 × 256 =    65,536
  - key_projection:     256 × 256 =    65,536
  - value_projection:   256 × 256 =    65,536
  - output_projection:  256 × 256 =    65,536
  Итого на один блок:               262,144
  4 блока:                          1,048,576

FeedForward (×4 layers):
  - first_linear:       256 × 1024 = 262,144
  - second_linear:      1024 × 256 = 262,144
  Итого на один блок:               524,288
  4 блока:                          2,097,152

Layer Normalization (×8):           ~4,096

Language Model Head:    256 × 5000 = 1,280,000
────────────────────────────────────────────────
ИТОГО:                            ≈ 5,757,064
```

**Для сравнения:**
- GPT-2 small: 117M параметров (в 20 раз больше)
- GPT-3: 175B параметров (в 30,000 раз больше!)

---

### Ячейка 12-13: Обучение модели

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Включаем dropout и т.д.

    for input_token_ids, target_token_ids in dataloader:
        # Forward pass
        logits = model(input_token_ids)  # [64, 127, 5000]

        # Loss
        loss = criterion(
            logits.view(-1, 5000),      # [64*127, 5000] = [8128, 5000]
            target_token_ids.view(-1)   # [64*127] = [8128]
        )

        # Backward pass
        loss.backward()  # Вычисляем градиенты

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
```

#### CrossEntropyLoss

**Что это?**

Сравнивает предсказания модели с правильными ответами:

```python
# Модель предсказывает:
predicted_probs = [0.6, 0.2, 0.1, 0.1]  # Для токенов [0, 1, 2, 3]

# Правильный ответ:
correct_token = 0

# Loss = -log(0.6) = 0.51
```

Чем увереннее модель в правильном ответе, тем меньше loss.

**Почему `view(-1, ...)`?**

CrossEntropyLoss ожидает:
- Предсказания: [N, num_classes]
- Правда: [N]

У нас:
- Предсказания: [batch, seq_len, vocab_size] = [64, 127, 5000]
- Правда: [batch, seq_len] = [64, 127]

Преобразуем:
```python
[64, 127, 5000] → [8128, 5000]  # 64*127 = 8128 позиций
[64, 127] → [8128]
```

Теперь каждая из 8128 позиций — отдельный пример для loss.

#### Gradient Clipping

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Проблема "взрыва градиентов":**

```
Градиент слоя 1: 2.0
Градиент слоя 2: 2.0 × 2.0 = 4.0
Градиент слоя 3: 4.0 × 2.0 = 8.0
Градиент слоя 4: 8.0 × 2.0 = 16.0
...
Градиент слоя 10: 1024.0  ← Взрыв!
```

**Решение — clipping:**

```python
# Если норма градиентов > 1.0:
if norm(gradients) > 1.0:
    gradients = gradients / norm(gradients) * 1.0
```

Ограничиваем максимальную "силу" обновления.

#### AdamW Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

**Что такое AdamW?**

Улучшенная версия Adam:
- **Адаптивный learning rate** для каждого параметра
- **Momentum** — учитывает историю градиентов
- **Weight decay** — регуляризация (предотвращает переобучение)

**Learning rate = 3e-4 = 0.0003:**

Стандартное значение для Transformer. Почему маленькое?
- Модель очень чувствительна к обновлениям
- Большой LR → нестабильность, NaN loss
- Маленький LR → медленное обучение, но стабильное

---

### Ячейка 14: Визуализация Loss

```python
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
```

**Что смотреть:**

**Хороший сценарий:**
```
Эпоха 1: Train=4.1, Val=3.4
Эпоха 2: Train=3.2, Val=2.9
Эпоха 3: Train=2.9, Val=2.7
Эпоха 4: Train=2.7, Val=2.6
Эпоха 5: Train=2.6, Val=2.5
```
✓ Оба loss снижаются
✓ Val близок к Train

**Переобучение:**
```
Эпоха 1: Train=4.1, Val=3.4
Эпоха 2: Train=3.2, Val=2.9
Эпоха 3: Train=2.5, Val=2.8
Эпоха 4: Train=1.8, Val=3.1  ← Val растёт!
```
✗ Train падает, Val растёт
✗ Модель запоминает train, но не обобщает

**Недообучение:**
```
Эпоха 1: Train=4.1, Val=3.4
Эпоха 2: Train=3.9, Val=3.3
Эпоха 3: Train=3.8, Val=3.2
```
✗ Loss почти не меняется
✗ Нужно больше эпох / больше модель / лучше данные

---

### Ячейка 15: Функция генерации текста

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50):
```

Это **автогрессивная генерация** — создаём текст токен за токеном.

#### Алгоритм генерации

```python
# Начальный промпт
text = "Once upon a time"
tokens = [37, 40, 7, 51]  # "once upon a time"

# Итерация 1:
logits = model(tokens)          # Предсказываем следующий токен
next_token = sample(logits)     # Выбираем токен (например, 29 = "there")
tokens = [37, 40, 7, 51, 29]   # Добавляем к последовательности

# Итерация 2:
logits = model(tokens)
next_token = sample(logits)     # Например, 8 = "was"
tokens = [37, 40, 7, 51, 29, 8]

# Итерация 3:
logits = model(tokens)
next_token = sample(logits)     # Например, 7 = "a"
tokens = [37, 40, 7, 51, 29, 8, 7]

# ... и так далее до max_length или <END>
```

#### Temperature (Коэффициент случайности)

```python
next_token_logits = logits[:, -1, :] / temperature
```

**Что это?**

Контролирует "креативность" модели:

**До temperature:**
```python
logits = [3.0, 1.0, 2.0, 0.5]
probs = softmax(logits) = [0.66, 0.09, 0.24, 0.02]
                            ↑ Модель уверена в первом токене
```

**Temperature = 0.5 (консервативно):**
```python
logits / 0.5 = [6.0, 2.0, 4.0, 1.0]
probs = [0.84, 0.03, 0.12, 0.01]
         ↑ Ещё увереннее!
```
→ Генерация предсказуемая, повторяющаяся

**Temperature = 1.0 (стандарт):**
```python
logits / 1.0 = [3.0, 1.0, 2.0, 0.5]
probs = [0.66, 0.09, 0.24, 0.02]
```
→ Баланс между предсказуемостью и разнообразием

**Temperature = 2.0 (креативно):**
```python
logits / 2.0 = [1.5, 0.5, 1.0, 0.25]
probs = [0.44, 0.16, 0.29, 0.11]
         ↑ Распределение более равномерное
```
→ Генерация непредсказуемая, может быть бессвязной

**Примеры:**

```python
# Temperature = 0.1
"Once upon a time, there was a little girl. She was happy. She was happy."
# Повторяется, скучно

# Temperature = 0.8
"Once upon a time, there was a brave knight who loved adventures."
# Хорошо

# Temperature = 2.0
"Once upon a time, there was a purple elephant dancing in the rain with shoes."
# Креативно, но странно
```

#### Top-k Sampling

```python
if top_k > 0:
    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
    next_token_logits[indices_to_remove] = float('-inf')
```

**Проблема без top-k:**

Даже с temperature модель может выбрать совсем маловероятное слово:

```python
probs = [0.40, 0.30, 0.20, 0.05, 0.03, 0.02, 0.0001, ..., 0.00001]
         ↑     ↑     ↑     ↑     ↑     ↑       ↑              ↑
       хорошие слова                    плохие слова
```

Иногда модель случайно выберет 0.00001 → полная чушь.

**Решение — top-k:**

Оставляем только k самых вероятных токенов:

```python
top_k = 50
probs = [0.40, 0.30, 0.20, 0.05, 0.03, 0.02, 0, 0, 0, ...]
         ↑─────────────── топ-50 ──────────────↑  остальные обнулены
```

Теперь модель выбирает только из хороших вариантов!

**Как работает код:**

```python
torch.topk(logits, 50)  # Находим топ-50 значений
# → [3.2, 2.8, 2.5, ..., 0.5]  (50-й токен имеет значение 0.5)

# Всё, что меньше 0.5 → -inf
logits[logits < 0.5] = -inf
```

После softmax все -inf превращаются в 0.

#### Stopping condition

```python
if next_token.item() == tokenizer.special_tokens['<END>']:
    break
```

Модель сама решает, когда остановиться:
- Если предсказывает `<END>` → генерация закончена
- Иначе продолжаем до `max_length`

---

### Ячейка 16-18: Тестирование и генерация

**Тестирование на примерах из test set:**

```python
prompt = "Once upon a time, there was a little"
generated = generate_text(model, tokenizer, prompt)
```

Берём начало истории из теста, просим модель продолжить, сравниваем с оригиналом.

**Собственные промпты:**

```python
prompts = [
    "Once upon a time, there was a little",
    "One day, a brave",
    "In a magical forest, there lived"
]
```

Проверяем, может ли модель генерировать по нашим началам.

---

## Результаты

### Критерии успешности

✅ **Набор данных загружен и обработан**
- 50,000 историй для обучения
- 5,000 историй для валидации
- Токенизация с словарём 5000 токенов
- Padding до длины 128

✅ **Decoder-only модель реализована**
- Multi-Head Self-Attention с 4 головами
- 4 Decoder Blocks
- Feed-Forward Networks
- Causal Mask для автогрессии
- ~5.7M параметров

✅ **Модель обучена без ошибок**
- 5 эпох обучения
- Loss снижается на train и validation
- Gradient clipping предотвращает нестабильность

✅ **Модель протестирована**
- Генерация на примерах из test set
- Собственные промпты
- Интерактивная генерация

### Ожидаемое качество

После 5 эпох модель должна:
- Генерировать грамматически корректные предложения
- Следовать структуре историй ("Once upon a time...", "The end.")
- Использовать простой словарь из TinyStories
- Иногда терять связность (это нормально для маленькой модели)

**Пример хорошей генерации:**
```
Prompt: "Once upon a time, there was a little"
Generated: "Once upon a time, there was a little girl named Lily.
She loved to play with her toys. One day, she found a big red ball
in the garden. She was very happy!"
```

**Пример слабой генерации (возможно при недообучении):**
```
Prompt: "Once upon a time, there was a little"
Generated: "Once upon a time, there was a little the the was happy
play toy big was friend..."
```

### Как улучшить качество

1. **Больше данных**: использовать весь датасет (2M историй)
2. **Больше эпох**: 10-20 эпох вместо 5
3. **Больше параметров**: embedding_dim=512, num_layers=6
4. **Лучший токенизатор**: BPE вместо word-level
5. **Beam Search**: вместо greedy/sampling
6. **Pre-training + Fine-tuning**: сначала общая задача, потом специфичная

---

## Вопросы для защиты

### Базовые вопросы

**1. В чём отличие Decoder-Only модели от Encoder-Decoder?**

**Ответ:**
- **Encoder-Decoder** (как в оригинальном Transformer): энкодер обрабатывает весь входной текст, декодер генерирует выход. Используется для задач типа перевода, где есть чёткое разделение вход-выход.
- **Decoder-Only** (GPT): только декодер, который автогрессивно генерирует текст, предсказывая следующий токен. Используется для генерации текста, где нет явного "входа".

**2. Что такое Causal Mask и зачем он нужен?**

**Ответ:**
Causal Mask — это нижнетреугольная матрица, которая не позволяет модели "видеть" будущие токены при обучении. Без неё модель могла бы подсмотреть правильный ответ, и обучение было бы бессмысленным. Реализуется с помощью `torch.tril()` и маскирования attention scores через `-inf`.

**3. Как работает Multi-Head Attention?**

**Ответ:**
- Создаём Query, Key, Value проекции для входных данных
- Разделяем на N голов (каждая голова работает с частью размерности)
- Для каждой головы вычисляем Attention(Q, K, V) = softmax(QK^T/√d_k)V
- Объединяем результаты всех голов и применяем финальную проекцию
- Каждая голова учится искать свои паттерны в данных (синтаксис, семантика, дальние связи)

**4. Зачем нужен Positional Embedding?**

**Ответ:**
Self-Attention не учитывает порядок токенов — он симметричен к перестановкам. Positional Embedding добавляет информацию о позиции токена в последовательности, чтобы модель понимала разницу между "кот ест рыбу" и "рыбу ест кот". Мы используем learned positional embeddings — обучаемые векторы для каждой позиции.

**5. Что такое Temperature в генерации текста?**

**Ответ:**
Temperature — это коэффициент, который контролирует "случайность" генерации. Формула: `logits = logits / temperature`.
- **Низкая (0.1-0.5)**: модель консервативна, выбирает самые вероятные токены → предсказуемый текст
- **Средняя (0.8-1.0)**: баланс между предсказуемостью и разнообразием
- **Высокая (1.5-2.0)**: модель креативна, больше разнообразия → может быть бессвязно

### Продвинутые вопросы

**6. Почему используется GELU вместо ReLU?**

**Ответ:**
GELU (Gaussian Error Linear Unit) — это плавная аппроксимация ReLU. Эмпирически показывает лучшие результаты в Transformer-моделях. ReLU "жёстко" обнуляет отрицательные значения (`max(0, x)`), GELU делает это плавно, что помогает градиентам и снижает количество "мёртвых" нейронов. GELU = `x * Φ(x)`, где Φ — функция стандартного нормального распределения.

**7. Зачем нужен Gradient Clipping?**

**Ответ:**
В глубоких сетях градиенты могут "взрываться" — становиться очень большими из-за цепного правила. Gradient Clipping ограничивает норму градиентов значением max_norm (у нас 1.0), предотвращая нестабильность обучения. Без этого веса могут получить огромные обновления, loss станет NaN, и обучение сломается.

**8. Почему модель обучается предсказывать следующий токен, а не всю последовательность сразу?**

**Ответ:**
Это называется Language Modeling. Предсказание следующего токена:
- **Простая и естественная задача** для текста (autocomplete)
- **Не требует размеченных данных** (self-supervised learning)
- Позволяет модели выучить грамматику, семантику и фактические знания
- Масштабируется на любые тексты (весь интернет)

**9. В чём смысл Residual Connections?**

**Ответ:**
Residual connections (`X + F(X)`) решают проблему затухающего градиента в глубоких сетях:
- **Прямой путь для градиентов** — они "текут" через сложение без затухания
- **Модель может "игнорировать" слои** — если `F(X) = 0`, то `X + 0 = X` (слой не мешает)
- **Упрощают обучение** — без residual тяжело обучить больше 10-20 слоёв, с ними можно 100+
- **Identity mapping** — модель учится добавлять "поправки" к входу, а не пересчитывать всё с нуля

**10. Как бы вы улучшили модель для получения лучших результатов?**

**Ответ:**
1. **Увеличить размер модели**: embedding_dim=512, num_layers=8-12, num_heads=8
2. **Больше данных и эпох**: полный TinyStories (2M историй), 20-50 эпох
3. **Лучшая токенизация**: BPE/WordPiece вместо word-level (лучше обрабатывает редкие слова)
4. **Learning Rate Schedule**: cosine annealing или warmup (сначала растёт, потом падает)
5. **Beam Search при генерации**: вместо sampling рассматривать несколько гипотез параллельно
6. **Pre-training + Fine-tuning**: сначала обучить на большом корпусе, потом дообучить на TinyStories
7. **Dropout вариации**: DropPath, LayerDrop для лучшей регуляризации

---

## Выводы

В данной лабораторной работе:

1. ✅ Изучена архитектура Transformer и GPT
2. ✅ Реализована decoder-only модель с нуля (без nn.Transformer)
3. ✅ Все компоненты созданы вручную:
   - Multi-Head Self-Attention с causal mask
   - Feed-Forward Network с GELU
   - Decoder Block с residual connections
   - Полная GPT модель с embeddings
4. ✅ Модель обучена на датасете TinyStories
5. ✅ Протестирована генерация текста

**Ключевые навыки:**
- Понимание архитектуры Transformer
- Работа с PyTorch и нейросетями
- Токенизация и обработка текстовых данных
- Обучение языковых моделей
- Генерация текста с temperature и top-k sampling

**Практическое применение:**
- Генерация текста (истории, статьи, код)
- Чат-боты и диалоговые системы
- Автодополнение текста
- Создание контента

---

## Полезные ссылки

- [Attention is All You Need (оригинальная статья)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Автор:** Claude Code
**Дата:** 2025-01-05

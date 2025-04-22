import os
from argparse import Namespace
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 设置参数
args = Namespace(
    # 数据路径
    surname_csv="surnames_with_splits.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage",

    # 模型参数
    char_embedding_size=32,
    rnn_hidden_size=32,

    # 训练参数
    seed=42,
    learning_rate=0.001,
    batch_size=128,
    num_epochs=20,
    early_stopping_criteria=5,

    # 运行时参数
    cuda=torch.cuda.is_available(),
    reload_from_files=False,
)

# 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)


# 词汇表类
class Vocabulary:
    def __init__(self, token_to_idx=None):
        self._token_to_idx = token_to_idx if token_to_idx else {}
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def add_token(self, token):
        if token not in self._token_to_idx:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return self._token_to_idx[token]

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        return self._idx_to_token[index]

    def __len__(self):
        return len(self._token_to_idx)


# 序列词汇表类
class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>", mask_token="<MASK>",
                 begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super().__init__(token_to_idx)
        self.mask_index = self.add_token(mask_token)
        self.unk_index = self.add_token(unk_token)
        self.begin_seq_index = self.add_token(begin_seq_token)
        self.end_seq_index = self.add_token(end_seq_token)

    def lookup_token(self, token):
        return self._token_to_idx.get(token, self.unk_index)


# 向量化器
class SurnameVectorizer:
    def __init__(self, char_vocab, nationality_vocab):
        self.char_vocab = char_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname, max_length):
        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(char) for char in surname)
        indices.append(self.char_vocab.end_seq_index)

        # 输入向量 (从序列)
        from_vector = np.full(max_length, self.char_vocab.mask_index, dtype=np.int64)
        from_vector[:len(indices) - 1] = indices[:-1]

        # 目标向量 (到序列)
        to_vector = np.full(max_length, self.char_vocab.mask_index, dtype=np.int64)
        to_vector[:len(indices) - 1] = indices[1:]

        return from_vector, to_vector

    @classmethod
    def from_dataframe(cls, df):
        char_vocab = SequenceVocabulary()
        nationality_vocab = Vocabulary()

        for _, row in df.iterrows():
            for char in row.surname:
                char_vocab.add_token(char)
            nationality_vocab.add_token(row.nationality)

        return cls(char_vocab, nationality_vocab)


# 数据集类
class SurnameDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer
        self.max_length = max(len(name) for name in df.surname) + 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        from_vec, to_vec = self.vectorizer.vectorize(row.surname, self.max_length)
        return {
            'x_data': torch.from_numpy(from_vec),
            'y_target': torch.from_numpy(to_vec),
            'class_index': self.vectorizer.nationality_vocab.lookup_token(row.nationality)
        }

    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv_path):
        df = pd.read_csv(csv_path)
        train_df = df[df.split == 'train']
        return cls(df, SurnameVectorizer.from_dataframe(train_df))


# 生成模型
class SurnameGenerator(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_in):
        embedded = self.embedding(x_in)
        output, _ = self.rnn(embedded)
        return self.fc(output)


# 训练函数
def train_model():
    # 加载数据
    dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
    vectorizer = dataset.vectorizer

    # 初始化模型
    model = SurnameGenerator(
        args.char_embedding_size,
        len(vectorizer.char_vocab),
        args.rnn_hidden_size,
        vectorizer.char_vocab.mask_index
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vectorizer.char_vocab.mask_index)

    # 训练循环
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0

        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            outputs = model(batch['x_data'].to(args.device))
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                batch['y_target'].view(-1).to(args.device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(dataset):.4f}")

    return model, vectorizer


# 生成样本函数
def generate_samples(model, vectorizer, num_samples=5):
    model.eval()
    samples = []

    for _ in range(num_samples):
        input_seq = [vectorizer.char_vocab.begin_seq_index]
        hidden = None

        while True:
            input_tensor = torch.LongTensor([input_seq[-1]]).unsqueeze(0)
            output, hidden = model.rnn(model.embedding(input_tensor), hidden)
            probs = F.softmax(model.fc(output.squeeze(0)), dim=-1)
            next_char = torch.multinomial(probs, 1).item()

            if next_char == vectorizer.char_vocab.end_seq_index or len(input_seq) >= 20:
                break

            input_seq.append(next_char)

        samples.append(''.join(vectorizer.char_vocab.lookup_index(i) for i in input_seq[1:]))

    return samples


# 主执行
if __name__ == "__main__":
    # 设置设备
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # 训练模型
    print("Training model...")
    model, vectorizer = train_model()

    # 生成样本
    print("\nGenerated samples:")
    samples = generate_samples(model, vectorizer)
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")

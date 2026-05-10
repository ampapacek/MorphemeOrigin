import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data_sentece import DataSentence, Morph
from data_transformers import EmbeddingTransformer
from model import Model
from utils import evaluate


IGNORE_INDEX = -100
PAD_INDEX = 0
UNK_INDEX = 1


@dataclass
class MorphSequenceExample:
    texts: List[str]
    word_texts: List[str]
    morph_type_ids: List[int]
    morph_position_ids: List[int]
    vowel_features: List[List[float]]
    morph_embeddings: Optional[List[List[float]]] = None
    word_embeddings: Optional[List[List[float]]] = None
    target_label_ids: Optional[List[List[int]]] = None
    morph_refs: Optional[List[Morph]] = None


class _MorphSequenceTagger(nn.Module):
    def __init__(
        self,
        *,
        char_vocab_size: int,
        type_vocab_size: int,
        position_vocab_size: int,
        label_count: int,
        char_emb_dim: int,
        char_hidden_dim: int,
        type_emb_dim: int,
        position_emb_dim: int,
        hidden_dim: int,
        dropout: float,
        use_morph_type: bool,
        use_morph_position: bool,
        use_vowel_features: bool,
        use_morph_embedding: bool,
        use_word_embedding: bool,
        external_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.char_hidden_dim = char_hidden_dim
        self.use_morph_type = use_morph_type
        self.use_morph_position = use_morph_position
        self.use_vowel_features = use_vowel_features
        self.use_morph_embedding = use_morph_embedding
        self.use_word_embedding = use_word_embedding

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=PAD_INDEX,
        )
        self.char_gru = nn.GRU(
            input_size=char_emb_dim,
            hidden_size=char_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        if self.use_morph_type:
            self.type_embedding = nn.Embedding(
                num_embeddings=type_vocab_size,
                embedding_dim=type_emb_dim,
                padding_idx=PAD_INDEX,
            )
        else:
            self.type_embedding = None

        if self.use_morph_position:
            self.position_embedding = nn.Embedding(
                num_embeddings=position_vocab_size,
                embedding_dim=position_emb_dim,
                padding_idx=PAD_INDEX,
            )
        else:
            self.position_embedding = None

        morph_dim = 2 * char_hidden_dim
        if self.use_morph_type:
            morph_dim += type_emb_dim
        if self.use_morph_position:
            morph_dim += position_emb_dim
        if self.use_vowel_features:
            morph_dim += 2
        if self.use_morph_embedding:
            morph_dim += external_embedding_dim
        if self.use_word_embedding:
            morph_dim += external_embedding_dim

        self.morph_dropout = nn.Dropout(dropout)
        self.sequence_gru = nn.GRU(
            input_size=morph_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(2 * hidden_dim, label_count)

    def forward(
        self,
        *,
        char_ids: torch.Tensor,
        char_lengths: torch.Tensor,
        morph_type_ids: torch.Tensor,
        morph_position_ids: torch.Tensor,
        word_lengths: torch.Tensor,
        vowel_features: Optional[torch.Tensor] = None,
        morph_embeddings: Optional[torch.Tensor] = None,
        word_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, max_morphs, max_chars = char_ids.shape
        flat_char_ids = char_ids.reshape(batch_size * max_morphs, max_chars)
        flat_char_lengths = char_lengths.reshape(batch_size * max_morphs)

        char_repr = torch.zeros(
            batch_size * max_morphs,
            2 * self.char_hidden_dim,
            device=char_ids.device,
        )
        valid_char_mask = flat_char_lengths > 0

        if torch.any(valid_char_mask):
            valid_char_ids = flat_char_ids[valid_char_mask]
            valid_char_lengths = flat_char_lengths[valid_char_mask].cpu()
            embedded_chars = self.char_embedding(valid_char_ids)
            packed_chars = pack_padded_sequence(
                embedded_chars,
                valid_char_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.char_gru(packed_chars)
            char_repr[valid_char_mask] = torch.cat([hidden[-2], hidden[-1]], dim=1)

        char_repr = char_repr.reshape(batch_size, max_morphs, -1)
        morph_parts = [char_repr]

        if self.use_morph_type and self.type_embedding is not None:
            morph_parts.append(self.type_embedding(morph_type_ids))
        if self.use_morph_position and self.position_embedding is not None:
            morph_parts.append(self.position_embedding(morph_position_ids))
        if self.use_vowel_features and vowel_features is not None:
            morph_parts.append(vowel_features)
        if self.use_morph_embedding and morph_embeddings is not None:
            morph_parts.append(morph_embeddings)
        if self.use_word_embedding and word_embeddings is not None:
            morph_parts.append(word_embeddings)

        morph_repr = torch.cat(morph_parts, dim=-1)
        morph_repr = self.morph_dropout(morph_repr)

        packed_words = pack_padded_sequence(
            morph_repr,
            word_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.sequence_gru(packed_words)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=max_morphs,
        )
        outputs = self.output_dropout(outputs)
        return self.output_layer(outputs)


class TorchGRUClassifier(Model):
    def __init__(
        self,
        name: Optional[str] = None,
        random_state: int = 42,
        lower_case: bool = True,
        min_label_freq: int = 1,
        verbose: bool = True,
        validation_data: Optional[List[DataSentence]] = None,
        char_emb_dim: int = 64,
        char_hidden_dim: int = 64,
        type_emb_dim: int = 16,
        position_emb_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        batch_size: int = 128,
        epochs: int = 25,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        patience: int = 10,
        use_morph_type: bool = True,
        use_morph_position: bool = True,
        use_vowel_start_end_features: bool = True,
        use_morph_embedding: bool = False,
        use_word_embedding: bool = False,
        embedding_dimension: int = 300,
        fasttext_model_path: str = "cc.cs.300.bin",
        multi_label: bool = False,
        sequence_scope: str = "word",
    ) -> None:
        super().__init__(name)
        self.random_state = random_state
        self.lower_case = lower_case
        self.min_label_freq = min_label_freq
        self.verbose = verbose
        self.validation_data = validation_data

        self.char_emb_dim = char_emb_dim
        self.char_hidden_dim = char_hidden_dim
        self.type_emb_dim = type_emb_dim
        self.position_emb_dim = position_emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience

        self.use_morph_type = use_morph_type
        self.use_morph_position = use_morph_position
        self.use_vowel_start_end_features = use_vowel_start_end_features
        self.use_morph_embedding = use_morph_embedding
        self.use_word_embedding = use_word_embedding
        self.embedding_dimension = embedding_dimension
        self.fasttext_model_path = fasttext_model_path
        self.multi_label = multi_label
        self.sequence_scope = sequence_scope

        if self.sequence_scope not in {"word", "sentence"}:
            raise ValueError("sequence_scope must be either 'word' or 'sentence'.")

        if not name:
            name_parts = [f"GRU_{self.sequence_scope}_char{char_hidden_dim}_seq{hidden_dim}"]
            if use_morph_embedding:
                name_parts.append(f"morph_emb{embedding_dimension}")
            if use_word_embedding:
                name_parts.append(f"word_emb{embedding_dimension}")
            if multi_label:
                name_parts.append("multi_label")
            self.name = "_".join(name_parts)

        self.device = torch.device("cpu")
        self.network: Optional[_MorphSequenceTagger] = None
        self._char_to_id: Dict[str, int] = {}
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: List[str] = []
        self._type_to_id: Dict[str, int] = {}
        self._position_to_id: Dict[str, int] = {}
        self.prediction_threshold = 0.5
        self.threshold_candidates = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
        self._fasttext_model = None

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _build_aux_vocabs(self) -> None:
        self._type_to_id = {"<pad>": PAD_INDEX}
        for morph_type in Morph.MorphType:
            self._type_to_id[morph_type.value] = len(self._type_to_id)

        self._position_to_id = {"<pad>": PAD_INDEX}
        for morph_position in Morph.MorphPosition:
            self._position_to_id[morph_position.value] = len(self._position_to_id)

    def _normalize_text(self, text: str) -> str:
        return text.lower() if self.lower_case else text

    def _vowel_features(self, text: str) -> List[float]:
        vowels = set("aeiouyáéěíóůúý")
        if not text:
            return [0.0, 0.0]
        return [
            1.0 if text[0].lower() in vowels else 0.0,
            1.0 if text[-1].lower() in vowels else 0.0,
        ]

    def _get_fasttext_model(self):
        if not (self.use_morph_embedding or self.use_word_embedding):
            return None
        if self._fasttext_model is None:
            loader = EmbeddingTransformer(
                column="text",
                embedding_dim=self.embedding_dimension,
                fasttext_model_path=self.fasttext_model_path,
                verbose=self.verbose,
            )
            self._fasttext_model = loader.get_model()
        return self._fasttext_model

    def _iter_valid_morphs(self, sentences: List[DataSentence]):
        for sentence in sentences:
            for word in sentence.words:
                valid_morphs = []
                for morph in word:
                    if morph.morph_type == Morph.MorphType.UNDEFINED:
                        continue
                    if not morph.text.isalpha():
                        continue
                    valid_morphs.append(morph)
                if valid_morphs:
                    yield valid_morphs

    def _build_label_vocab(self, data: List[DataSentence]) -> None:
        label_counts: Dict[str, int] = {}
        char_set = set()

        for morphs in self._iter_valid_morphs(data):
            for morph in morphs:
                if self.multi_label:
                    for label in morph.etymology:
                        label_counts[label] = label_counts.get(label, 0) + 1
                else:
                    label = ",".join(morph.etymology)
                    if label:
                        label_counts[label] = label_counts.get(label, 0) + 1
                char_set.update(self._normalize_text(morph.text))

        kept_labels = sorted(
            label for label, count in label_counts.items()
            if count >= self.min_label_freq
        )
        if not kept_labels:
            raise ValueError("No training labels remain after frequency filtering.")

        self._label_to_id = {label: idx for idx, label in enumerate(kept_labels)}
        self._id_to_label = kept_labels
        self._char_to_id = {"<pad>": PAD_INDEX, "<unk>": UNK_INDEX}
        for char in sorted(char_set):
            self._char_to_id[char] = len(self._char_to_id)

    def _build_network(self) -> None:
        self.network = _MorphSequenceTagger(
            char_vocab_size=len(self._char_to_id),
            type_vocab_size=len(self._type_to_id),
            position_vocab_size=len(self._position_to_id),
            label_count=len(self._label_to_id),
            char_emb_dim=self.char_emb_dim,
            char_hidden_dim=self.char_hidden_dim,
            type_emb_dim=self.type_emb_dim,
            position_emb_dim=self.position_emb_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            use_morph_type=self.use_morph_type,
            use_morph_position=self.use_morph_position,
            use_vowel_features=self.use_vowel_start_end_features,
            use_morph_embedding=self.use_morph_embedding,
            use_word_embedding=self.use_word_embedding,
            external_embedding_dim=self.embedding_dimension,
        ).to(self.device)

    def _encode_target_labels(self, morph: Morph) -> List[int]:
        if self.multi_label:
            return [
                self._label_to_id[label]
                for label in morph.etymology
                if label in self._label_to_id
            ]
        label = ",".join(morph.etymology)
        if label in self._label_to_id:
            return [self._label_to_id[label]]
        return []

    def _build_sequence_example(
        self,
        morph_entries: List[tuple[Morph, str]],
        include_targets: bool,
        include_refs: bool,
        fasttext_model,
    ) -> Optional[MorphSequenceExample]:
        texts: List[str] = []
        word_texts: List[str] = []
        morph_type_ids: List[int] = []
        morph_position_ids: List[int] = []
        vowel_features: List[List[float]] = []
        morph_embeddings: List[List[float]] = []
        word_embeddings: List[List[float]] = []
        target_label_ids: List[List[int]] = []
        morph_refs: List[Morph] = []

        for morph, source_word_text in morph_entries:
            normalized_text = self._normalize_text(morph.text)
            normalized_word = self._normalize_text(source_word_text)

            texts.append(normalized_text)
            word_texts.append(normalized_word)
            morph_type_ids.append(self._type_to_id[morph.morph_type.value])
            morph_position_ids.append(self._position_to_id[morph.morph_position.value])
            vowel_features.append(self._vowel_features(normalized_text))

            if self.use_morph_embedding and fasttext_model is not None:
                morph_embeddings.append(
                    fasttext_model.get_word_vector(normalized_text).astype(np.float32).tolist()
                )
            if self.use_word_embedding and fasttext_model is not None:
                word_embeddings.append(
                    fasttext_model.get_word_vector(normalized_word).astype(np.float32).tolist()
                )

            if include_targets:
                target_label_ids.append(self._encode_target_labels(morph))
            if include_refs:
                morph_refs.append(morph)

        if not texts:
            return None
        if include_targets and all(not labels for labels in target_label_ids):
            return None

        return MorphSequenceExample(
            texts=texts,
            word_texts=word_texts,
            morph_type_ids=morph_type_ids,
            morph_position_ids=morph_position_ids,
            vowel_features=vowel_features,
            morph_embeddings=morph_embeddings if self.use_morph_embedding else None,
            word_embeddings=word_embeddings if self.use_word_embedding else None,
            target_label_ids=target_label_ids if include_targets else None,
            morph_refs=morph_refs if include_refs else None,
        )

    def _extract_sequences(
        self,
        sentences: List[DataSentence],
        include_targets: bool,
        include_refs: bool,
    ) -> List[MorphSequenceExample]:
        examples: List[MorphSequenceExample] = []
        fasttext_model = self._get_fasttext_model()

        for sentence in sentences:
            if self.sequence_scope == "word":
                for word in sentence.words:
                    morph_entries = []
                    for morph in word:
                        if morph.morph_type == Morph.MorphType.UNDEFINED:
                            continue
                        if not morph.text.isalpha():
                            continue
                        morph_entries.append((morph, word.text))

                    example = self._build_sequence_example(
                        morph_entries,
                        include_targets,
                        include_refs,
                        fasttext_model,
                    )
                    if example is not None:
                        examples.append(example)
            else:
                morph_entries = []
                for word in sentence.words:
                    for morph in word:
                        if morph.morph_type == Morph.MorphType.UNDEFINED:
                            continue
                        if not morph.text.isalpha():
                            continue
                        morph_entries.append((morph, word.text))

                example = self._build_sequence_example(
                    morph_entries,
                    include_targets,
                    include_refs,
                    fasttext_model,
                )
                if example is not None:
                    examples.append(example)

        return examples

    def _collate_examples(
        self,
        examples: List[MorphSequenceExample],
        include_targets: bool,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        max_morphs = max(len(example.texts) for example in examples)
        max_chars = max(max(len(text), 1) for example in examples for text in example.texts)

        char_ids = torch.zeros((batch_size, max_morphs, max_chars), dtype=torch.long)
        char_lengths = torch.zeros((batch_size, max_morphs), dtype=torch.long)
        morph_type_ids = torch.zeros((batch_size, max_morphs), dtype=torch.long)
        morph_position_ids = torch.zeros((batch_size, max_morphs), dtype=torch.long)
        word_lengths = torch.zeros(batch_size, dtype=torch.long)
        vowel_features = torch.zeros((batch_size, max_morphs, 2), dtype=torch.float32)
        morph_embeddings = None
        word_embeddings = None
        if self.use_morph_embedding:
            morph_embeddings = torch.zeros(
                (batch_size, max_morphs, self.embedding_dimension),
                dtype=torch.float32,
            )
        if self.use_word_embedding:
            word_embeddings = torch.zeros(
                (batch_size, max_morphs, self.embedding_dimension),
                dtype=torch.float32,
            )

        target_mask = torch.zeros((batch_size, max_morphs), dtype=torch.bool)
        targets_single = None
        targets_multi = None
        if include_targets:
            if self.multi_label:
                targets_multi = torch.zeros(
                    (batch_size, max_morphs, len(self._label_to_id)),
                    dtype=torch.float32,
                )
            else:
                targets_single = torch.full(
                    (batch_size, max_morphs),
                    IGNORE_INDEX,
                    dtype=torch.long,
                )

        for batch_index, example in enumerate(examples):
            word_lengths[batch_index] = len(example.texts)
            for morph_index, text in enumerate(example.texts):
                encoded_chars = [self._char_to_id.get(char, UNK_INDEX) for char in text] or [UNK_INDEX]
                char_lengths[batch_index, morph_index] = len(encoded_chars)
                char_ids[batch_index, morph_index, :len(encoded_chars)] = torch.tensor(encoded_chars, dtype=torch.long)
                morph_type_ids[batch_index, morph_index] = example.morph_type_ids[morph_index]
                morph_position_ids[batch_index, morph_index] = example.morph_position_ids[morph_index]
                vowel_features[batch_index, morph_index] = torch.tensor(
                    example.vowel_features[morph_index],
                    dtype=torch.float32,
                )

                if self.use_morph_embedding and morph_embeddings is not None and example.morph_embeddings is not None:
                    morph_embeddings[batch_index, morph_index] = torch.tensor(
                        example.morph_embeddings[morph_index],
                        dtype=torch.float32,
                    )
                if self.use_word_embedding and word_embeddings is not None and example.word_embeddings is not None:
                    word_embeddings[batch_index, morph_index] = torch.tensor(
                        example.word_embeddings[morph_index],
                        dtype=torch.float32,
                    )

                if include_targets and example.target_label_ids is not None:
                    label_ids = example.target_label_ids[morph_index]
                    if label_ids:
                        target_mask[batch_index, morph_index] = True
                        if self.multi_label and targets_multi is not None:
                            targets_multi[batch_index, morph_index, label_ids] = 1.0
                        elif targets_single is not None:
                            targets_single[batch_index, morph_index] = label_ids[0]

        batch = {
            "char_ids": char_ids.to(self.device),
            "char_lengths": char_lengths.to(self.device),
            "morph_type_ids": morph_type_ids.to(self.device),
            "morph_position_ids": morph_position_ids.to(self.device),
            "word_lengths": word_lengths.to(self.device),
            "target_mask": target_mask.to(self.device),
        }
        if self.use_vowel_start_end_features:
            batch["vowel_features"] = vowel_features.to(self.device)
        if self.use_morph_embedding and morph_embeddings is not None:
            batch["morph_embeddings"] = morph_embeddings.to(self.device)
        if self.use_word_embedding and word_embeddings is not None:
            batch["word_embeddings"] = word_embeddings.to(self.device)
        if include_targets:
            if self.multi_label and targets_multi is not None:
                batch["targets"] = targets_multi.to(self.device)
            elif targets_single is not None:
                batch["targets"] = targets_single.to(self.device)
        return batch

    def _iterate_batches(self, examples: List[MorphSequenceExample], shuffle: bool):
        indices = list(range(len(examples)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield [examples[index] for index in batch_indices]

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(
            char_ids=batch["char_ids"],
            char_lengths=batch["char_lengths"],
            morph_type_ids=batch["morph_type_ids"],
            morph_position_ids=batch["morph_position_ids"],
            word_lengths=batch["word_lengths"],
            vowel_features=batch.get("vowel_features"),
            morph_embeddings=batch.get("morph_embeddings"),
            word_embeddings=batch.get("word_embeddings"),
        )

    def _predict_sentences(
        self,
        data: List[DataSentence],
        threshold_override: Optional[float] = None,
    ) -> List[DataSentence]:
        if self.network is None:
            raise ValueError("The model has not been trained. Call fit() or load() first.")

        threshold = threshold_override if threshold_override is not None else self.prediction_threshold
        updated_data = copy.deepcopy(data)
        prediction_examples = self._extract_sequences(
            updated_data,
            include_targets=False,
            include_refs=True,
        )

        self.network.eval()
        with torch.no_grad():
            for batch_examples in self._iterate_batches(prediction_examples, shuffle=False):
                batch = self._collate_examples(batch_examples, include_targets=False)
                logits = self._forward_batch(batch)

                if self.multi_label:
                    probabilities = torch.sigmoid(logits).cpu()
                    for batch_index, example in enumerate(batch_examples):
                        if not example.morph_refs:
                            continue
                        for morph_index, morph in enumerate(example.morph_refs):
                            selected_labels = [
                                self._id_to_label[label_index]
                                for label_index, probability in enumerate(probabilities[batch_index, morph_index].tolist())
                                if probability >= threshold
                            ]
                            morph.etymology = selected_labels if selected_labels else ["ces"]
                else:
                    predicted_ids = logits.argmax(dim=-1).cpu()
                    for batch_index, example in enumerate(batch_examples):
                        if not example.morph_refs:
                            continue
                        for morph_index, morph in enumerate(example.morph_refs):
                            predicted_label = self._id_to_label[predicted_ids[batch_index, morph_index].item()]
                            morph.etymology = predicted_label.split(",")

        return updated_data

    def _validation_score(self) -> tuple[float, float]:
        if not self.validation_data:
            return 0.0, self.prediction_threshold

        if not self.multi_label:
            predictions = self._predict_sentences(self.validation_data)
            results = evaluate(
                predictions,
                self.validation_data,
                instance_eval=True,
                micro_eval=False,
                native_borrowed_eval=False,
                group_by_text_eval=False,
                morph_type_eval=False,
            )
            return results["f1score_instance"], self.prediction_threshold

        best_score = float("-inf")
        best_threshold = self.prediction_threshold
        for threshold in self.threshold_candidates:
            predictions = self._predict_sentences(self.validation_data, threshold_override=threshold)
            results = evaluate(
                predictions,
                self.validation_data,
                instance_eval=True,
                micro_eval=False,
                native_borrowed_eval=False,
                group_by_text_eval=False,
                morph_type_eval=False,
            )
            score = results["f1score_instance"]
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return best_score, best_threshold

    def fit(self, data: List[DataSentence]) -> None:
        self._set_seed()
        self._build_aux_vocabs()
        self._build_label_vocab(data)
        self._build_network()

        train_examples = self._extract_sequences(
            data,
            include_targets=True,
            include_refs=False,
        )
        if not train_examples:
            raise ValueError(f"No valid {self.sequence_scope}-level training sequences were created.")

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_score = float("-inf")
        best_threshold = self.prediction_threshold
        best_state = copy.deepcopy(self.network.state_dict())
        epochs_without_improvement = 0

        if self.verbose:
            print(f"Fiting model: {self.name}")
            print(
                f"GRU training data: {len(train_examples)} {self.sequence_scope} sequences, "
                f"{len(self._label_to_id)} labels, {len(self._char_to_id)} chars"
            )

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            total_loss = 0.0
            total_targets = 0

            for batch_examples in self._iterate_batches(train_examples, shuffle=True):
                batch = self._collate_examples(batch_examples, include_targets=True)
                logits = self._forward_batch(batch)
                target_mask = batch["target_mask"]
                if not torch.any(target_mask):
                    continue

                if self.multi_label:
                    targets = batch["targets"]
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[target_mask],
                        targets[target_mask],
                    )
                else:
                    targets = batch["targets"]
                    loss = torch.nn.functional.cross_entropy(
                        logits[target_mask],
                        targets[target_mask],
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()

                valid_target_count = int(target_mask.sum().item())
                total_loss += float(loss.item()) * valid_target_count
                total_targets += valid_target_count

            average_loss = total_loss / total_targets if total_targets else 0.0
            validation_score, validation_threshold = self._validation_score() if self.validation_data else (-average_loss, self.prediction_threshold)

            if self.verbose:
                message = f"Epoch {epoch}/{self.epochs}: train_loss={average_loss:.4f}, val_f1={validation_score:.2f}"
                if self.multi_label:
                    message += f", val_threshold={validation_threshold:.2f}"
                print(message)

            if validation_score > best_score:
                best_score = validation_score
                best_threshold = validation_threshold
                best_state = copy.deepcopy(self.network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.validation_data and epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"Early stopping after epoch {epoch}.")
                    break

        self.prediction_threshold = best_threshold
        self.network.load_state_dict(best_state)

    def predict(self, data: List[DataSentence]) -> List[DataSentence]:
        return self._predict_sentences(data)

    def save(self, filename: str) -> None:
        if self.network is None:
            raise ValueError("No network to save. Train the model first.")

        checkpoint = {
            "name": self.name,
            "random_state": self.random_state,
            "lower_case": self.lower_case,
            "min_label_freq": self.min_label_freq,
            "char_emb_dim": self.char_emb_dim,
            "char_hidden_dim": self.char_hidden_dim,
            "type_emb_dim": self.type_emb_dim,
            "position_emb_dim": self.position_emb_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "use_morph_type": self.use_morph_type,
            "use_morph_position": self.use_morph_position,
            "use_vowels": self.use_vowel_start_end_features,
            "use_morph_embedding": self.use_morph_embedding,
            "use_word_embedding": self.use_word_embedding,
            "embedding_dimension": self.embedding_dimension,
            "fasttext_model_path": self.fasttext_model_path,
            "multi_label": self.multi_label,
            "sequence_scope": self.sequence_scope,
            "prediction_threshold": self.prediction_threshold,
            "char_to_id": self._char_to_id,
            "label_to_id": self._label_to_id,
            "id_to_label": self._id_to_label,
            "type_to_id": self._type_to_id,
            "position_to_id": self._position_to_id,
            "state_dict": self.network.state_dict(),
        }
        torch.save(checkpoint, filename)
        if self.verbose:
            print(f"Model saved to {filename}")

    def load(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)

        self.name = checkpoint["name"]
        self.random_state = checkpoint["random_state"]
        self.lower_case = checkpoint["lower_case"]
        self.min_label_freq = checkpoint["min_label_freq"]
        self.char_emb_dim = checkpoint["char_emb_dim"]
        self.char_hidden_dim = checkpoint["char_hidden_dim"]
        self.type_emb_dim = checkpoint["type_emb_dim"]
        self.position_emb_dim = checkpoint["position_emb_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.dropout = checkpoint["dropout"]
        self.batch_size = checkpoint["batch_size"]
        self.epochs = checkpoint["epochs"]
        self.learning_rate = checkpoint["learning_rate"]
        self.weight_decay = checkpoint["weight_decay"]
        self.patience = checkpoint["patience"]
        self.use_morph_type = checkpoint.get("use_morph_type", self.use_morph_type)
        self.use_morph_position = checkpoint.get("use_morph_position", self.use_morph_position)
        self.use_vowel_start_end_features = checkpoint.get("use_vowels", self.use_vowel_start_end_features)
        self.use_morph_embedding = checkpoint.get("use_morph_embedding", self.use_morph_embedding)
        self.use_word_embedding = checkpoint.get("use_word_embedding", self.use_word_embedding)
        self.embedding_dimension = checkpoint.get("embedding_dimension", self.embedding_dimension)
        self.fasttext_model_path = checkpoint.get("fasttext_model_path", self.fasttext_model_path)
        self.multi_label = checkpoint.get("multi_label", self.multi_label)
        self.sequence_scope = checkpoint.get("sequence_scope", self.sequence_scope)
        self.prediction_threshold = checkpoint.get("prediction_threshold", self.prediction_threshold)
        self._char_to_id = checkpoint["char_to_id"]
        self._label_to_id = checkpoint["label_to_id"]
        self._id_to_label = checkpoint["id_to_label"]
        self._type_to_id = checkpoint["type_to_id"]
        self._position_to_id = checkpoint["position_to_id"]

        self._build_network()
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()

        if self.verbose:
            print(f"Model loaded from {filename}")

import copy
import pickle
from typing import List, Optional, Literal, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer,LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from model import Model
from data_sentece import DataSentence,Word,Morph
from data_transformers import EmbeddingTransformer, VowelStartEndTransformer

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress all ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MorphClassifier(Model):
    """
    A morph-level classification model that predicts etymology for each morph,
    with flexible feature usage (char n-grams, morph type, morph position,
    morph embedding, word embedding) and choice of classifier (SVM, MLP ensemble, or logistic regression).
    
    Can do either single-label classification or multi-label classification 
    (one-vs-rest) based on whether 'multi_label' is True.

    Parameters
    ----------
    name : str
        Custom name for this model instance.

    classifier_type : Literal["svm", "mlp", "lr"]
        Which classifier to use. Supported: "svm", "mlp", or "lr".

    mlp_ensemble_size : int
        Number of MLPs in the ensemble (only relevant if classifier_type="mlp").
        If 1, use a single MLP. If >1, create a VotingClassifier of that many MLPs.
        (only relevant if classifier_type="mlp" and multilabel=False).

    mlp_hidden_size : int
        Size of the one and only hidden layer in MLP(s). (only relevant if classifier_type="mlp").

    svm_c : float
        C parameter for SVM inversly proportionate to regularization strength (only relevant if classifier_type="svm").

    svm_kernel : str
        Kernel to use for SVM (only relevant if classifier_type="svm").

    random_state : int
        Base random state for reproducibility.

    use_char_ngrams : bool
        Whether to extract character n-grams from the morph text.

    char_ngram_range : tuple[int, int]
        (min_n, max_n) for character n-grams.

    use_morph_type : bool
        Whether to one-hot encode the morph type.

    use_morph_position : bool
        Whether to one-hot encode the morph position.

    use_morph_embedding : bool
        Whether to include an embedding for the morph itself.

    use_word_embedding : bool
        Whether to include an embedding for the entire word.

    use_vowel_start_end_features : bool
        Whether to include information if the morph starts and/or ends with a vowel.

    embedding_dimension : int
        Dimension of the fastText embeddings. (only relevant if word or morph embeddings are used).

    fasttext_model_path : str
        Path to the fastText .bin model file.

    verbose : bool
        If True, prints additional information during fit/predict.

    lower_case : bool
        If True, converts morph and word texts to lowercase before processing.

    multi_label : bool
        If True, treats etymologies as multi-label sets. If False, treats 
        the comma-joined string as a single class label.
   
    mlp_alpha : float
        Alpha parameter for MLP controls regularization strength (only relevant if classifier_type="mlp").
    mlp_max_iter : int
        Maximum iterations (epochs) to run. If the algorithm does not converge earlier. (only relevant if classifier_type="mlp").
            
    """

    def __init__(
        self,
        name: str = None,
        classifier_type: Literal["svm", "mlp", "lr"] = "mlp",
        mlp_ensemble_size: int = 1,
        mlp_hidden_size: int = 30,
        svm_c: float = 1.0,
        svm_kernel:str = 'rbf',
        random_state: int = 42,
        use_char_ngrams: bool = True,
        char_ngram_range: Tuple[int, int] = (1, 2),
        use_morph_type: bool = True,
        use_morph_position: bool = True,
        use_morph_embedding: bool = False,
        use_word_embedding: bool = False,
        embedding_dimension: int = 300,
        fasttext_model_path: str = "cc.cs.300.bin",
        verbose: bool = True,
        lower_case: bool = True,
        multi_label: bool = False,
        min_label_freq: int = 2,
        use_vowel_start_end_features: bool = True,
        early_stopping:bool = False,
        mlp_alpha:float = 0.0001,
        mlp_max_iter:int = 400
    ) -> None:
        super().__init__(name)
        if not name:
            classifier_description = classifier_type.upper()
            if  classifier_type == 'mlp':
                classifier_description += f"_{mlp_hidden_size}"
            if mlp_ensemble_size > 1:
                classifier_description += f"_ensemble{mlp_ensemble_size}"
            embeding_description = ''
            if use_word_embedding:
                embeding_description = '_word_emb'
            if use_morph_embedding:
                embeding_description += '_morph_emb'
            if use_word_embedding or use_morph_embedding:
                embeding_description += '_dim' + str(embedding_dimension)
            multi_label_describtion = ''
            if multi_label:
                multi_label_describtion = "_multi_label"


            self.name = f"{classifier_description}{embeding_description}{multi_label_describtion}"
        
        self.pipeline: Optional[Pipeline] = None

        # Classifier options
        self.classifier_type = classifier_type
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_alpha = mlp_alpha
        self.mlp_max_iter = mlp_max_iter
        self.svm_c = svm_c
        self.svm_kernel = svm_kernel
        self.random_state = random_state
        if not multi_label or mlp_ensemble_size == 1:
            self.mlp_ensemble_size = mlp_ensemble_size
        else:
            self.mlp_ensemble_size = 1
            if verbose:
                print("Ensembles cannot be used in multi label setting. Using single classifier.")


        # Feature flags
        self.use_char_ngrams = use_char_ngrams
        self.char_ngram_range = char_ngram_range
        self.use_morph_type = use_morph_type
        self.use_morph_position = use_morph_position
        self.use_morph_embedding = use_morph_embedding
        self.use_word_embedding = use_word_embedding
        self.use_vowel_start_end_features = use_vowel_start_end_features 

        # Embedding params
        self.embedding_dimension = embedding_dimension
        self.fasttext_model_path = fasttext_model_path

        # Rest
        self.verbose = verbose
        self.lower_case = lower_case
        self.min_label_freq = min_label_freq
        self._label_encoder:  Optional[LabelEncoder]        = None  
        self.early_stopping = early_stopping

        # Multi-label
        self.multi_label = multi_label
        self._multi_label_binarizer: Optional[MultiLabelBinarizer] = None  
        
    def _build_preprocessor(self) -> ColumnTransformer:
        # Build the list of transformers for ColumnTransformer
        transformers = []

        if self.use_char_ngrams:
            transformers.append((
                "char_ngrams",
                CountVectorizer(analyzer="char", ngram_range=self.char_ngram_range,lowercase=self.lower_case),
                "text"
            ))

        if self.use_morph_type:
            transformers.append((
                "morph_type",
                OneHotEncoder(),
                ["morph_type"]
            ))

        if self.use_morph_position:
            transformers.append((
                "morph_position",
                OneHotEncoder(),
                ["morph_position"]
            ))

        if self.use_word_embedding:
            transformers.append((
                "word_embedding",
                EmbeddingTransformer(
                    column="word",
                    embedding_dim=self.embedding_dimension,
                    fasttext_model_path=self.fasttext_model_path
                ),
                ["word"]
            ))

        if self.use_morph_embedding:
            transformers.append((
                "morph_embedding",
                EmbeddingTransformer(
                    column="text",
                    embedding_dim=self.embedding_dimension,
                    fasttext_model_path=self.fasttext_model_path
                ),
                ["text"]
            ))

        if self.use_vowel_start_end_features:
            transformers.append((
                "vowel_start_end_features",
                VowelStartEndTransformer(),
                ["text"]
            ))

        return ColumnTransformer(transformers=transformers)
  
    def _make_base_classifier(self):
         # Choose the base classifier based on user selection
        classifier_type = self.classifier_type.lower()
        if classifier_type == "svm":
            base_classifier = SVC(
                kernel=self.svm_kernel,
                C=self.svm_c,
                max_iter=1000,
                random_state=self.random_state
            )
        elif classifier_type == "mlp":
            # If ensemble_size <= 1, single MLP; else a VotingClassifier
            if self.mlp_ensemble_size <= 1:
                base_classifier = MLPClassifier(
                    hidden_layer_sizes=[self.mlp_hidden_size],
                    max_iter=self.mlp_max_iter,
                    verbose=False,
                    random_state=self.random_state,
                    early_stopping=self.early_stopping,
                    alpha=self.mlp_alpha
                )
            else:
                mlp_estimators = []
                for i in range(self.mlp_ensemble_size):
                    rs = self.random_state + i
                    mlp = MLPClassifier(
                        hidden_layer_sizes=[self.mlp_hidden_size],
                        max_iter=self.mlp_max_iter,
                        verbose=False,
                        random_state=rs
                    )
                    mlp_estimators.append((f"mlp_{i+1}", mlp))

                # Wrap them in a VotingClassifier
                base_classifier = VotingClassifier(
                    estimators=mlp_estimators,
                    voting='hard',
                    n_jobs=-1
                )
        elif classifier_type == "lr":
            base_classifier = LogisticRegression(
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        return base_classifier

    def _sentences_to_dataframe(
            self,
            sentences: List["DataSentence"],
            lowercase: bool = True
    ) -> pd.DataFrame:
        """Flatten a list of DataSentence objects into a pandas DataFrame."""
        rows: list[dict] = []
        for sentence in sentences:
            for word in sentence.words:
                for morph in word:
                    if not morph.etymology:        # skip unlabeled morphs
                        continue
                    rows.append({
                        "text": morph.text.lower() if lowercase else morph.text,
                        "word": word.text.lower() if lowercase else word.text,
                        "morph_type": morph.morph_type.value,
                        "morph_position": morph.morph_position.value,
                        "label": ",".join(morph.etymology)
                    })
        return pd.DataFrame(rows)

    def fit(self, data: List["DataSentence"]) -> None:
        """
        Builds and trains the morph-level classification pipeline.
        
        It flattens the training data into a DataFrame, extracting:
            - "text": the morph text
            - "word": the complete word text
            - "morph_type": the string value of the morph type
            - "morph_position": the string value of the morph position
            - "label": comma-separated etymology (single-label) or a list if multi_label.
        """
        if self.verbose:
            print(f"Fiting model: {self.name}")
        df_train = self._sentences_to_dataframe(data,lowercase=self.lower_case)


        if df_train.empty:
            raise ValueError("No training data with non‑empty etymology.")
        
        # Discard low-frequency labels
        label_counts = df_train["label"].value_counts()
        number_frames_before = len(df_train)
        df_train = df_train[df_train["label"].map(label_counts) >= self.min_label_freq]
        if df_train.empty:
            raise ValueError(
                f"All morphs were discarded because their labels' frequency < {self.min_label_freq}."
            )
       
        if self.verbose:
            number_removed = number_frames_before-len(df_train)
            if number_removed > 0:
                print(f"Removed {number_removed} morphs with low occurence etymology sequences")
        
  

        base_classifier = self._make_base_classifier()        

        preprocessor = self._build_preprocessor()

        if self.multi_label:
            self._multi_label_binarizer = MultiLabelBinarizer()
            y_train = self._multi_label_binarizer.fit_transform(df_train["label"].str.split(","))
            final_classifier = OneVsRestClassifier(base_classifier)

        else:
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(df_train["label"])
            final_classifier = base_classifier



        X_train = df_train[["text", "word", "morph_type", "morph_position"]]
        Xt_train = preprocessor.fit_transform(X_train)      
        if self.verbose:
            print("Shape of transformed data:", Xt_train.shape)

            print(f"Parameters:")
            print(f"  classifier_type={self.classifier_type}")
            if self.classifier_type.lower() == "svm":
                print(f"  svm_c={self.svm_c}")
            elif self.classifier_type.lower() == "mlp":
                print(f"  mlp_hidden_size={self.mlp_hidden_size}")
            elif self.classifier_type.lower() == "lr":
                print("  Using LogisticRegression")
            if self.mlp_ensemble_size > 1:
                print(f"  ensemble_size={self.mlp_ensemble_size}")
            if self.multi_label:   
                 print("  multi_label=True (using OneVsRestClassifier)")
            else:
                print("  multi_label=False (single-label)")
  
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", final_classifier)])
        final_classifier.fit(Xt_train,y_train)

    def predict(self, data: List["DataSentence"]) -> List["DataSentence"]:
        """
        Predicts the etymology for each morph in the provided DataSentence objects.
        
        - If multi_label=False, we do single-label classification. The pipeline outputs a single string like "AA,BB".
          We split that string by comma to assign morph.etymology = ["AA", "BB"].
        - If multi_label=True, we do multi-label classification. The pipeline outputs a binary matrix from the 
          OneVsRestClassifier. We then use MultiLabelBinarizer.inverse_transform() to get the list of labels.

          Returns list of DataSentence objects with predicted etymology.
        """
        if self.pipeline is None:
            raise ValueError("The model has not been trained. Call fit() or load() first.")
        updated_data = copy.deepcopy(data)
        rows = []
        morph_refs :list[Morph] = []  # for assigning predictions later
        MorphType = Morph.MorphType
        for sentence in updated_data:
            for word in sentence.words:
                for morph in word:
                    if morph.morph_type == MorphType.UNDEFINED:
                        continue
                    if not morph.text.isalpha():
                        continue

                    rows.append({
                        "text": morph.text,
                        "word": word.text,
                        "morph_type": morph.morph_type.value,
                        "morph_position": morph.morph_position.value
                    })
                    morph_refs.append(morph)

        if not rows:
            return data  # Nothing to predict

        df_all = pd.DataFrame(rows)

        if self.multi_label:
            preds_binary = self.pipeline.predict(df_all)
            decoded_preds = self._multi_label_binarizer.inverse_transform(preds_binary)
        else:
            preds = self.pipeline.predict(df_all)
            decoded_preds = self._label_encoder.inverse_transform(preds)
            decoded_preds = [label.split(",") for label in decoded_preds]

        for morph, labels in zip(morph_refs, decoded_preds):
            if labels:
                morph.etymology = list(labels)
            else:
                morph.etymology = ["ces"]  # fallback if prediction is empty

        return updated_data

    def save(self, filename: str) -> None:
        """
        Saves the trained pipeline and (if multi-label) the MultiLabelBinarizer to a file.
        Note that the fastText .bin model is not saved here.
        """
        if self.pipeline is None:
            raise ValueError("No pipeline to save. Train the model first.")

        # Store all neccesery parts (pipelines and mlb binarizer) and flags into dict
        objects_to_save = {
            "pipeline": self.pipeline,
            "mlb": self._multi_label_binarizer,
            'name' :self.name,
            'multi_label' : self.multi_label,
            'use_word_embedd' : self.use_word_embedding,
            'use_morph_embedd' : self.use_morph_embedding,
            'embed_dim' : self.embedding_dimension,
            'use_morph_type': self.use_morph_type,
            'use_char_ngrams' : self.use_char_ngrams,
            'use_morph_position' : self.use_morph_position,
            'use_vowels' : self.use_vowel_start_end_features,
            "label_encoder": self._label_encoder,     
        }

        with open(filename, "wb") as f:
            pickle.dump(objects_to_save, f)
        if self.verbose:
            print(f"Model saved to {filename}")

    def load(self, filename: str) -> None:
        """
        Loads the pipeline and the MultiLabelBinarizer from disk.
        Additionaly restore flags what features to use.
        The fastText model path is not included; 
        ensure fasttext_model_path is set correctly for inference.
        """
        with open(filename, "rb") as f:
            saved_data:dict = pickle.load(f)

        self.pipeline = saved_data["pipeline"]
        self._multi_label_binarizer = saved_data["mlb"]
        self.name = saved_data["name"]
        self.multi_label = saved_data['multi_label']
        self._label_encoder = saved_data.get("label_encoder")  
        
        # feature flags
        self.use_word_embedding = saved_data.get("use_word_embedd", self.use_word_embedding)
        self.use_morph_embedding = saved_data.get("use_morph_embedd", self.use_morph_embedding)
        self.embedding_dimension = saved_data.get("embed_dim", self.embedding_dimension)
        self.use_morph_type = saved_data.get("use_morph_type", self.use_morph_type)
        self.use_char_ngrams = saved_data.get("use_char_ngrams", self.use_char_ngrams)
        self.use_morph_position = saved_data.get("use_morph_position", self.use_morph_position)
        self.use_vowel_start_end_features = saved_data.get("use_vowels", self.use_vowel_start_end_features)

        if self.verbose:
            print(f"Model loaded from {filename}")
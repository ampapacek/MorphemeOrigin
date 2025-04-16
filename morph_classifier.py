import copy
import pickle
from typing import List, Optional, Literal, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from model import Model
from data_sentece import DataSentence
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

    mlp_hidden_size : int
        Size of the hidden layer in MLP(s). (Single-layer only.)

    svm_c : float
        C parameter for SVM (only relevant if classifier_type="svm").

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
        Dimension of the fastText embeddings.

    fasttext_model_path : str
        Path to the fastText .bin model file.

    verbose : bool
        If True, prints additional information during fit/predict.

    lower_case : bool
        If True, converts morph and word texts to lowercase before processing.

    multi_label : bool
        If True, treats etymologies as multi-label sets. If False, treats 
        the comma-joined string as a single class label.
   
     fallback_single_label : bool
        If True, train both single and multi label pipelines.
        When the multilabel pipeline predicts empty sequence use the single label as fallback.
        Used only when multi_label=True.
    """

    def __init__(
        self,
        name: str = None,
        classifier_type: Literal["svm", "mlp", "lr"] = "mlp",
        mlp_ensemble_size: int = 1,
        mlp_hidden_size: int = 30,
        svm_c: float = 1.0,
        svm_kernel:str = 'svm',
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
        fallback_single_label: bool = False,
        use_vowel_start_end_features: bool = True

    ) -> None:
        super().__init__(name)
        if not name:
            classifier_description = classifier_type.upper()
            if  classifier_type == 'mlp':
                classifier_description += f"_{mlp_hidden_size}"
            if mlp_ensemble_size > 1:
                classifier_description += f"_ensamble{mlp_ensemble_size}"
            embeding_description = ''
            if use_word_embedding:
                embeding_description = '_word_emb'
            if use_morph_embedding:
                embeding_description += '_morph_emb'
            if embeding_description == '': 
                embeding_description = '_no_emb'
            if use_word_embedding or use_morph_embedding:
                embeding_description += '_dim' + str(embedding_dimension)
            multi_label_describtion = ''
            if multi_label:
                multi_label_describtion = "_multi_label"


            self.name = f"{classifier_description}{embeding_description}{multi_label_describtion}_model"
        
        self.pipeline: Optional[Pipeline] = None

        # Classifier options
        self.classifier_type = classifier_type
        self.mlp_ensemble_size = mlp_ensemble_size
        self.mlp_hidden_size = mlp_hidden_size
        self.svm_c = svm_c
        self.svm_kernel = svm_kernel
        self.random_state = random_state

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

        # Multi-label
        self.multi_label = multi_label
        self.use_fallback_pipeline = fallback_single_label
        self._mlb: Optional[MultiLabelBinarizer] = None  # For multi-label binarizing
        self.fallback_pipeline:Optional[Pipeline] = None

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
        morph_rows = []
        for sentence in data:
            for word in sentence.words:
                for morph in word:
                    if morph.etymology:
                        # Lowercase logic
                        if self.lower_case:
                            morph_text = morph.text.lower()
                            word_text = word.text.lower()
                        else:
                            morph_text = morph.text
                            word_text = word.text

                        # Store the joined label as is. We handle multi-label below.
                        full_label = ",".join(morph.etymology)

                        morph_rows.append({
                            "text": morph_text,
                            "word": word_text,
                            "morph_type": morph.morph_type.value,
                            "morph_position": morph.morph_position.value,
                            "label": full_label
                        })

        df = pd.DataFrame(morph_rows)
        if df.empty:
            raise ValueError("No training data available (no morphs with non-empty etymology).")
        
        # Discard low-frequency labels
        label_counts = df["label"].value_counts()
        number_frames_before = len(df)
        df = df[df["label"].map(label_counts) >= self.min_label_freq]
        if df.empty:
            raise ValueError(
                f"All morphs were discarded because their labels' frequency < {self.min_label_freq}."
            )
        if self.verbose:
            print(f"Removed {number_frames_before-len(df)} morphs with low occurence etymology sequences")

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

        preprocessor = ColumnTransformer(transformers=transformers)

        # Choose the base classifier based on user selection
        classifier_type = self.classifier_type.lower()
        if classifier_type == "svm":
            base_classifier = SVC(
                kernel=self.svm_kernel,
                C=self.svm_c,
                max_iter=5000,
                random_state=self.random_state
            )
        elif classifier_type == "mlp":
            # If ensemble_size <= 1, single MLP; else a VotingClassifier
            if self.mlp_ensemble_size <= 1:
                base_classifier = MLPClassifier(
                    hidden_layer_sizes=[self.mlp_hidden_size],
                    max_iter=400,
                    verbose=False,
                    random_state=self.random_state
                )
            else:
                mlp_estimators = []
                for i in range(self.mlp_ensemble_size):
                    rs = self.random_state + i
                    mlp = MLPClassifier(
                        hidden_layer_sizes=[self.mlp_hidden_size],
                        max_iter=300,
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
                max_iter=300
            )
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        # If multi_label, we wrap the base_classifier in a OneVsRestClassifier
        # and do multi-label binarization for y.
        if self.multi_label:
            self._mlb = MultiLabelBinarizer()
            # Convert the comma-separated string, for example "lat,ell" to list ["lat", "ell"]
            y_list = [label_str.split(",") for label_str in df["label"]]
            y_bin = self._mlb.fit_transform(y_list)
            final_classifier = OneVsRestClassifier(base_classifier)
            
            if self.use_fallback_pipeline: # prepare another pipeline single-label classifier in case of empty prediction
                y_bin_fallback =  df["label"]
                self.fallback_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", base_classifier)
            ])
        else:
            # Single-label: use the full label (whole sequence) as one class
            self._mlb = None
            y_bin = df["label"]
            final_classifier = base_classifier

        # Create the pipeline
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", final_classifier)
        ])

        # Separate features from the dataframe
        X = df[["text", "word", "morph_type", "morph_position"]]

        if self.verbose:
            print(f"Fitting MorphClassifier {self.name} with parameters:")
            print(f"  classifier_type={self.classifier_type}")
            if classifier_type == "svm":
                print(f"  svm_c={self.svm_c}")
            elif classifier_type == "mlp":
                print(f"  mlp_hidden_size={self.mlp_hidden_size}")
            elif classifier_type == "lr":
                print("  Using LogisticRegression")
            if self.mlp_ensemble_size > 1:
                print(f"  ensemble_size={self.mlp_ensemble_size}")
            if self.multi_label:
                if self.use_fallback_pipeline:
                    print("  multi_label=True (using OneVsRestClassifier), with fallback single-label pipeline")
                else:
                    print("  multi_label=True (using OneVsRestClassifier)")
            else:
                print("  multi_label=False (single-label)")

        # Show shape after preprocessing (if verbose)
        X_transformed = self.pipeline.named_steps["preprocessor"].fit_transform(X)
        if self.verbose:
            print("Feature matrix shape after preprocessing:", X_transformed.shape)

        # Fit the pipeline
        self.pipeline.fit(X, y_bin)
        if self.use_fallback_pipeline and self.fallback_pipeline:
            if self.verbose:
                print("Fitting additional fallback pipeline...")
            self.fallback_pipeline.fit(X,y_bin_fallback)

        if self.verbose:
            print(f"Training complete for model {self.name}")


    def predict(self, data: List["DataSentence"]) -> List["DataSentence"]:
        """
        Predicts the etymology for each morph in the provided DataSentence objects.
        
        - If multi_label=False, we do single-label classification. The pipeline outputs a single string like "AA,BB".
          We split that string by comma to assign morph.etymology = ["AA", "BB"].
        - If multi_label=True, we do multi-label classification. The pipeline outputs a binary matrix from the 
          OneVsRestClassifier. We then use MultiLabelBinarizer.inverse_transform() to get the list of labels.
        """
        if self.pipeline is None:
            raise ValueError("The model has not been trained. Call fit() or load() first.")

        updated_data = []
        for sentence in data:
            sent_copy = copy.deepcopy(sentence)
            for word in sent_copy.words:
                for morph in word:
                    if morph.morph_type == morph.__class__.MorphType.UNDEFINED:
                        continue

                    morph_text = morph.text
                    word_text = word.text

                    if self.lower_case:
                        morph_text = morph_text.lower()
                        word_text = word_text.lower()

                    # Only predict if alpha
                    if morph_text.isalpha():
                        df_morph = pd.DataFrame([{
                            "text": morph_text,
                            "word": word_text,
                            "morph_type": morph.morph_type.value,
                            "morph_position": morph.morph_position.value
                        }])
                        if self.multi_label:
                            # OneVsRestClassifier => pipeline outputs a binary matrix
                            # We'll use our stored MultiLabelBinarizer to get label sets
                            bin_pred = self.pipeline.predict(df_morph)
                            # bin_pred is shape (1, n_classes) => a single row
                            # inverse_transform returns a list of tuples or lists
                            label_list = self._mlb.inverse_transform(bin_pred)
                            # label_list is something like [("AA", "BB")]
                            # Convert that to a Python list of strings
                            morph.etymology = list(label_list[0])  # the first (and only) row
                            if morph.etymology == []:
                                # The classifier returned an empty sequence
                                if self.use_fallback_pipeline and self.fallback_pipeline:
                                    # Use fallback single pipeline
                                    morph.etymology = (self.fallback_pipeline.predict(df_morph)[0]).split(',')
                                else:
                                    # Predict ['ces']
                                    morph.etymology = ['ces']
                        else:
                            # Single-label => pipeline outputs a single string
                            pred_label = self.pipeline.predict(df_morph)[0]
                            morph.etymology = pred_label.split(",")
                    else:
                        morph.etymology = []
            updated_data.append(sent_copy)

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
            'fallback_pipeline': self.fallback_pipeline,
            "mlb": self._mlb,
            'name' :self.name,
            'multi_label' : self.multi_label,
            'use_word_embedd' : self.use_word_embedding,
            'use_morph_embedd' : self.use_morph_embedding,
            'embed_dim' : self.embedding_dimension,
            'use_morph_type': self.use_morph_type,
            'use_char_ngrams' : self.use_char_ngrams,
            'use_morph_position' : self.use_morph_position,
            'use_vowels' : self.use_vowel_start_end_features,
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
            saved_data = pickle.load(f)

        self.pipeline = saved_data["pipeline"]
        self._mlb = saved_data["mlb"]
        self.fallback_pipeline = saved_data["fallback_pipeline"]
        self.name = saved_data["name"]
        self.multi_label = saved_data['multi_label']

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
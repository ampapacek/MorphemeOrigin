import urllib.request
import gzip
import shutil
import sys
import os
import pandas as pd
import numpy as np
import fasttext
import fasttext.util
from sklearn.base import BaseEstimator, TransformerMixin

# Use a global dictionary as a cache,
# so we do not reload fastText multiple times.
# Key is (model path + dimension), value is the loaded fastText model.
# the loaded fastText model cannot be pickled so its causing problems if the model itself would be in the transformer

_ft_cache = {}

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that converts a column of text (e.g., word or morph)
    into its corresponding embedding vector using a fastText model.

    If the specified fastText model does not exist locally, it is downloaded
    from `download_url` and decompressed (if necessary).
    """

    def __init__(self,
                 column: str,
                 embedding_dim: int = 300,
                 fasttext_model_path: str = "cc.cs.300.bin",
                 download_url: str = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.bin.gz",
                 verbose: bool = True):
        self.column = column

        if embedding_dim > 300:
            print(f"Invalid embedding dimenstion: {embedding_dim}. Maximum embedding dimension is 300. Setting the dimension to 300.")
            embedding_dim = 300
        if embedding_dim <= 0:
            print(f"Invalid embedding dimenstion: {embedding_dim}. Embedding dimension must be positive. Setting the dimension to 300.")
            embedding_dim = 300
        self.embedding_dim = embedding_dim

        self.fasttext_model_path = fasttext_model_path
        self.download_url = download_url
        self.verbose = verbose

        self.fasttext_model_name = fasttext_model_path + str(embedding_dim)


    def fit(self, X, y=None):
        # Ensure the fastText model is loaded
        # Load the model from file if not in cache already
        # If the file is not present download it
        self.get_model()
        return self

    def transform(self, X, y=None):
        model = self.get_model()
        # Build embeddings from the specified column
        embeddings = X[self.column].apply(lambda word: model.get_word_vector(word))
        return np.vstack(embeddings.values)

    def get_model(self):
        """
        Loads the fastText model from disk if not already cached.
        If the file doesn't exist, downloads from self.download_url, then decompresses.
        If embedding_dim < 300, reduce dimensions accordingly.
        """
        global _ft_cache

        # Check cache first
        if self.fasttext_model_name in _ft_cache:
            return _ft_cache[self.fasttext_model_name]

        # Check if the file exists on disk
        if not os.path.exists(self.fasttext_model_path):
            self._download_and_decompress()

        if self.verbose:
            print(f"Loading fastText model from: {self.fasttext_model_path}")

        # Load the model
        _ft_cache[self.fasttext_model_name] = fasttext.load_model(self.fasttext_model_path)

        if self.verbose:
            print("Embeddings loaded.")

        # If dimension < 300, reduce dimension
        if 0 < self.embedding_dim < 300:
            if self.verbose:
                print("Reducing dimension...")
            fasttext.util.reduce_model(_ft_cache[self.fasttext_model_name], self.embedding_dim)
            assert self.embedding_dim == _ft_cache[self.fasttext_model_name].get_dimension()
            if self.verbose:
                print(f"Reduced dimension to {self.embedding_dim}")

        return _ft_cache[self.fasttext_model_name]
        
    def _download_and_decompress(self):
        """
        Downloads the model file from self.download_url to self.fasttext_model_path + '.gz',
        then decompresses it to self.fasttext_model_path, and removes the .gz file.
        Includes a simple console progress bar for the download.
        """

        gz_path = self.fasttext_model_path + ".gz"
        if self.verbose:
            print(f"Downloading fastText model from {self.download_url} ...")

        def _progress(block_count, block_size, total_size):
            if total_size > 0:
                percent = block_count * block_size * 100 / total_size
                sys.stdout.write(f"\rDownload progress: {percent:.2f}%")
                sys.stdout.flush()
            else:
                # In case total_size is unknown (some servers may not send it)
                downloaded = block_count * block_size
                sys.stdout.write(f"\rDownload progress: {downloaded} bytes transferred")
                sys.stdout.flush()

        # Download the gzipped model with a progress callback
        urllib.request.urlretrieve(self.download_url, gz_path, reporthook=_progress)
        sys.stdout.write("\n")  # move to the next line after download is complete

        if self.verbose:
            print(f"Decompressing {gz_path} to {self.fasttext_model_path} ...")

        # Decompress
        with gzip.open(gz_path, 'rb') as f_in, open(self.fasttext_model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Remove the .gz file
        os.remove(gz_path)

        if self.verbose:
            print(f"Decompressed and saved model to {self.fasttext_model_path}")


class VowelStartEndTransformer(BaseEstimator, TransformerMixin):
    """
    A simple transformer that adds two binary features:
      1) starts_with_vowel (0 or 1)
      2) ends_with_vowel   (0 or 1)

    By default, we define vowels as 'aeiouáéíóúý' etc. (customize as you wish).
    """
    def __init__(self, vowels: str = "aeiouyáéěíóůúý"):
        self.vowels = vowels

    def fit(self, X, y=None):
        """Nothing to fit, just return self"""
        return self

    def transform(self, X):
        """
        Transforms input text to two dimension binary vector [starts_vowel, ends_vowel].
        1 means it starts with a vowel, 0 that it starts with a consonant
        X is expected to be a DataFrame or array with at least one column: "text"
        """
        vowels_set = set(self.vowels)
        starts_ends = []
        for text in X["text"]:
            if not text:
                # If it's empty or something else, default to 0,0
                starts_ends.append([0,0])
                continue
            first = text[0]
            last = text[-1]

            starts_vowel = 1 if first.lower() in vowels_set else 0
            ends_vowel = 1 if last.lower() in vowels_set else 0
            starts_ends.append([starts_vowel, ends_vowel])
        
        return pd.DataFrame(starts_ends, columns=["starts_vowel", "ends_vowel"])

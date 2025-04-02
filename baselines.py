from collections import defaultdict, Counter
from typing import List
from data_sentece import DataSentence,Word,Morph
from model import Model
import sys
import requests
import copy

class DummyModel(Model):
    """
    A dummy model that sets the etymology for every alphabetic morph to ["ces"] (native most frequent class).
    """
    
    def __init__(self, name: str = "DummyModel") -> None:
        super().__init__(name)
    
    def fit(self, data: List[DataSentence] = None) -> None:
        """Does nothing."""
        # No training needed.
        pass

    def predict(self, data: List[DataSentence]) -> List[DataSentence]:
        # Make a deep copy so original data is not altered.
        predictions = copy.deepcopy(data)
        for sentence in predictions:
            for word in sentence.words:
                for morph in word.morphs:
                    if morph.text.isalpha():
                        morph.etymology = ["ces"]
        return predictions

class MostFrequentOriginModel(Model):
    """
    A simple baseline model that learns the most frequent origin sequence for each morph 
    from training data. The key used is a tuple (morph.text, morph.morph_type).
    
    When predicting, if a morph with the same (text, type) was seen during training,
    its predicted etymology is set to the most frequent sequence; otherwise, it defaults to ["ces"].
    """
    
    def __init__(self, name: str = "MostFrequentOriginModel") -> None:
        super().__init__(name)
        # Dictionary mapping (morph.text, morph.morph_type) -> most frequent etymology sequence (list of str)
        self.most_freq = {}
    
    def fit(self, sentences: List[DataSentence]) -> None:
        """
        Learns the most frequent origin sequence for each morph (keyed by (text, type)) 
        from the training sentences.
        """
        freq = defaultdict(Counter)
        for sentence in sentences:
            # Iterate over every morph in the sentence.
            for morph in sentence:
                # Create a key as a tuple (morph.text, morph.morph_type)
                key = (morph.text, morph.morph_type)
                # Convert the etymology list to a tuple (hashable) for counting.
                etym_seq = tuple(morph.etymology)
                freq[key][etym_seq] += 1
        
        # For each key, select the most common etymology sequence.
        for key, counter in freq.items():
            most_common, _ = counter.most_common(1)[0]
            # Store the most common sequence (converted back to list).
            self.most_freq[key] = list(most_common)
    
    def predict(self, sentences: List[DataSentence]) -> List[DataSentence]:
        """
        Predicts etymology for each morph in the given sentences.
        
        For each morph, if the (text, type) key was observed during training, its etymology 
        is set to the most frequent sequence; otherwise, it defaults to ["ces"].
        """
        predictions = []
        # Create deep copies to preserve original data.
        for sentence in sentences:
            sent_copy = copy.deepcopy(sentence)
            for morph in sent_copy:
                key = (morph.text, morph.morph_type)
                if key in self.most_freq:
                    morph.etymology = self.most_freq[key]
                else:
                    morph.etymology = ["ces"]
            predictions.append(sent_copy)
        return predictions

class RuleBasedEtymologyModel(Model):
    """
    A rule-based model that assigns etymology based on root and affix dictionaries.
    
    For each alphabetic morph:
      - If the morph is a root (and longer than 2 characters) and is found in the root etymology dictionary,
        its etymology is assigned accordingly.
      - If it is a derivational affix, the model attempts to assign its etymology using the affix dictionaries.
      - Otherwise, it defaults to ["ces"].
    """
    def __init__(self, root_etym_file: str, affixes_file: str, name: str = "RuleBasedEtymologyModel") -> None:
        super().__init__(name)
        self.roots_etymology = load_etym_dict(root_etym_file)
        self.prefixes, self.suffixes = load_affixes(affixes_file)

    def fit(self, data: List["DataSentence"]) -> None:
        """Does nothing."""
        # No training is necessary.
        pass

    def predict(self, data: List["DataSentence"]) -> List["DataSentence"]:
        predictions = copy.deepcopy(data)
        for sentence in predictions:
            for word in sentence.words:
                for morph in word.morphs:
                    if morph.text.isalpha():
                        if (morph.morph_type == morph.__class__.MorphType.ROOT and
                            len(morph.text) > 2 and 
                            (morph.text in self.roots_etymology or morph.text.lower() in self.roots_etymology)):
                            key = morph.text if morph.text in self.roots_etymology else morph.text.lower()
                            morph.etymology = self.roots_etymology[key]
                        elif morph.morph_type == morph.__class__.MorphType.DERIVATIONAL_AFFIX:
                            affix_etym = get_affix_etymology(morph.text, word.text, self.prefixes, self.suffixes)
                            morph.etymology = affix_etym if affix_etym else ["ces"]
                        else:
                            morph.etymology = ["ces"]
        return predictions


class WordBasedEtymologyModel(Model):
    """
    A word-based model that assigns etymology using a word-level etymology dictionary.
    
    The model uses the MorphoDiTa API to obtain lemmata for the full text,
    then for each word it looks up etymology in the word-level dictionary.
    For derivational affixes, it falls back to affix analysis.
    """
    def __init__(self, word_etym_file: str, affixes_file: str, name: str = "WordBasedEtymologyModel") -> None:
        super().__init__(name)
        self.words_etymology = load_etym_dict(word_etym_file)
        self.prefixes, self.suffixes = load_affixes(affixes_file)


    def fit(self, data: List["DataSentence"]) -> None:
        # In this simple model, fit does nothing.
        pass

    @staticmethod
    def get_lemmata(text: str) -> List[str]:
        """
        Retrieves lemmata for the given text using the MorphoDiTa API.
        """
        url = "http://lindat.mff.cuni.cz/services/morphodita/api/analyze"
        params = {
            "data": text,
            "output": "json",
            "guesser": "yes",
            "convert_tagset": "strip_lemma_comment",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        json_data = response.json()
        sentences = json_data.get("result", [])
        lexemes = []
        for sentence in sentences:
            for token in sentence:
                analyses = token.get("analyses", [])
                if analyses:
                    lexeme = analyses[0].get("lemma", token.get("token", ""))
                else:
                    lexeme = token.get("token", "")
                lexemes.append(lexeme)
        return lexemes

    def predict(self, data: List["DataSentence"]) -> List["DataSentence"]:
        predictions = copy.deepcopy(data)
        # Build the full text from sentences.
        whole_text = " ".join(sentence.sentence for sentence in predictions)
        lemmata = WordBasedEtymologyModel.get_lemmata(whole_text)
        # Collect all words from the sentences.
        all_words = []
        for sentence in predictions:
            all_words.extend(sentence.words)
        
        if len(lemmata) != len(all_words):
            raise Exception(f"Length mismatch: {len(lemmata)} lemmata vs {len(all_words)} words.")
        
        # For each word, attempt to assign etymology based on the word-level dictionary.
        for word, lemma in zip(all_words, lemmata):
            word_etym = self.words_etymology.get(word.text, [])
            if not word_etym:
                word_etym = self.words_etymology.get(lemma, [])
            for morph in word:
                if morph.text.isalpha():
                    if morph.morph_type == morph.__class__.MorphType.ROOT:
                        morph.etymology = word_etym if word_etym else ["ces"]
                    elif morph.morph_type == morph.__class__.MorphType.DERIVATIONAL_AFFIX:
                        affix_etym = get_affix_etymology(morph.text, word.text, self.prefixes, self.suffixes)
                        morph.etymology = affix_etym if affix_etym else ["ces"]
                    else:
                        morph.etymology = ["ces"]
        return predictions

def load_etym_dict(filepath:str) -> dict[str,list[str]]:
    etymology = {}
    with open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip blank lines.
            parts = line.split('\t')
            if len(parts) == 2:
                entry = parts[0].strip()
                languages_str = parts[1].strip()
                # Split the languages string by comma and strip each language.
                languages = [lang.strip() for lang in languages_str.split(',') if lang.strip()]
                etymology[entry] = languages
            else:
                continue # skip lines that are not in the right format
    return etymology

def get_affix_etymology(morph:str, word:str, prefixes:dict[str,list[str]], suffixes:dict[str,list[str]]) -> str:
    """Returns the etymology for the morph if it is affix and the information is in the dictionary. Empty string otherwise"""
    # naive simple clasification not too acurate, but in prefixes and affixes dictionaries are just the affixes which usually cannot be roots or other affixes
    morph = morph.lower() # lowercase the morph
    if word.startswith(morph):
        # prefix
        if morph in prefixes:
            return prefixes[morph] 
    elif word.endswith(morph) or (not word.startswith(morph) and len(morph) < 3):
        # suffix
        if morph in suffixes:
            # print(f"Suffix found!!!, In word {word}, suffix -{morph} with etymology {suffixes[morph]}")
            return suffixes[morph]
    return "" # if morph is not suffix nor prefix or is not in the dictionaries fall back on empty string

def load_affixes(afixes_file:str = 'affixes.tsv') -> tuple[dict[str,list[str]],dict[str,list[str]]]:
    prefixes = {}
    suffixes = {}
    with open(afixes_file, 'rt') as affixes:
        for line in affixes:
            if len(line.split('\t')) != 2: continue
            affix, etymology = line.split('\t')
            if affix.startswith('-'):
                suffixes[affix[1:]] = etymology.strip().split(',')
            elif affix.endswith('-'):
                prefixes[affix[:-1]] = etymology.strip().split(',')
            else:
                pass # ignore everything else
    return prefixes,suffixes






# These methods (below) are not used anywhere

def dummy_predict_deprecated(sentences: List[DataSentence]) -> List[DataSentence]:
    """
    A dummy prediction function that sets the etymology to ["ces"]
    for each morph whose text is alphabetic.
    """
    for sentence in sentences:
        for word in sentence.words:
            for morph in word:
                if morph.text.isalpha():
                    morph.etymology = ["ces"]
    return sentences


def rule_predict_deprecated(sentences: List[DataSentence], roots_etymology:dict[str,list[str]], prefixes:dict[str,list[str]], suffixes:dict[str,list[str]]) -> List[DataSentence]:
    """
    Prediction function that looks at roots and affixes and sets etymology for them if in dictionary,
    sets the etymology of rest to ["ces"] for each morph whose text is alphabetic (rest keep without etymology).
    """
    
    for sentence in sentences:
        for word in sentence.words:
            for morph in word:
                if morph.text.isalpha():
                    if morph.morph_type == Morph.MorphType.ROOT and len(morph.text) > 2 and (morph.text in roots_etymology or morph.text.lower() in roots_etymology):
                        if morph.text in roots_etymology:
                            morph.etymology = roots_etymology[morph.text]
                        else:
                            morph.etymology = roots_etymology[morph.text.lower()]
                    elif morph.morph_type == Morph.MorphType.DERIVATIONAL_AFFIX:
                        affix_etymology = get_affix_etymology(morph.text, word.text, prefixes, suffixes) # returns the etymology or ""
                        if affix_etymology:
                            morph.etymology = affix_etymology
                        else:
                            morph.etymology = ["ces"]
                    else:
                        # Inflectional affix or undefined
                        morph.etymology = ["ces"]
                else:
                    pass # keep blank etymology for punctuation, numbers etc.
    return sentences


def get_lemmata_deprecated(text: str) -> List[str]:
    """
    Retrieves the lexemes (lemmas) for the input text using the MorphoDiTa API.
    
    This method sends the input text to the MorphoDiTa 'analyze' endpoint with JSON output.
    The API returns a JSON array of sentences, where each sentence is an array of tokens.
    Each token is an object that contains the original token, and an 'analyses' field
    (a list of analyses). We take the lemma from the first analysis for each token.
    
    Args:
        text (str): Input text.
        
    Returns:
        List[str]: A list of lemmas corresponding to each token in the input.
                  For example, from "going" it would return ["go"].
    
    Raises:
        requests.HTTPError: If the API request fails.
    """
    url = "http://lindat.mff.cuni.cz/services/morphodita/api/analyze"
    
    # Parameters for the API call.
    params = {
        "data": text,
        "output": "json",

        "guesser": "yes",
        "convert_tagset": "strip_lemma_comment",
        # "derivation": "root"
    }
    
    # Send GET request to the MorphoDiTa API.
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raises an error if the request failed.
    
    # Parse the JSON response.
    json_data = response.json()
    sentences = json_data.get("result", [])
    
    lexemes = []
    # Iterate over each sentence and token to extract lemmas.
    for sentence in sentences:
        for token in sentence:
            analyses = token.get("analyses", [])
            # Use the lemma from the first analysis if available;
            # otherwise, default to the token form.
            if analyses:
                lexeme = analyses[0].get("lemma", token.get("token", ""))
            else:
                lexeme = token.get("token", "")
            lexemes.append(lexeme)
    
    return lexemes

def word_predict_deprecated(sentences: List[DataSentence], words_etymology:dict[str,list[str]], prefixes:dict[str,list[str]], suffixes:dict[str,list[str]]) -> List[DataSentence]:
    """
    Prediction function that looks at etymology of given wordroots and affixes and sets etymology for them if in dictionary,
    sets the etymology of rest to ["ces"] for each morph whose text is alphabetic (rest keep without etymology).
    """
    whole_text = ' '.join([sentence.sentence for sentence in sentences])
    lematized_text = [lemma_base.strip().split('-')[0] for lemma_base in get_lemmata_deprecated(whole_text)]
    words = []
    for sentence in sentences:
        words.extend(sentence.words)

    if len(lematized_text) != len(words):
        # print(lematized_text, "Lemmata")
        # print(sentence.words, "Sentence")
        raise Exception(f"An error occured, the count of lemmata: {len(lematized_text)} obtained from sentence is not same as the number of words in the sentence: {len(words)}.")
    
    for word,lemma in zip(words, lematized_text):
            word_etymology = words_etymology.get(word.text,[])
            if word_etymology == []:
                word_etymology = words_etymology.get(lemma,[])                

            for morph in word:
                if morph.text.isalpha():
                    if morph.morph_type == Morph.MorphType.ROOT:
                        if word_etymology:
                            morph.etymology = word_etymology
                        else:
                            morph.etymology = ["ces"]
                    elif morph.morph_type == Morph.MorphType.DERIVATIONAL_AFFIX:
                        affix_etymology = get_affix_etymology(morph.text, word.text, prefixes, suffixes) # returns the etymology or ""
                        if affix_etymology:
                            morph.etymology = affix_etymology
                        else:
                            morph.etymology = ["ces"]
                    else:
                        # Inflectional affix
                        morph.etymology = ["ces"]
                else:
                    pass # keep blank etymology for punctuation, numbers etc.
    return sentences


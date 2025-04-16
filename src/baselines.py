from collections import defaultdict, Counter
from typing import List
from data_sentece import DataSentence,Word,Morph
from model import Model
import requests
import copy
import urllib.request
import urllib.error

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

def load_etym_dict(filepath: str) -> dict[str, list[str]]:
    """
    Loads a dictionary mapping words/entries to their etymological information.
    The file is expected to have lines in the format:
    
        <entry>\t<comma_separated_list_of_languages>
    
    Blank lines and lines not conforming to this format are skipped.
    """
    etymology = {}
    with open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                entry = parts[0].strip()
                langs_str = parts[1].strip()
                langs = [lang.strip() for lang in langs_str.split(",") if lang.strip()]
                etymology[entry] = langs
    return etymology

def load_affixes(afixes_file: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Loads affix dictionaries for prefixes and suffixes from a file.
    Each line is expected to be in the format:
      affix<TAB>language1,language2,...
    If the affix starts with '-', it is considered a suffix.
    If the affix ends with '-', it is considered a prefix.
    """
    prefixes = {}
    suffixes = {}
    with open(afixes_file, "rt", encoding="utf-8") as aff_file:
        for line in aff_file:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            affix, etyms = parts
            langs = [lang.strip() for lang in etyms.split(",") if lang.strip()]
            if affix.startswith("-"):
                # suffix
                suffixes[affix[1:].lower()] = langs
            elif affix.endswith("-"):
                # prefix
                prefixes[affix[:-1].lower()] = langs
            else:
                pass  # ignore any that don't match prefix/suffix pattern
    return prefixes, suffixes

def get_affix_etymology(morph_text: str, position:Morph.MorphPosition, prefixes: dict[str, list[str]], suffixes: dict[str, list[str]]) -> list[str]:
    """
    Returns the etymology for a prefix/suffix morph if found in the dictionaries.
    Otherwise returns an empty list. The logic is now based on morph_position:
      - If morph_position is 'PREFIX', we look it up in prefixes.
      - If morph_position is 'SUFFIX', we look it up in suffixes.
    """
    morph_lc = morph_text.lower()
    if position.name == "PREFIX":
        return prefixes.get(morph_lc, [])
    elif position.name == "SUFFIX":
        return suffixes.get(morph_lc, [])
    return []

class MorphDictModel(Model):
    """
    A rule-based model that assigns etymology based on root and affix dictionaries.
    
    For each alphabetic morph:
      - If the morph is a root (longer than 2 chars) found in the root dictionary,
        assign that root's etymology.
      - If it's a derivational affix, we check prefix/suffix dictionaries 
        based on morph.morph_position.
      - Otherwise, default to ["ces"].
    """
    def __init__(self, root_etym_file: str, affixes_file: str,binary:bool = False, name: str = "MorphDictModel") -> None:
        super().__init__(name)
        self.roots_etymology = load_etym_dict(root_etym_file)
        self.prefixes, self.suffixes = load_affixes(affixes_file)
        self.binary = binary
    def fit(self, data: List[DataSentence]) -> None:
        """No training needed for this rule-based approach."""
        pass

    def predict(self, data: List[DataSentence]) -> List[DataSentence]:
        predictions = copy.deepcopy(data)
        for sentence in predictions:
            for word in sentence.words:
                for morph in word.morphs:
                    if morph.text.isalpha():
                        if (morph.morph_type == morph.__class__.MorphType.ROOT
                            and len(morph.text) > 2
                            and (morph.text in self.roots_etymology or morph.text.lower() in self.roots_etymology)):
                            key = morph.text if morph.text in self.roots_etymology else morph.text.lower()
                            morph.etymology = self.roots_etymology[key]
                        elif morph.morph_type == morph.__class__.MorphType.DERIVATIONAL_AFFIX:
                            # Use position-based lookup in prefix/suffix dictionaries.
                            affix_etyms = get_affix_etymology(morph.text, morph.morph_position, self.prefixes, self.suffixes)
                            morph.etymology = affix_etyms if affix_etyms else ["ces"]
                        else:
                            morph.etymology = ["ces"]
                    if self.binary:
                        if morph.etymology != ["ces"]: 
                            morph.etymology = ["borrowed"]
        return predictions

class WordDictModel(Model):
    """
    A word-based model that assigns etymology using a word-level dictionary.
    
    The model uses the MorphoDiTa API to obtain lemmata for the entire text,
    then for each word it looks up etymology in a word-level dictionary.
    For derivational affixes, we rely on the prefix/suffix dictionaries 
    (using morph_position).
    If no dictionary entry is found, defaults to ["ces"].
    """
    class NetworkError(Exception):
        """Custom exception indicating a network-related failure."""
        pass    

    def __init__(self, word_etym_file: str, affixes_file: str,binary:bool = False, name: str = "WordDictModel") -> None:
        super().__init__(name)
        self.words_etymology = load_etym_dict(word_etym_file)
        self.prefixes, self.suffixes = load_affixes(affixes_file)
        self.binary = binary
    def fit(self, data: List[DataSentence]) -> None:
        """No training necessary for this word-level dictionary model."""
        pass

    @staticmethod
    def get_lemmata(text: str) -> List[str]:
        """
        Retrieves lemmata for the given text using the MorphoDiTa HTTP API.
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
        for sent in sentences:
            for token in sent:
                analyses = token.get("analyses", [])
                if analyses:
                    lemma = analyses[0].get("lemma", token.get("token", ""))
                else:
                    lemma = token.get("token", "")
                lexemes.append(lemma)
        return lexemes

    def predict(self, data: List[DataSentence]) -> List[DataSentence]:
        predictions = copy.deepcopy(data)
        
        # Build the full text from all sentences.
        whole_text = " ".join(sentence.sentence for sentence in predictions)
        try:
            lemmata = self.get_lemmata(whole_text)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            # URLError, HTTPError, or a socket-related OSError indicates network trouble
            raise WordDictModel.NetworkError(f"Network error contacting MorphoDiTa")        
        
        # Collect all words in the same order.
        all_words:list[Word] = []
        for sentence in predictions:
            all_words.extend(sentence.words)
        
        if len(lemmata) != len(all_words):
            raise ValueError(f"Mismatch: {len(lemmata)} lemmata vs {len(all_words)} words.")
        
        for word_obj, lemma in zip(all_words, lemmata):
            # Look up word-level etymology from dictionary. 
            word_etym = self.words_etymology.get(word_obj.text, [])
            if not word_etym:
                # If not found by surface form, try the lemma.
                word_etym = self.words_etymology.get(lemma, [])
            
            for morph in word_obj.morphs:
                if morph.text.isalpha():
                    # Assign the found etymology to root
                    if morph.morph_type == morph.__class__.MorphType.ROOT:
                        morph.etymology = word_etym if word_etym else ["ces"]
                        # Look up the etymology for derivational affixes
                    elif morph.morph_type == morph.__class__.MorphType.DERIVATIONAL_AFFIX:
                        affix_etyms = get_affix_etymology(morph.text, morph.morph_position,
                                                          self.prefixes, self.suffixes)
                        morph.etymology = affix_etyms if affix_etyms else ["ces"]
                    else:
                        morph.etymology = ["ces"]

                    if self.binary:
                        if morph.etymology != ["ces"]: 
                            morph.etymology = ["borrowed"]
        return predictions
    

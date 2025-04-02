from dataclasses import dataclass
import copy
from typing import List, Iterator
from enum import Enum


@dataclass
class Morph:
    class MorphType(Enum):
        ROOT = "root"
        DERIVATIONAL_AFFIX = "derivational affix"
        INFLECTIONAL_AFFIX = "inflectional affix"
        UNDEFINED = "undefined"

        @classmethod
        def _missing_(cls, value):
            if isinstance(value, str):
                value = value.upper()
            mapping = {
                "R": cls.ROOT,
                "D": cls.DERIVATIONAL_AFFIX,
                "I": cls.INFLECTIONAL_AFFIX
            }
            # if invalid type entered return undefined type
            return mapping.get(value, cls.UNDEFINED)
        
        def __str__(self):
            abbrevations = {
                "root": "R",
                "derivational affix": "D",
                "inflectional affix": "I",
                "undefined": "U"
            }
            return abbrevations[self.value]
    
    class MorphPosition(Enum):
        ROOT = "root"
        PREFIX = "prefix"
        INTERFIX = "interfix"
        SUFFIX = "suffix"
        UNDEFINED = "undefined"

        @classmethod
        def _missing_(cls, value):
            if isinstance(value, str):
                value = value.lower()
            mapping = {
                "root": cls.ROOT,
                "prefix": cls.PREFIX,
                'interfix': cls.INTERFIX,
                "suffix": cls.SUFFIX
            }
            return mapping.get(value, cls.UNDEFINED)

        def __str__(self):
            abbreviations = {
                "root": "R",
                "prefix": "P",
                'interfix': 'I',
                "suffix": "S",
                "undefined": "U"
            }
            return abbreviations[self.value]

    # The text of the morph.
    text: str = None
    # The etymology of the morph as a list of language codes.
    etymology: List[str] = None
    # The type of the morph
    morph_type: "Morph.MorphType" = None  

    morph_position: "Morph.MorphPosition" = None

    def __init__(self, text: str, etymology: List[str],
                 morph_type: "Morph.MorphType" = None,
                 position: "Morph.MorphPosition" = None):
        """
        Initialize a Morph instance.

        Args:
            text (str): The morph text.
            etymology (List[str]): A list of language codes (etymological info).
            morph_type (Morph.MorphType): The type of the morph. Defaults to UNDEFINED.
            position (Morph.MorphPosition): The position relative to the root.
                Use PREFIX for a prefix, SUFFIX for a suffix, or ROOT if this morph is the root.
                Defaults to UNDEFINED.
        """
        self.text = text
        self.etymology = etymology
        self.morph_type = morph_type if morph_type is not None else Morph.MorphType.UNDEFINED
        self.morph_position = position if position is not None else Morph.MorphPosition.UNDEFINED

    def __repr__(self):
        return (f"Morph(text='{self.text}', etymology={','.join(self.etymology)}, "
                f"morph_type='{self.morph_type.value}', position='{self.morph_position.value}')")

    def __str__(self):
        # A concise, user-friendly representation.
        etym = ",".join(self.etymology) if self.etymology else "No etymology"
        return f"{self.text} [{self.morph_type}, {self.morph_position}, {etym}]"
    
@dataclass
class Word:
    # A list of Morph objects that constitute the word.
    morphs: List[Morph]

    @property
    def text(self) -> str:
        """
        Constructs the word's string by concatenating the text of each morph.
        """
        return "".join(morph.text for morph in self.morphs)

    def __iter__(self) -> Iterator[Morph]:
        """
        Allows iteration over the morphs in the word.
        """
        return iter(self.morphs)

    def __repr__(self):
        return f"Word(text='{self.text}', morphs={self.morphs})"
    
    def __str__(self):
        return self.text

@dataclass
class DataSentence:
    # A list of Word objects that constitute the sentence.
    words: List[Word]

    @property
    def sentence(self) -> str:
        """
        Constructs the sentence string by concatenating the text of each word with spaces.
        """
        return " ".join(word.text for word in self.words)

    def __iter__(self) -> Iterator[Morph]:
        """
        Generator method that yields each Morph in the sentence in left-to-right order.
        """
        for word in self.words:
            for morph in word:
                yield morph

    @classmethod
    def from_data_sentence(cls, data_sentence: "DataSentence") -> "DataSentence":
        """
        Alternative constructor that creates a new DataSentence instance with the same data
        as the given instance. Since the morphs are mutable, a deep copy is performed on each morph.
        """
        new_words = [
            Word(morphs=[copy.deepcopy(morph) for morph in word.morphs])
            for word in data_sentence.words
        ]
        return cls(words=new_words)

    def __repr__(self):
        return f"DataSentence(sentence='{self.sentence}', words={self.words})"
    
    def __str__(self):
        return self.sentence


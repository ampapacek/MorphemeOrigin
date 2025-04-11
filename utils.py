import sys
import random
from typing import List, Tuple, Optional, Dict
from data_sentece import DataSentence,Word,Morph
from collections import Counter, defaultdict
def load_annotations(filepath: str, indent: int = 4) -> List[DataSentence]:
    """
    Reads an annotation file and returns a list of DataSentence objects.
    
    Expected file structure:
    
        <sentence text>
            <word text>
                <morph_text>\t<morph_type>\t<etymology_text>
                ...
            ...
        (Blank lines separate sentences)
    
    The etymology and morph_type fields are optional (defaulting to [] and UNDEFINED, respectively).
    Morph positions are deduced from the morph types:
      - If exactly one morph is marked as ROOT, morphs before it are PREFIX and after it are SUFFIX.
      - If multiple roots are present, morphs before the first root are PREFIX, morphs after the last root are SUFFIX,
        and any morphs between roots are marked as INTERFIX.
      - Otherwise, positions remain UNDEFINED.
    """
    sentences = []
    current_sentence_header = None  # Holds the sentence text.
    current_words = []              # List of words; each word is a list of Morph objects.
    current_word:list["Morph"] = None             # The current word's list of Morph objects.

    def flush_word():
        nonlocal current_word, current_words
        if current_word is not None:
            # Deduce morph positions based on morph types.
            root_indices = [i for i, m in enumerate(current_word) if m.morph_type == m.__class__.MorphType.ROOT]
            # If no root found set the first morph's position as root (usually prepositions and conjuctions of just one morph)
            if len(root_indices) == 0:
                root_indices = [0]
                # current_word[0].morph_type = Morph.MorphType.ROOT # optionaly change the morph type to root too
            if len(root_indices) == 1:
                root_index = root_indices[0]
                for i, m in enumerate(current_word):
                    if i < root_index:
                        m.morph_position = m.__class__.MorphPosition.PREFIX
                    elif i == root_index:
                        m.morph_position = m.__class__.MorphPosition.ROOT
                    else:
                        m.morph_position = m.__class__.MorphPosition.SUFFIX
            elif len(root_indices) > 1:
                first_root = root_indices[0]
                last_root = root_indices[-1]
                for i in range(0, first_root):
                    current_word[i].morph_position = current_word[i].__class__.MorphPosition.PREFIX
                for i in root_indices:
                    current_word[i].morph_position = current_word[i].__class__.MorphPosition.ROOT
                for i in range(first_root + 1, last_root):
                    if i not in root_indices:
                        current_word[i].morph_position = current_word[i].__class__.MorphPosition.INTERFIX
                for i in range(last_root + 1, len(current_word)):
                    current_word[i].morph_position = current_word[i].__class__.MorphPosition.SUFFIX
            elif len(root_indices) == 0:
                #no root - some mistake
                current_word_str = ""
                for morph in current_word: current_word_str += morph.text
                raise Exception ("No root found in word: ", current_word_str)
                # print ("\t".join(map(str,current_word)))
                # input()

            # Append the processed word and reset.
            current_words.append(current_word)
            current_word = None

    def flush_sentence():
        nonlocal current_sentence_header, current_words, current_word, sentences
        if current_sentence_header is not None:
            flush_word()  # Flush any remaining word.
            sentence_obj = DataSentence(words=[Word(morphs=w) for w in current_words])
            sentences.append(sentence_obj)
            current_sentence_header = None
            current_words = []

    with open(filepath, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line.strip():
                flush_sentence()
                continue
            if line.startswith('#'): # comments in annotation file
                continue
            stripped_line = line.lstrip()
            # Sentence line: no indentation.
            if not line.startswith(" "):
                flush_sentence()  # Finalize previous sentence.
                current_sentence_header = stripped_line
            # Word line: exactly indent spaces (e.g., 4 spaces) but not 8.
            elif line.startswith(" " * indent) and not line.startswith(" " * (2 * indent)):
                flush_word()  # Finalize the current word.
                current_word = []  # Start a new word.
            # Morph line: indent 2*indent spaces.
            elif line.startswith(" " * (2 * indent)):
                parts = stripped_line.split("\t")
                morph_text = parts[0] if len(parts) >= 1 else ""
                if len(parts) >= 3:
                    morph_type_field = parts[1].strip()
                    morph_type = Morph.MorphType(morph_type_field)
                    etymology_field = parts[2].strip()
                    morph_etymology = [code.strip() for code in etymology_field.split(',')] if etymology_field else []
                elif len(parts) == 2 and parts[1].strip():
                    etymology_field = parts[1].strip()
                    morph_etymology = [code.strip() for code in etymology_field.split(',')] if etymology_field else []
                    morph_type = Morph.MorphType.UNDEFINED
                else:
                    morph_etymology = []
                    morph_type = Morph.MorphType.UNDEFINED

                if current_word is None:
                    current_word = []
                current_word.append(Morph(morph_text, morph_etymology, morph_type))
            else:
                # Ignore any lines with unexpected indentation.
                pass

    flush_sentence()  # Final flush if file doesn't end with a blank line.
    return sentences

def fill_classification(filepath: str, sentences: List[DataSentence]) ->List[DataSentence]:
    """
    Loads word classification from a file and fills in the classification for each morph.
    Returns the input sentences with the classification filled in for each morph.
    
    The file is expected to have lines in the following format:
    
        segmented_word[TAB]code1 code2 ... codeN
    
    For example:
        Tř i krát    R D D
        rychl ejš í  R D I
        než         R
        slov o      R I
    
    The segmented_word is the word split into its morph segments by spaces. The classification
    codes (e.g., R, D, I) correspond one-to-one with the morphs of the word.
    
    Args:
        filepath (str): Path to the classification file.
        sentences (List[DataSentence]): List of sentences containing Word and Morph objects.
    
    Returns:
        List[DataSentence]: The input sentences with the classification filled in for each morph.
    Raises:
        ValueError: If there is any mismatch between the file and the sentence data.
    """
    # Mapping from classification code letter to MorphType enum.
    code_to_morph_type = {
        "R": Morph.MorphType.ROOT,
        "D": Morph.MorphType.DERIVATIONAL_AFFIX,
        "I": Morph.MorphType.INFLECTIONAL_AFFIX,
        "U": Morph.MorphType.UNDEFINED,
    }
       
    # Read all non-empty lines from the file.
    with open(filepath, 'rt', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Collect all non-punctuation words from the sentences.
    words_from_sentences:List[Word] = []
    for sentence in sentences:
        for word in sentence.words:
            if word.text.isalnum():
                words_from_sentences.append(word)

    # Process each line and fill in the classification.
    for line, word in zip(lines, words_from_sentences):
        # Split line into the segmented word and the classification codes.
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"Line '{line}' is not in the expected format (segmented word[TAB]codes).")
        segmented_word = parts[0].strip()
        codes_str = parts[1].strip()
        
        # Reconstruct the word from the segmented form (by removing spaces).
        reconstructed_word = segmented_word.replace(" ", "")
        
        # Check that the reconstructed word matches the word's text.
        if reconstructed_word != word.text:
            raise ValueError(f"Word mismatch: classification file word '{reconstructed_word}' does not match "
                             f"sentence word '{word.text}'.")
        
        # Get the classification for each morph
        codes = codes_str.split()
        
        # Check that the number of codes equals the number of morphs in the word.
        if len(codes) != len(word.morphs):
            raise ValueError(f"Number of classification codes ({len(codes)}) does not match number of morphs "
                             f"({len(word.morphs)}) for word '{word.text}'.")
        
        # Assign the classification (morph type) to each morph.
        for morph, code in zip(word.morphs, codes):
            if code not in code_to_morph_type:
                raise ValueError(f"Unknown classification code '{code}' for word '{word.text}'.")
            morph.morph_type = code_to_morph_type[code]
    return sentences

def write_morph_statistics(target_sentences: List["DataSentence"], languages_file: str = None, morphs_file: str = None) -> None:
    """
    Computes morphological statistics from the given list of sentences and 
    writes them to the specified files for languages and morphs.
    
    Args:
        target_sentences: A list of DataSentence objects.
        languages_file: Path to the file where language occurrences will be written. 
                       If None, no file is written.
        morphs_file: Path to the file where morph occurrences will be written. 
                     If None, no file is written.
                     
    Note:
        This function no longer returns anything; it simply writes files 
        with stats (if the file paths are provided).
    """
    morph_count = 0
    morphs = defaultdict(Counter)
    languages = Counter()
    word_count = 0

    for sentence in target_sentences:
        for word in sentence.words:
            # Skip non-alphabetic single-morph words
            if not word.text.isalpha() and len(word.morphs) == 1:
                continue
            word_count += 1
            for morph in word:
                # Consider only morphs with non-empty text and etymology.
                if morph.text and morph.etymology:
                    morph_count += 1
                    etym = ",".join(morph.etymology)
                    morphs[morph.text][etym] += 1
                    languages[etym] += 1

    if languages_file is not None:
        with open(languages_file, 'wt') as lang_f:
            for language, count in languages.most_common():
                print(f"{language}\t{count}", file=lang_f)

    if morphs_file is not None:
        with open(morphs_file, 'wt') as morphs_f:
            lines = []
            # Sort by descending sum of morph counts
            for morph_text, etymology_counter in sorted(
                morphs.items(), key=lambda item: sum(item[1].values()), reverse=True
            ):
                if len(etymology_counter) > 1:
                    print(f"{morph_text}\t{dict(etymology_counter)}", file=morphs_f)
                else:
                    lines.append(f"{morph_text}\t{dict(etymology_counter)}")
            for line in lines:
                print(line, file=morphs_f)
                
def count_sentences_words_morphs(sentences:List["DataSentence"]):
    sentence_count = len(sentences)
    word_count = 0
    morph_count = 0
    for sentence in sentences:
        word_count += len(sentence.words)
        morph_count += sentence.morph_count
    return sentence_count,word_count,morph_count

def split(data: List, ratio: float = 0.2, random_seed:int = None) -> Tuple[List[DataSentence], List[DataSentence]]:
    """
    Splits the input data into two random parts based on the provided ratio.
    
    The second part contains approximately the given ratio of the total items.
    """
    data_copy = data.copy()
    random.seed(random_seed)
    random.shuffle(data_copy)
    split_index = int(len(data_copy) * (1 - ratio))
    return data_copy[:split_index], data_copy[split_index:]

def relative_error_reduction(baseline_f1: float, new_f1: float) -> float:
    """
    Calculate the relative error reduction improvement metric on a 0–100 scale.
    
    The error is defined as (100 - F1). The improvement metric is the percentage
    reduction of errors relative to the baseline:
    
        improvement = 100 * (baseline_error - new_error) / baseline_error
                    = 100 * ((100 - baseline_f1) - (100 - new_f1)) / (100 - baseline_f1)
                    = 100 * (new_f1 - baseline_f1) / (100 - baseline_f1)
    
    Args:
        baseline_f1 (float): The baseline F1 score (in percentage).
        new_f1 (float): The new F1 score (in percentage).
        
    Returns:
        float: The improvement metric, where 0 means no improvement over baseline and 
               100 means perfect prediction (i.e. baseline errors completely eliminated).
               
    Raises:
        ValueError: If either F1 score is outside the range [0, 100] or if baseline_f1 is 100.
    """
    if not (0 <= baseline_f1 <= 100) or not (0 <= new_f1 <= 100):
        raise ValueError("F1 scores must be between 0 and 100.")
    
    if baseline_f1 == 100:
        raise ValueError("Baseline F1 cannot be 100, as improvement metric is undefined.")
    
    improvement = 100 * (new_f1 - baseline_f1) / (100 - baseline_f1)
    return improvement

def remove_targets(sentences: List[DataSentence]) -> List[DataSentence]:
    """
    Removes etymology targets by creating a deep copy of the sentences
    and setting each morph's etymology to an empty list.
    """
    sentences_deep_copy = list(map(DataSentence.from_data_sentence, sentences))
    for sentence in sentences_deep_copy:
        for morph in sentence:
            morph.etymology = []
    return sentences_deep_copy

def pprint_sentences(sentences: List[DataSentence], filename: Optional[str] = None, indent: int = 4) -> None:
    """
    Writes the sentences in a nicely indented format, with each morph on a separate line.
    
    If a filename is provided, the output is written to that file; otherwise, it is printed to the console.
    
    The output structure is:
      <sentence text>
          <word text>
              <morph text>    <morph type>    <comma-separated etymology>
          <word text>
              ...
    
    Args:
      sentences: A list of DataSentence objects.
      filename: The output file path. If None, the output is printed to the console.
      indent: Number of spaces per indentation level (default is 4).
    """
    # Determine output stream: file or stdout.
    if filename:
        out_stream = open(filename, "wt", encoding="utf-8")
        close_stream = True
    else:
        out_stream = sys.stdout
        close_stream = False

    try:
        for sentence in sentences:
            out_stream.write(sentence.sentence + "\n")
            for word in sentence.words:
                # Only print words that are alphanumeric.
                if word.text.isalnum():
                    out_stream.write(" " * indent + word.text + "\n")
                    for morph in word.morphs:
                        # Write each morph with an indent level of 2.
                        out_stream.write(" " * (indent * 2) + morph.text +
                                         "\t" + str(morph.morph_type) +
                                         "\t" + ",".join(morph.etymology) + "\n")
            out_stream.write("\n")
    finally:
        if close_stream:
            out_stream.close()

def single_morph_sentences_from_dict(morphs_etymology_dict_path: str) -> List[DataSentence]:
    """
    Reads an etymology dictionary file and builds a list of single-morph sentences.
    
    Each line in the file should be in the format:
        <entry>\t<comma_separated_etymologies>
    where <entry> can be:
      - A prefix (ending with '-') 
      - A suffix (starting with '-')
      - An inflectional suffix (starting with '~')
      - An interfix (starting with '$')
      - Otherwise, treated as a root (with morph position MorphPosition.ROOT)
    
    The function creates a single DataSentence per line, containing one Word
    with one Morph. The etymology is parsed from the comma-separated values.
    
    Args:
        morphs_etymology_dict_path: Path to the dictionary file containing morph entries.
    
    Returns:
        A list of DataSentence objects, each representing a single-morph sentence.
    """
    sentences: List[DataSentence] = []
    
    with open(morphs_etymology_dict_path, "rt", encoding="utf-8") as file_in:
        for line in file_in:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) != 2:
                # Skip lines not matching the expected format
                continue

            morph_text = parts[0].strip()
            etymology_str = parts[1].strip()
            etymologies = [lang.strip() for lang in etymology_str.split(",") if lang.strip()]

            # Determine morph position and type
            if morph_text.startswith('-'):
                # It's a suffix: remove leading '-'
                morph_text = morph_text[1:]
                morph_position = Morph.MorphPosition.SUFFIX
                morph_type = Morph.MorphType.DERIVATIONAL_AFFIX
            elif morph_text.endswith('-'):
                # It's a prefix: remove trailing '-'
                morph_text = morph_text[:-1]
                morph_position = Morph.MorphPosition.PREFIX
                morph_type = Morph.MorphType.DERIVATIONAL_AFFIX
            elif morph_text.startswith('~'):
                # It's an inflectional suffix: remove leading '~'
                morph_text = morph_text[1:]
                morph_position = Morph.MorphPosition.SUFFIX
                morph_type = Morph.MorphType.INFLECTIONAL_AFFIX
            elif morph_text.startswith('$'):
                # It's an interfix: remove leading '$'
                morph_text = morph_text[1:]
                morph_position = Morph.MorphPosition.INTERFIX
                morph_type = Morph.MorphType.DERIVATIONAL_AFFIX
            elif morph_text.isalpha():
                # Otherwise, treat as root
                morph_position = Morph.MorphPosition.ROOT
                morph_type = Morph.MorphType.ROOT
            else:
                # Invalid line
                continue
            
            # Build a single-morph Word in a single-sentence DataSentence
            single_morph = Morph(
                text=morph_text,
                etymology=etymologies,
                morph_type=morph_type,
                position=morph_position
            )
            single_word = Word([single_morph])
            single_sentence = DataSentence(words=[single_word])
            sentences.append(single_sentence)
    
    return sentences

def calculate_cohen_kappa(sentences1: List[DataSentence], sentences2: List[DataSentence]) -> float:
    """
    Computes Cohen's kappa for inter-annotator agreement on etymology annotations.
    
    For each morph in corresponding DataSentence objects, the etymology (a list of strings)
    is converted to a sorted tuple to represent a categorical label. The function then 
    computes the observed agreement (proportion of morphs for which the labels agree) and 
    the expected agreement (based on the marginal label distributions), and returns the kappa value.
    
    Args:
        sentences1 (List[DataSentence]): Annotated sentences from annotator 1.
        sentences2 (List[DataSentence]): Annotated sentences from annotator 2.
    
    Returns:
        float: The Cohen's kappa value.
    
    Raises:
        AssertionError: If corresponding sentences, words, or morph texts do not align.
    """
    labels1 = []
    labels2 = []
    
    # Iterate over aligned sentences.
    for s1, s2 in zip(sentences1, sentences2):
        # Ensure sentences match.
        assert s1.sentence == s2.sentence, f"Sentence mismatch: {s1.sentence} != {s2.sentence}"
        for w1, w2 in zip(s1.words, s2.words):
            assert w1.text == w2.text, f"Word mismatch: {w1.text} != {w2.text}"
            for m1, m2 in zip(w1.morphs, w2.morphs):
                # Check that the morph texts match.
                assert m1.text == m2.text, f"Morph text mismatch: {m1.text} != {m2.text}"
                # Convert the etymology lists to sorted tuples (empty tuple for no etymology).
                label1 = tuple(sorted(m1.etymology))
                label2 = tuple(sorted(m2.etymology))
                labels1.append(label1)
                labels2.append(label2)
    
    if not labels1:
        # If no morphs were evaluated, return perfect agreement.
        return 1.0

    total = len(labels1)
    # Observed agreement: proportion of morphs with identical labels.
    observed_agreement = sum(1 for a, b in zip(labels1, labels2) if a == b) / total

    # Compute marginal frequencies for annotator 1 and annotator 2.
    freq1 = {}
    freq2 = {}
    for label in labels1:
        freq1[label] = freq1.get(label, 0) + 1
    for label in labels2:
        freq2[label] = freq2.get(label, 0) + 1

    # Expected agreement: sum over all labels of the product of marginal probabilities.
    expected_agreement = 0.0
    all_labels = set(freq1.keys()) | set(freq2.keys())
    for label in all_labels:
        p1 = freq1.get(label, 0) / total
        p2 = freq2.get(label, 0) / total
        expected_agreement += p1 * p2

    if expected_agreement == 1:
        return 1.0  # Avoid division by zero 

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa

def evaluate(
    sentences_prediction: List["DataSentence"],
    sentences_target: List["DataSentence"],
    standard_eval: bool = True,
    native_borrowed_eval: bool = False,
    group_by_text_eval: bool = False,
    file_mistakes: Optional[str] = None
) -> Dict[str, float]:
    """
    Performs the evaluation, collecting metrics in three possible ways:
    
    1) Standard (macro) F1 across all morphs + percentage of fully correct morphs
       - Enabled by standard_eval=True, default = True
    2) Native vs. Borrowed F1 (target = {"ces"} vs. otherwise). Compute separately for native and for borrowed.
       - Enabled by native_borrowed_eval=True
    3) Group by `morph.text`, average F1 per unique morphs (just by text) and compute the average score in each group,
        then average across all groups (unique morphs).
       - Enabled by group_by_text_eval=True

    The function returns a dictionary whose keys depend on which evaluations are enabled.
    Example keys:
      "f1score", "accuracy",
      "native_f1", "borrowed_f1",
      "grouped_fscore".

    If file_mistakes is provided, logs mistakes (a morph is a 'mistake' if its F1 != 1.0).

    Returns:
        A dictionary with any requested metrics:
          {
            "f1score": float,
            "accuracy": float,
            "native_f1": float,
            "borrowed_f1": float,
            "grouped_fscore": float,
          }
        (keys only present if that evaluation is enabled)
    """

    # For mistakes logging (applies to standard approach)
    mistakes_f = open(file_mistakes, 'wt') if file_mistakes else None
    if mistakes_f and standard_eval:
        print("word\tmorph\tprediction\ttarget", file=mistakes_f)

    # (1) Standard approach
    total_f1_sum = 0.0
    morph_count = 0
    mistakes_count = 0

    # (2) Native vs. Borrowed
    native_f1_sum = 0.0
    native_count = 0
    borrowed_f1_sum = 0.0
    borrowed_count = 0

    # (3) Group by text
    if group_by_text_eval:
        f1_accumulators = defaultdict(lambda: {"sum": 0.0, "count": 0})

    for sent_pred, sent_tgt in zip(sentences_prediction, sentences_target):
        assert sent_pred.sentence == sent_tgt.sentence, \
            f"Sentence mismatch: {sent_pred.sentence} != {sent_tgt.sentence}"

        for word_pred, word_tgt in zip(sent_pred.words, sent_tgt.words):
            for morph_pred, morph_tgt in zip(word_pred, word_tgt):
                # Ensure morph text matches
                assert morph_pred.text == morph_tgt.text, \
                    f"Morph text mismatch: {morph_pred.text} != {morph_tgt.text}"

                pred_set = set(morph_pred.etymology)
                tgt_set = set(morph_tgt.etymology)

                # If target is empty, skip (punct, numeric, etc.)
                if not tgt_set:
                    continue

                # Compute F1
                intersection = pred_set.intersection(tgt_set)
                precision = len(intersection) / len(pred_set) if pred_set else 0
                recall = len(intersection) / len(tgt_set) if tgt_set else 0
                denom = precision + recall
                f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

                # (1) Standard approach: macro average across all morphs
                if standard_eval:
                    total_f1_sum += f1
                    morph_count += 1
                    if f1 != 1.0:
                        mistakes_count += 1
                        if mistakes_f:
                            print(
                                f"{word_tgt.text}\t{morph_tgt.text}\t{morph_pred.etymology}\t{morph_tgt.etymology}",
                                file=mistakes_f
                            )

                # (2) native vs. borrowed
                if native_borrowed_eval:
                    is_native = (tgt_set == {"ces"})
                    if is_native:
                        native_f1_sum += f1
                        native_count += 1
                    else:
                        borrowed_f1_sum += f1
                        borrowed_count += 1

                # (3) grouping by morph text
                if group_by_text_eval:
                    text = morph_tgt.text  # same as morph_pred.text
                    f1_accumulators[text]["sum"] += f1
                    f1_accumulators[text]["count"] += 1

    # Close mistakes file if open
    if mistakes_f:
        mistakes_f.close()

    # Prepare results
    results = {}

    # (1) Compute standard if needed
    if standard_eval and morph_count > 0:
        standard_fscore = 100.0 * total_f1_sum / morph_count
        standard_accuracy = 100.0 * (morph_count - mistakes_count) / morph_count
        results["f1score"] = standard_fscore
        results["accuracy"] = standard_accuracy

    # (2) Compute native vs borrowed if needed
    if native_borrowed_eval:
        # If no native or borrowed morphs, treat that as 100% by default
        if native_count > 0:
            native_f1 = 100.0 * native_f1_sum / native_count
        else:
            native_f1 = 100.0

        if borrowed_count > 0:
            borrowed_f1 = 100.0 * borrowed_f1_sum / borrowed_count
        else:
            borrowed_f1 = 100.0

        results["native_f1"] = native_f1
        results["borrowed_f1"] = borrowed_f1

    # (3) Group by morph text
    if group_by_text_eval and f1_accumulators:
        text_level_averages = []
        for morph_text, vals in f1_accumulators.items():
            if vals["count"] > 0:
                avg_f1 = vals["sum"] / vals["count"]
                text_level_averages.append(avg_f1)
        if text_level_averages:
            grouped_fscore = 100.0 * sum(text_level_averages) / len(text_level_averages)
        else:
            grouped_fscore = 0.0
        results["grouped_fscore"] = grouped_fscore

    return results


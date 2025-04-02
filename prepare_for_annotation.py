#!/usr/bin/env python3
"""
This script reads an annotation file in TSV format and outputs the data in an indented format.

Expected input format (TSV):
  <sentence>\t<segmentation>

Where:
  - <sentence> is the full sentence text.
  - <segmentation> is a series of morph units separated by spaces. A token that does not start with "@@"
    indicates the start of a new word, while tokens starting with "@@" belong to the current word.

The output format will be:

<sentence text>
    <reconstructed word>
        <morph text>
        <morph text>
    <reconstructed word>
        <morph text>
        ...

Usage example:
    python make_anotable_indent.py --input anotations/ces_dev_sentences.tsv --output anotations/dev_for_annotation.tsv
"""

import argparse

def setup_parser() -> argparse.ArgumentParser:
    """
    Sets up the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser with a descriptive epilog.
    """
    parser = argparse.ArgumentParser(
        description="Convert an annotation TSV file to a nicely indented format for morph annotation.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-i', "--input",
        type=str,
        default="anotations/ces_dev_sentences.tsv",
        help="Path to the input TSV file (default: anotations/ces_dev_sentences.tsv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="anotations/dev_for_annotation.txt",
        help="Path to the output file (default: anotations/dev_for_annotation.txt)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug output."
    )
    return parser

def process_annotations(input_file: str, output_file: str, verbose: bool = False, debug: bool = False) -> None:
    """
    Reads the annotation TSV file and writes an indented version to the output file.
    
    Indentation levels:
      - Level 0 (no indent): Sentence text.
      - Level 1 (4 spaces): Word.
      - Level 2 (8 spaces): Morph text.
    
    Args:
      input_file: Path to the input TSV file.
      output_file: Path to the output file.
      verbose: If True, prints additional informational messages.
      debug: If True, prints debugging messages.
    """
    INDENT = 4  # Number of spaces per indentation level.
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        # Process each line of the input file.
        for line in infile:
            line = line.strip()
            if not line:
                continue  # Skip empty lines.
            
            parts = line.split("\t")
            if len(parts) != 2:
                if debug:
                    print(f"Skipping malformed line: {line}")
                continue
            
            sentence = parts[0]
            segmentation = parts[1]
            
            # Write the sentence (level 0).
            outfile.write(f"{sentence}\n")
            
            # Split the segmentation into tokens.
            tokens = segmentation.split()
            current_morphs = []  # Collect morph tokens for the current word.
            
            for token in tokens:
                # A token that does not start with "@@" signals the start of a new word.
                if not token.startswith("@@"):
                    if current_morphs:
                        # Flush the previous word.
                        word = "".join(current_morphs).replace("@@", "")
                        if word.isalnum():
                            outfile.write(" " * INDENT + f"{word}\n")
                            for morph_token in current_morphs:
                                morph_text = morph_token.replace("@@", "")
                                outfile.write(" " * (INDENT * 2) + f"{morph_text}\n")
                        else:
                            if debug:
                                print(f"Skipping non-alphanumeric word: {word}")
                    # Start a new word.
                    current_morphs = [token]
                else:
                    # Token belongs to the current word.
                    current_morphs.append(token)
            
            # Flush the last word if any tokens remain.
            if current_morphs:
                word = "".join(current_morphs).replace("@@", "")
                if word.isalnum():
                    outfile.write(" " * INDENT + f"{word}\n")
                    for morph_token in current_morphs:
                        morph_text = morph_token.replace("@@", "")
                        outfile.write(" " * (INDENT * 2) + f"{morph_text}\n")
                else:
                    if debug:
                        print(f"Skipping non-alphanumeric word: {word}")
            
            # Add a blank line after each sentence block.
            outfile.write("\n")

def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.debug: args.verbose = True # Enable verbose output automaticaly if debug is enabled.
    
    if args.verbose:
        print(f"Processing annotations from '{args.input}'...")
    
    try:
        process_annotations(args.input, args.output, args.verbose, args.debug)
    except Exception as e:
        print(f"Error processing annotations: {e}")
        return
    
    if args.verbose:
        print(f"Annotations successfully written to '{args.output}'.")

if __name__ == "__main__":
    main()
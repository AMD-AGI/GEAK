# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
Utilities for code parsing, diffing, and manipulation
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


# ---------------------------------------------------------------------------
# Multi-file diff support
# ---------------------------------------------------------------------------

def extract_multifile_diffs(diff_text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract diff blocks that carry a ``file:path`` annotation.

    Format expected::

        <<<<<<< SEARCH file:kernel.py
        old code
        =======
        new code
        >>>>>>> REPLACE

    Args:
        diff_text: LLM response containing one or more SEARCH/REPLACE blocks.

    Returns:
        Dictionary mapping file paths to lists of (search_text, replace_text).
        If no ``file:`` prefix is found on any block, returns an empty dict
        (caller should fall back to single-file ``extract_diffs``).
    """
    # Pattern: <<<<<<< SEARCH file:some/path.py
    pattern = r"<<<<<<< SEARCH\s+file:(\S+)\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    matches = re.findall(pattern, diff_text, re.DOTALL)

    if not matches:
        return {}

    result: Dict[str, List[Tuple[str, str]]] = {}
    for file_path, search_text, replace_text in matches:
        file_path = file_path.strip()
        result.setdefault(file_path, []).append(
            (search_text.rstrip(), replace_text.rstrip())
        )
    return result


def apply_multifile_diff(
    files: Dict[str, str], diff_text: str, default_file: Optional[str] = None
) -> Dict[str, str]:
    """
    Apply SEARCH/REPLACE diffs to a dictionary of files.

    Supports two modes:

    1. **Multi-file**: Diffs carry ``file:path`` annotations.  Each diff is
       applied only to the specified file.
    2. **Single-file fallback**: If no ``file:`` annotations are found, all
       diffs are applied to *default_file* (or the first file if not given).

    Args:
        files: Dictionary mapping relative file paths to their contents.
        diff_text: Raw LLM response with SEARCH/REPLACE blocks.
        default_file: Fallback file for single-file diffs.

    Returns:
        Updated files dictionary (new dict; originals are not mutated).
    """
    updated = dict(files)  # shallow copy

    multifile_diffs = extract_multifile_diffs(diff_text)

    if multifile_diffs:
        # Multi-file mode
        for file_path, diffs in multifile_diffs.items():
            if file_path not in updated:
                logger.warning(
                    f"Diff targets file '{file_path}' which does not exist. "
                    f"Available files: {list(updated.keys())}"
                )
                continue
            code = updated[file_path]
            for search_text, replace_text in diffs:
                search_lines = search_text.split("\n")
                replace_lines = replace_text.split("\n")
                code_lines = code.split("\n")
                applied = False
                for i in range(len(code_lines) - len(search_lines) + 1):
                    if code_lines[i : i + len(search_lines)] == search_lines:
                        code_lines[i : i + len(search_lines)] = replace_lines
                        applied = True
                        break
                if not applied:
                    logger.warning(
                        f"Could not apply diff to '{file_path}': search text not found"
                    )
                code = "\n".join(code_lines)
            updated[file_path] = code
    else:
        # Single-file fallback: apply all diffs to default_file
        target = default_file or (sorted(updated.keys())[0] if updated else None)
        if target and target in updated:
            updated[target] = apply_diff(updated[target], diff_text)
        elif target:
            logger.warning(f"Default file '{target}' not found in files dict")

    return updated


def format_multifile_diff(file_path: str, search: str, replace: str) -> str:
    """
    Generate a properly formatted multi-file SEARCH/REPLACE block.

    Args:
        file_path: Relative path of the target file.
        search: The original code to find.
        replace: The replacement code.

    Returns:
        Formatted diff block string.
    """
    return (
        f"<<<<<<< SEARCH file:{file_path}\n"
        f"{search}\n"
        f"=======\n"
        f"{replace}\n"
        f">>>>>>> REPLACE"
    )


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"

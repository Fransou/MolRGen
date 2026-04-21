"""Molecular property verifier for property prediction tasks.

This module provides the MolPropVerifier class which computes rewards for
molecular property prediction tasks, supporting both regression and
classification objectives.
"""

import logging
import re
from typing import List, Optional

import numpy as np

from molrgen.reward.verifiers.abstract_verifier import (
    Verifier,
)
from molrgen.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
)
from molrgen.reward.verifiers.mol_prop_reward.input_metadata import (
    MolPropVerifierInputMetadataModel,
)
from molrgen.reward.verifiers.mol_prop_reward.mol_prop_verifier_pydantic_model import (
    MolPropVerifierConfigModel,
    MolPropVerifierMetadataModel,
    MolPropVerifierOutputModel,
)


class MolPropVerifier(Verifier):
    """Verifier for molecular property prediction tasks.

    This verifier computes rewards for property prediction based on how close
    the predicted value is to the ground truth. It supports both regression
    tasks (using normalized squared error) and classification tasks (using
    exact match accuracy).

    Attributes:
        verifier_config: Configuration for the property verifier.
        logger: Logger instance for the verifier.

    Example:
        ```python
        from molrgen.reward.verifiers import (
            MolPropVerifier,
            MolPropVerifierConfigModel,
            BatchVerifiersInputModel,
            MolPropVerifierInputMetadataModel
        )

        config = MolPropVerifierConfigModel(reward="property")
        verifier = MolPropVerifier(config)

        inputs = BatchVerifiersInputModel(
            completions=["<answer>0.75</answer>"],
            metadatas=[MolPropVerifierInputMetadataModel(
                objectives=["regression"],
                target=[0.8],
                norm_var=0.1
            )]
        )
        results = verifier.get_score(inputs)
        ```
    """

    # ==========================================================================
    # Regex patterns for number extraction
    # ==========================================================================

    # Base patterns (building blocks)
    SIGN_PATTERN = r"[-−+]?"
    INTEGER_PATTERN = r"\d+"
    DECIMAL_PATTERN = r"(?:\.\d+)?"
    BASE_FLOAT_PATTERN = rf"{SIGN_PATTERN}{INTEGER_PATTERN}{DECIMAL_PATTERN}"

    # Multiplication symbol pattern: "x", "×", or "\times" (single or double backslash)
    MULT_PATTERN = r"(?:\\{1,2}times|[x×])"

    # Unicode superscript digits and signs for patterns like "10⁻⁶"
    UNICODE_SUPERSCRIPT_SIGN = r"[⁺⁻]?"
    UNICODE_SUPERSCRIPT_DIGITS = r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+"

    # Pattern for scientific notation with <sup> tags: "1.0 x 10<sup>-6</sup>"
    SCI_SUP_PATTERN = (
        rf"{BASE_FLOAT_PATTERN}\s*[x×]\s*10\s*<sup>{BASE_FLOAT_PATTERN}</sup>"
    )

    # Pattern for Unicode superscript notation: "2.1 × 10⁻⁶"
    SCI_UNICODE_PATTERN = rf"{BASE_FLOAT_PATTERN}\s*{MULT_PATTERN}\s*10{UNICODE_SUPERSCRIPT_SIGN}{UNICODE_SUPERSCRIPT_DIGITS}"

    # Pattern for LaTeX scientific notation: "1.3 \times 10^{-5}" or "1.7 × 10^{-4}"
    SCI_LATEX_PATTERN = (
        rf"{BASE_FLOAT_PATTERN}\s*{MULT_PATTERN}\s*10\s*\^\s*\{{{BASE_FLOAT_PATTERN}\}}"
    )

    # Pattern for caret notation without braces: "1.3 × 10^-4" or "1.6 x 10^{-5}"
    SCI_CARET_PATTERN = rf"{BASE_FLOAT_PATTERN}\s*{MULT_PATTERN}\s*10\s*\^?\s*\{{?{BASE_FLOAT_PATTERN}\}}?"

    # Pattern for plus-minus notation: "-2.1 ± 0.5" or "1.3 +- 4"
    PM_PATTERN = rf"{BASE_FLOAT_PATTERN}\s*(?:±|\+-|\+/-)\s*{BASE_FLOAT_PATTERN}"

    # Pattern for standard scientific notation: "1e-3", "1.5E+6"
    SCI_E_PATTERN = rf"{BASE_FLOAT_PATTERN}[eE]{BASE_FLOAT_PATTERN}"

    # Pattern for plain float: "1.0" or "-2.5"
    FLOAT_PATTERN = BASE_FLOAT_PATTERN

    # Pattern for percentage: "14%" should be matched and converted to 0.14
    PERCENTAGE_PATTERN = rf"{BASE_FLOAT_PATTERN}%"

    # Combined number pattern (order matters: most specific first)
    NUM_PATTERN = rf"(?:{SCI_SUP_PATTERN}|{SCI_UNICODE_PATTERN}|{SCI_LATEX_PATTERN}|{SCI_CARET_PATTERN}|{PM_PATTERN}|{SCI_E_PATTERN}|{FLOAT_PATTERN})"

    # Pattern with boundary checks to exclude numbers with slashes or preceded by ^ (possibly between {})
    NUM_WITH_BOUNDARY_PATTERN = rf"(?<![/\w.\^+−-])(?<!\^\{{)(?<!\^\()({SCI_SUP_PATTERN}|{SCI_UNICODE_PATTERN}|{SCI_LATEX_PATTERN}|{SCI_CARET_PATTERN}|{PM_PATTERN}|{SCI_E_PATTERN}|{PERCENTAGE_PATTERN}|{FLOAT_PATTERN})(?![%\d/])(?!°C)"

    # ==========================================================================
    # Extraction patterns (with capture groups for parsing)
    # ==========================================================================

    # Extraction pattern for standard scientific notation: captures base and exponent
    SCI_E_EXTRACT = rf"({BASE_FLOAT_PATTERN})[eE]({BASE_FLOAT_PATTERN})"

    # Extraction pattern for <sup> notation: captures base and exponent
    SCI_SUP_EXTRACT = (
        rf"({BASE_FLOAT_PATTERN})\s*[x×]\s*10\s*<sup>({BASE_FLOAT_PATTERN})</sup>"
    )

    # Extraction pattern for Unicode superscript: captures base and exponent
    SCI_UNICODE_EXTRACT = rf"({BASE_FLOAT_PATTERN})\s*{MULT_PATTERN}\s*10({UNICODE_SUPERSCRIPT_SIGN}{UNICODE_SUPERSCRIPT_DIGITS})"

    # Extraction pattern for LaTeX notation: captures base and exponent
    SCI_LATEX_EXTRACT = rf"({BASE_FLOAT_PATTERN})\s*{MULT_PATTERN}\s*10\s*\^\s*\{{({BASE_FLOAT_PATTERN})\}}"

    # Extraction pattern for caret notation: captures base and exponent (with optional braces)
    SCI_CARET_EXTRACT = rf"({BASE_FLOAT_PATTERN})\s*{MULT_PATTERN}\s*10\s*\^?\s*\{{?({BASE_FLOAT_PATTERN})\}}?"

    # Extraction pattern for plus-minus: captures central value
    PM_EXTRACT = rf"({BASE_FLOAT_PATTERN})\s*(?:±|\+-|\+/-)\s*{BASE_FLOAT_PATTERN}"

    # ==========================================================================
    # Classification answer keywords
    # ==========================================================================

    CLASSIFICATION_TRUE_KEYWORDS = [
        "true",
        "yes",
        "1",
        "1.0",
        "high",
        "highly",
        "likely",
        "y",
    ]
    CLASSIFICATION_FALSE_KEYWORDS = ["false", "no", "0", "0.0", "low", "poor", "n"]

    def __init__(self, verifier_config: MolPropVerifierConfigModel) -> None:
        """Initialize the MolPropVerifier.

        Args:
            verifier_config: Configuration containing reward type settings.
        """
        super().__init__(verifier_config)
        self.verifier_config: MolPropVerifierConfigModel = verifier_config
        self.logger = logging.getLogger("MolPropVerifier")

    def _parse_float_with_sup(self, s: str) -> float:
        """Parse a float that may include scientific notation, percentages, or other formats.

        Handles patterns like:
        - "1.0 x 10<sup>-6</sup>" -> 1.0e-6
        - "1.3 \times 10^{-5}" -> 1.3e-5
        - "3.97 \\times 10^{-5}" -> 3.97e-5
        - "1.7 × 10^{-4}" -> 1.7e-4
        - "1.6 x 10^-5" -> 1.6e-5
        - "2.1 × 10⁻⁶" -> 2.1e-6 (Unicode superscript)
        - "5.0 × 10<sup>3</sup>" -> 5000.0
        - "1e-3" or "1E-3" -> 0.001
        - "-2.1 ± 0.5" -> -2.1 (takes the central value)
        - "1.5" -> 1.5
        - "14%" -> 0.14

        Args:
            s: String to parse.

        Returns:
            Parsed float value.

        Raises:
            ValueError: If string cannot be parsed as a float.
        """
        s = s.strip()
        # Check for percentage pattern: "14%" -> 0.14
        if re.match(self.PERCENTAGE_PATTERN, s):
            perc = s[:-1].replace("−", "-")
            return float(perc) / 100

        out: float
        # Match standard scientific notation: "1e-3" or "1E-3"
        sci_e_match = re.match(self.SCI_E_EXTRACT, s)
        if sci_e_match:
            base = float(sci_e_match.group(1).replace("−", "-"))
            exp = float(sci_e_match.group(2).replace("−", "-"))
            out = base * (10**exp)
            return out

        # Match HTML <sup> pattern: "1.0 x 10<sup>-6</sup>"
        sup_match = re.match(self.SCI_SUP_EXTRACT, s)
        if sup_match:
            base = float(sup_match.group(1).replace("−", "-"))
            exp = float(sup_match.group(2).replace("−", "-"))
            out = base * (10**exp)
            return out

        # Match Unicode superscript pattern: "2.1 × 10⁻⁶"
        unicode_match = re.match(self.SCI_UNICODE_EXTRACT, s)
        if unicode_match:
            base = float(unicode_match.group(1))
            exp_str = unicode_match.group(2)
            exp = self._parse_unicode_superscript(exp_str)
            out = base * (10**exp)
            return out

        # Match LaTeX scientific notation: "1.3 \times 10^{-5}" or "1.7 × 10^{-4}"
        latex_match = re.match(self.SCI_LATEX_EXTRACT, s)
        if latex_match:
            base = float(latex_match.group(1).replace("−", "-"))
            exp = float(latex_match.group(2).replace("−", "-"))
            out = base * (10**exp)
            return out

        # Match caret notation: "1.3 × 10^-4" or "1.0 x 10^6" or "1.6 x 10^{-5}"
        caret_match = re.match(self.SCI_CARET_EXTRACT, s)
        if caret_match:
            base = float(caret_match.group(1).replace("−", "-"))
            exp = float(caret_match.group(2).replace("−", "-"))
            out = base * (10**exp)
            return out

        # Match ± pattern: "-2.1 ± 0.5" -> take central value (-2.1)
        pm_match = re.match(self.PM_EXTRACT, s)
        if pm_match:
            return float(pm_match.group(1).replace("−", "-"))

        # Standard float() handles "1e-3", "1E-3", "1.5e-3", etc.
        return float(s.replace("−", "-"))

    def _parse_unicode_superscript(self, s: str) -> int:
        """Convert Unicode superscript characters to an integer.

        Args:
            s: String containing Unicode superscript characters (e.g., "⁻⁶").

        Returns:
            Integer value represented by the superscript.
        """
        # Mapping of Unicode superscript characters to their values
        superscript_map = {
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁺": "+",
            "⁻": "-",
        }
        result = "".join(superscript_map.get(c, c) for c in s)
        return int(result)

    def _extract_regression_answer(
        self, answer_text: str, property: str
    ) -> Optional[float]:
        """Extract a regression answer from text.

        Handles various formats:
        - Plain floats: "1.5", "-2.0"
        - Scientific notation: "1e-3", "1E-3", "1.5e-6"
        - LaTeX notation: "1.3 \times 10^{-5}", "1.7 × 10^{-4}"
        - Caret notation: "1.3 × 10^-4"
        - Plus-minus notation: "-2.1 ± 0.5" -> returns central value (-2.1)
        - Ranges with 'to': "1.0 to 2.0" -> returns average (1.5)
        - Ranges with '-': "1.0 - 2.0" -> returns average (1.5)
        - Ranges with 'between/and': "between 1.0 and 2.0" -> returns average (1.5)
        - Scientific notation with <sup>: "1.0 x 10<sup>-6</sup> to 5.0 x 10<sup>-6</sup>"

        Numbers preceded or followed by a slash (e.g., "0.02 g/100 mL") are excluded.

        Args:
            answer_text: The text content from within <answer> tags.
            property: The property name to look for in "property = value" patterns.

        Returns:
            Extracted float value, or None if extraction fails or is ambiguous.
        """
        # If "property = value", "property=value" or "property is value" format, extract the value
        prop_pattern = (
            rf"{re.escape(property.lower())}\s*(?:=|is)\s*({self.NUM_PATTERN})"
        )
        prop_match = re.search(prop_pattern, answer_text.lower(), flags=re.IGNORECASE)
        if prop_match:
            return self._parse_float_with_sup(prop_match.group(1))

        # Check for "between X and Y" pattern
        between_regex = rf"between\s*({self.NUM_PATTERN})\s*and\s*({self.NUM_PATTERN})"
        between_matches = re.findall(between_regex, answer_text, re.IGNORECASE)
        if len(between_matches) == 1:
            start, end = between_matches[0]
            return (
                self._parse_float_with_sup(start) + self._parse_float_with_sup(end)
            ) / 2

        # Check for range patterns: "{num} to {num}" or "{num} - {num}"
        range_regex = rf"({self.NUM_PATTERN})(?:\s+-\s+|\s+to\s+)({self.NUM_PATTERN})"
        range_matches = re.findall(range_regex, answer_text)

        if len(range_matches) > 0 and all(
            range_matches[0] == rm for rm in range_matches
        ):
            start, end = range_matches[0]
            return (
                self._parse_float_with_sup(start) + self._parse_float_with_sup(end)
            ) / 2

        # Otherwise extract all float numbers (including scientific notation)
        # Use negative lookbehind and lookahead to exclude numbers with slashes
        ys: List[float] = []
        all_nums = re.findall(self.NUM_WITH_BOUNDARY_PATTERN, answer_text)
        for num_str in all_nums:
            try:
                ys.append(self._parse_float_with_sup(num_str))
            except ValueError:
                continue

        if len(ys) == 0:
            return None
        if len(ys) > 1 and not all(y == ys[0] for y in ys):
            return None  # Ambiguous: multiple different values

        return ys[0]

    def _extract_classification_answer(self, answer_text: str) -> Optional[int]:
        """Extract a classification answer from text.

        Handles various formats:
        - Boolean strings: "true", "yes" -> 1; "false", "no" -> 0
        - Numeric strings: "1", "1.0" -> 1; "0", "0.0" -> 0

        Args:
            answer_text: The text content from within <answer> tags.

        Returns:
            Extracted int value (0 or 1), or None if extraction fails or is ambiguous.
        """
        ys: List[int] = []
        split_answer = re.split(r"\n| |\t|:|`|'|,", answer_text)
        for spl in split_answer:
            if spl.lower().replace(".", "") in self.CLASSIFICATION_TRUE_KEYWORDS:
                ys.append(1)
            elif spl.lower().replace(".", "") in self.CLASSIFICATION_FALSE_KEYWORDS:
                ys.append(0)

        if len(ys) == 0:
            return None
        if len(ys) > 1 and not all(y == ys[0] for y in ys):
            return None  # Ambiguous: multiple different values

        return ys[0]

    def get_score(
        self, inputs: BatchVerifiersInputModel
    ) -> List[MolPropVerifierOutputModel]:
        """Compute property prediction rewards for a batch of completions.

        This method extracts predicted values from answer tags in completions
        and computes rewards based on the objective type:
        - Regression: reward = clip(1 - ((predicted - target) / std)^2, 0, 1)
        - Classification: reward = 1.0 if predicted == target, else 0.0

        Args:
            inputs: Batch of completions and metadata for verification.

        Returns:
            List of MolPropVerifierOutputModel containing rewards and metadata.

        Notes:
            - Answers must be enclosed in <answer></answer> tags
            - Supports "true"/"yes" as 1 and "false"/"no" as 0 for classification
            - Invalid or missing answers result in 0.0 reward
        """
        completions = inputs.completions
        assert all(
            isinstance(meta, MolPropVerifierInputMetadataModel)
            for meta in inputs.metadatas
        )
        metadatas: List[MolPropVerifierInputMetadataModel] = inputs.metadatas  # type: ignore

        verifier_metadatas: List[MolPropVerifierMetadataModel] = []
        all_matches: List[str] = []
        for answer, meta in zip(completions, metadatas):
            match = self.parse_answer(answer)
            all_matches.append(match)
            self.logger.info(f"Match: {match}")
            extracted: float | int | None = None
            if match != "":
                try:
                    if meta.objectives[0] == "classification":
                        extracted = self._extract_classification_answer(match)
                        if extracted is None:
                            self.logger.info(
                                f"Could not extract classification value from: {match}"
                            )
                    elif meta.objectives[0] == "regression":
                        extracted = self._extract_regression_answer(
                            match, meta.properties[0]
                        )
                        if extracted is None:
                            self.logger.info(
                                f"Could not extract regression value from: {match}"
                            )
                    else:
                        extracted = None
                    verifier_metadatas.append(
                        MolPropVerifierMetadataModel(
                            extracted_answer=extracted,
                            extraction_success=True,
                        )
                    )
                except ValueError:
                    verifier_metadatas.append(
                        MolPropVerifierMetadataModel(
                            extracted_answer=-10000.0,
                            extraction_success=False,
                        )
                    )
            else:
                verifier_metadatas.append(
                    MolPropVerifierMetadataModel(
                        extracted_answer=-10000.0,
                        extraction_success=False,
                    )
                )

        if self.verifier_config.reward == "valid_smiles":
            return [
                MolPropVerifierOutputModel(
                    reward=float(
                        isinstance(verifier_meta.extracted_answer, (float, int))
                    ),
                    parsed_answer=m,
                    verifier_metadata=verifier_meta,
                )
                for m, verifier_meta in zip(all_matches, verifier_metadatas)
            ]

        rewards = []
        for meta, verifier_meta in zip(metadatas, verifier_metadatas):
            if not verifier_meta.extraction_success:
                rewards.append(0.0)
            else:
                if meta.objectives[0] == "regression":
                    std = meta.norm_var if meta.norm_var is not None else 1.0
                    rewards.append(
                        np.clip(
                            1
                            - ((verifier_meta.extracted_answer - meta.target[0]) / std)
                            ** 2,
                            a_min=0.0,
                            a_max=1.0,
                        )
                    )
                elif meta.objectives[0] == "classification":
                    rewards.append(
                        float(verifier_meta.extracted_answer == meta.target[0])
                    )
                else:
                    self.logger.error(f"Not a valid objective: {meta.objectives[0]}")
                    raise NotImplementedError

        self.logger.info(f"Rewards: {rewards}")
        return [
            MolPropVerifierOutputModel(
                reward=reward, parsed_answer=m, verifier_metadata=verifier_metadata
            )
            for reward, m, verifier_metadata in zip(
                rewards, all_matches, verifier_metadatas
            )
        ]

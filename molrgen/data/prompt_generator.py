"""Flexible prompt generator for converting metadata dictionaries to textual prompts."""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union, overload

from jinja2 import Template


class PromptGenerator:
    """
    A flexible class for generating textual prompts from metadata dictionaries.

    This class allows you to define custom transformation functions that convert
    a metadata dictionary into a textual prompt. It supports various prompt templates
    and can be easily extended with custom transformation logic.

    Example:
        >>> generator = PromptGenerator()
        >>> metadata = {
        ...     "properties": ["docking_score", "logP"],
        ...     "objectives": ["minimize", "maximize"],
        ...     "target": [-8.5, 4.0]
        ... }
        >>> prompt = generator(metadata)
    """

    def __init__(
        self,
        transform_fn: Optional[Callable[[Dict[str, Any]], str] | str] = None,
        data_path: Optional[str] = None,
    ):
        """
        Initialize the PromptGenerator.

        Args:
            transform_fn: Optional custom function that transforms metadata dict to text.
                         If None, uses the default transformation.
                         Jinja2 template strings are also supported.
            data_path: Optional path to data directory for loading property mappings
                      and objective templates.
        """
        self.transform_fn: Callable[[Dict[str, Any]], str] | None = None
        if transform_fn is not None:
            self.set_transform_function(transform_fn)
        self.data_path = data_path

        # Load property mappings and templates if data_path is provided
        self.prop_name_mapping: Dict[str, str] = {}
        self.docking_targets: List[str] = []
        self.obj_templates: Dict[str, List[str]] = {}
        self.prompt_templates: List[str] = []

        if data_path:
            self._load_property_configs(data_path)

    def _load_property_configs(self, path: str) -> None:
        """Load property configurations from the data directory."""
        # Load property name mapping
        prop_name_mapping_path = os.path.join(path, "names_mapping.json")
        if os.path.exists(prop_name_mapping_path):
            with open(prop_name_mapping_path) as f:
                self.prop_name_mapping = json.load(f)
        else:
            raise ValueError(
                f"Property name mapping file not found at {prop_name_mapping_path}"
            )

        # Load docking targets
        docking_target_list_path = os.path.join(path, "docking_targets.json")
        if os.path.exists(docking_target_list_path):
            with open(docking_target_list_path) as f:
                self.docking_targets = json.load(f)
        else:
            raise ValueError(
                f"Docking target list file not found at {docking_target_list_path}"
            )

    @overload
    def set_transform_function(
        self, transform: Callable[[Dict[str, Any]], str]
    ) -> None:
        """
        Set a custom transformation function.

        Args:
            transform: A callable that takes a metadata dict and returns a string prompt.
        """
        ...

    @overload
    def set_transform_function(self, transform: str) -> None:
        """
        Set a Jinja2 template pattern for prompt generation.

        The template will be rendered with the metadata dictionary as context.
        This provides a flexible way to define custom prompt formats without writing Python code.

        Args:
            transform: A Jinja2 template string. The template can access all keys in the
                      metadata dictionary.

        Example:
            >>> template = '''Generate molecules optimized for:
            ... {% for prop, obj, target in zip(properties, objectives, target) %}
            ... - {{ obj|capitalize }} {{ prop }}{% if obj not in ['maximize', 'minimize'] %} to {{ target|round(2) }}{% endif %}
            ... {% endfor %}
            ...
            ... Number of objectives: {{ properties|length }}'''
            >>> generator.set_transform_function(template)
        """
        ...

    def set_transform_function(
        self, transform: Union[Callable[[Dict[str, Any]], str], str]
    ) -> None:
        """
        Set a custom transformation function or Jinja2 template pattern for prompt generation.

        Args:
            transform: Either a callable that takes a metadata dict and returns a string,
                      or a Jinja2 template string that will be rendered with metadata as context.
        """
        if isinstance(transform, str):
            # Handle Jinja2 template
            template = Template(transform)

            def jinja_transform_fn(metadata: Dict[str, Any]) -> str:
                """Render the Jinja2 template with the metadata as context."""
                # Provide built-in functions to the template context
                out: str = template.render(zip=zip, **metadata)
                return out

            self.transform_fn = jinja_transform_fn
        else:
            # Handle callable function
            self.transform_fn = transform

    def __call__(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a textual prompt from a metadata dictionary.

        Args:
            metadata: Dictionary containing metadata to transform into a prompt.
                     Common keys might include: properties, objectives, targets, etc.

        Returns:
            A string containing the generated prompt.
        """
        if not self.validate_metadata(metadata):
            raise ValueError("Invalid metadata structure.")

        if self.transform_fn is not None:
            return self.transform_fn(metadata)
        else:
            return self._default_transform(metadata)

    def _default_transform(self, metadata: Dict[str, Any]) -> str:
        """
        Default transformation function that converts metadata to a prompt.

        Handles common metadata patterns for molecular generation tasks.

        Args:
            metadata: Dictionary with keys like 'properties', 'objectives', 'targets'

        Returns:
            A formatted text prompt.
        """
        prompt_parts = []

        # Handle objectives/properties
        if "objectives" in metadata and "properties" in metadata:
            objectives = metadata["objectives"]
            properties = metadata["properties"]
            targets = metadata["target"]

            prompt_parts.append("Generate molecules with the following objectives:")

            for i, (prop, obj) in enumerate(zip(properties, objectives)):
                target_str = ""
                if obj not in ["maximize", "minimize"]:
                    target_str = f" {targets[i]:.2f}"

                prompt_parts.append(f"  - {obj} {prop}{target_str}")

        return "\n".join(prompt_parts)

    def generate_batch(self, metadata_list: List[Dict[str, Any]]) -> List[str]:
        """
        Generate prompts for a batch of metadata dictionaries.

        Args:
            metadata_list: List of metadata dictionaries.

        Returns:
            List of generated prompts.
        """
        return [self(metadata) for metadata in metadata_list]

    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate that the metadata has the required structure.

        Args:
            metadata: Dictionary to validate.

        Returns:
            True if metadata is valid, False otherwise.
        """
        required_keys = ["properties", "objectives", "target"]
        return all(key in metadata for key in required_keys)

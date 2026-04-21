"""Tests for the PromptGenerator class."""

from typing import Any, Dict, List

import pytest

from molrgen.data.prompt_generator import PromptGenerator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture  # type: ignore
def generator() -> PromptGenerator:
    """Create a basic PromptGenerator instance."""
    return PromptGenerator()


@pytest.fixture  # type: ignore
def mol_gen_metadata() -> Dict[str, Any]:
    """Metadata for molecular generation tasks."""
    return {
        "properties": ["docking_score", "logP", "QED"],
        "objectives": ["minimize", "maximize", "above"],
        "target": [-8.5, 4.0, 0.7],
    }


@pytest.fixture  # type: ignore
def property_prediction_metadata() -> Dict[str, Any]:
    """Metadata for property prediction tasks."""
    return {
        "properties": ["solubility"],
        "objectives": ["regression"],
        "target": [0.0, 0.0],
        "smiles": ["CCO"],
        "task_type": "property_prediction",
    }


@pytest.fixture  # type: ignore
def reaction_metadata() -> Dict[str, Any]:
    """Metadata for reaction tasks."""
    return {
        "properties": ["smarts"],
        "objectives": ["smarts"],
        "target": [
            "[#6:1][C:2](=O)[#6:3].[#6:4][OH,nH,NH:5]>>[#6:4][*:5][C:2]([#6:1])([#6:3])C(=O)O"
        ],
        "prompt_id": "synth0",
        "full_reaction": "O=C(CCC(F)F)c1ccccc1 + C#CCNC(=O)c1ccccn1 -> C#CCN(C(=O)c1ccccn1)C(CCC(F)F)(C(=O)O)c1ccccc1",
        "or_smarts": [
            "[#6:1][C:2](=O)[#6:3].[#6:4][OH,nH,NH:5]>>[#6:4][*:5][C:2]([#6:1])([#6:3])C(=O)O"
        ],
        "impossible": True,
        "smarts": [
            "[#6:1][C:2](=O)[#6:3].[#6:4][OH,nH,NH:5]>>[#6:4][*:5][C:2]([#6:1])([#6:3])C(=O)O"
        ],
        "reactants": [
            ["Cn1cc(Br)c(S(=O)(=O)Cl)n1", "CCOC(=O)c1c(NC(=O)CCl)sc(C(C)=O)c1C"]
        ],
        "products": ["C#CCN(C(=O)c1ccccn1)C(CCC(F)F)(C(=O)O)c1ccccc1"],
        "building_blocks": [],
        "idx_chosen": 0,
        "n_building_blocks": 0,
        "pass_filters": [True],
    }


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestPromptGeneratorInit:
    """Tests for PromptGenerator initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        generator = PromptGenerator()
        assert generator.transform_fn is None
        assert generator.data_path is None
        assert generator.prop_name_mapping == {}
        assert generator.docking_targets == []

    def test_init_with_callable(self) -> None:
        """Test initialization with a callable transform function."""

        def custom_fn(metadata: Dict[str, Any]) -> str:
            return "custom"

        generator = PromptGenerator(transform_fn=custom_fn)
        assert generator.transform_fn is custom_fn


class TestValidateMetadata:
    """Tests for metadata validation."""

    def test_valid_metadata(
        self, generator: PromptGenerator, mol_gen_metadata: Dict[str, Any]
    ) -> None:
        """Test validation with valid metadata."""
        assert generator.validate_metadata(mol_gen_metadata) is True

    def test_missing_properties(self, generator: PromptGenerator) -> None:
        """Test validation with missing properties key."""
        metadata = {"objectives": ["minimize"], "target": [0.0]}
        assert generator.validate_metadata(metadata) is False

    def test_missing_objectives(self, generator: PromptGenerator) -> None:
        """Test validation with missing objectives key."""
        metadata = {"properties": ["logP"], "target": [0.0]}
        assert generator.validate_metadata(metadata) is False

    def test_missing_target(self, generator: PromptGenerator) -> None:
        """Test validation with missing target key."""
        metadata = {"properties": ["logP"], "objectives": ["minimize"]}
        assert generator.validate_metadata(metadata) is False


class TestDefaultTransform:
    """Tests for default transformation."""

    def test_default_transform_mol_gen(
        self, generator: PromptGenerator, mol_gen_metadata: Dict[str, Any]
    ) -> None:
        """Test default transform with molecular generation metadata."""
        prompt = generator(mol_gen_metadata)
        assert "Generate molecules" in prompt
        assert "docking_score" in prompt
        assert "logP" in prompt
        assert "QED" in prompt
        assert "minimize" in prompt
        assert "maximize" in prompt

    def test_default_transform_includes_target_for_threshold_objectives(
        self, generator: PromptGenerator
    ) -> None:
        """Test that target values are included for threshold objectives."""
        metadata = {
            "properties": ["logP"],
            "objectives": ["above"],
            "target": [3.5],
        }
        prompt = generator(metadata)
        assert "3.50" in prompt


# =============================================================================
# Callable Transform Function Tests
# =============================================================================


class TestCallableTransforms:
    """Tests for callable transformation functions."""

    def test_callable_simple_mol_gen(
        self, generator: PromptGenerator, mol_gen_metadata: Dict[str, Any]
    ) -> None:
        """Test callable #1: Simple molecular generation prompt."""

        def mol_gen_transform(metadata: Dict[str, Any]) -> str:
            props = metadata["properties"]
            objs = metadata["objectives"]
            lines = ["Design a molecule with:"]
            for prop, obj in zip(props, objs):
                lines.append(f"  - {obj.upper()} {prop}")
            return "\n".join(lines)

        generator.set_transform_function(mol_gen_transform)
        prompt = generator(mol_gen_metadata)

        assert "Design a molecule with:" in prompt
        assert "MINIMIZE docking_score" in prompt
        assert "MAXIMIZE logP" in prompt
        assert "ABOVE QED" in prompt

    def test_callable_property_prediction(
        self, generator: PromptGenerator, property_prediction_metadata: Dict[str, Any]
    ) -> None:
        """Test callable #2: Property prediction prompt."""

        def property_prediction_transform(metadata: Dict[str, Any]) -> str:
            smiles = metadata.get("smiles", "UNKNOWN")
            props = metadata["properties"]
            return (
                f"Predict the following properties for the molecule {smiles}:\n"
                f"Properties: {', '.join(props)}\n"
                f"Provide numerical predictions for each property."
            )

        generator.set_transform_function(property_prediction_transform)
        prompt = generator(property_prediction_metadata)

        assert "Predict the following properties" in prompt
        assert "CCO" in prompt
        assert "solubility" in prompt

    def test_callable_reaction_task(
        self, generator: PromptGenerator, reaction_metadata: Dict[str, Any]
    ) -> None:
        """Test callable #3: Reaction task prompt."""

        def reaction_transform(metadata: Dict[str, Any]) -> str:
            reactants = metadata.get("reactants", [[]])
            product = metadata.get("products", [""])[0]
            reaction_type = metadata.get("reaction_type", "unknown")
            props = metadata["properties"]
            objs = metadata["objectives"]

            lines = [
                f"Reaction Type: {reaction_type}",
                f"Reactants: {' + '.join(reactants[0])}",
                f"Expected Product: {product}",
                "Optimization Goals:",
            ]
            for prop, obj in zip(props, objs):
                lines.append(f"  - {obj} {prop}")
            return "\n".join(lines)

        generator.set_transform_function(reaction_transform)
        prompt = generator(reaction_metadata)

        for r in reaction_metadata["reactants"][0]:
            assert r in prompt
        assert reaction_metadata["products"][0] in prompt

    def test_callable_with_docking_metadata(self, generator: PromptGenerator) -> None:
        """Test callable #4: Docking-specific prompt with pocket info."""

        def docking_transform(metadata: Dict[str, Any]) -> str:
            props = metadata["properties"]
            docking_meta = metadata.get("docking_metadata", [])

            lines = ["Generate a molecule that binds to the following targets:"]
            for i, prop in enumerate(props):
                if i < len(docking_meta) and "pdb_id" in docking_meta[i]:
                    lines.append(
                        f"  - Target: {prop} (PDB: {docking_meta[i]['pdb_id']})"
                    )
                else:
                    lines.append(f"  - Optimize: {prop}")
            return "\n".join(lines)

        metadata = {
            "properties": ["1ABC_pocket", "logP"],
            "objectives": ["minimize", "maximize"],
            "target": [-9.0, 3.0],
            "docking_metadata": [{"pdb_id": "1ABC"}, {}],
        }

        generator.set_transform_function(docking_transform)
        prompt = generator(metadata)

        assert "binds to the following targets" in prompt
        assert "PDB: 1ABC" in prompt


# =============================================================================
# Jinja2 Template Tests
# =============================================================================


class TestJinjaTemplates:
    """Tests for Jinja2 template-based transformations."""

    def test_jinja_simple_mol_gen(
        self, generator: PromptGenerator, mol_gen_metadata: Dict[str, Any]
    ) -> None:
        """Test Jinja template #1: Simple molecular generation."""
        template = """Design a molecule with the following properties:
{% for prop, obj in zip(properties, objectives) %}
- {{ obj|capitalize }} {{ prop }}
{% endfor %}"""

        generator.set_transform_function(template)
        prompt = generator(mol_gen_metadata)

        assert "Design a molecule" in prompt
        assert "Minimize docking_score" in prompt
        assert "Maximize logP" in prompt
        assert "Above QED" in prompt

    def test_jinja_with_targets(
        self, generator: PromptGenerator, mol_gen_metadata: Dict[str, Any]
    ) -> None:
        """Test Jinja template #2: Template with conditional target values."""
        template = """### Molecular Optimization Task

**Objectives:**
{% for prop, obj, tgt in zip(properties, objectives, target) %}
  • {{ prop }}: {{ obj }}{% if obj not in ['maximize', 'minimize'] %} (target: {{ tgt|round(2) }}){% endif %}
{% endfor %}

Generate SMILES for molecules meeting these criteria."""

        generator.set_transform_function(template)
        prompt = generator(mol_gen_metadata)

        assert "Molecular Optimization Task" in prompt
        assert "docking_score: minimize" in prompt
        assert "logP: maximize" in prompt
        assert "QED: above" in prompt
        assert "(target: 0.7)" in prompt
        assert "Generate SMILES" in prompt

    def test_jinja_property_prediction(
        self, generator: PromptGenerator, property_prediction_metadata: Dict[str, Any]
    ) -> None:
        """Test Jinja template #3: Property prediction task."""
        template = """You are a chemistry expert. Predict molecular properties.

Input SMILES: {{ smiles }}

Properties to predict:
{% for prop in properties %}
- {{ prop }}
{% endfor %}

Provide your predictions in JSON format."""

        generator.set_transform_function(template)
        prompt = generator(property_prediction_metadata)

        assert "chemistry expert" in prompt
        assert "CCO" in prompt
        assert "solubility" in prompt
        assert "JSON format" in prompt

    def test_jinja_reaction_task(
        self, generator: PromptGenerator, reaction_metadata: Dict[str, Any]
    ) -> None:
        """Test Jinja template #4: Reaction optimization task."""
        template = """### Chemical Reaction Optimization

**Reaction:** {{ reaction_type|capitalize }}

**Reactants:**
{% for r in reactants %}
  {{ loop.index }}. {{ r }}
{% endfor %}

**Target Product:** {{ products }}

**Optimization Goals:**
{% for prop, obj, tgt in zip(properties, objectives, target) %}
  - {{ obj|capitalize }} {{ prop }}{% if tgt != 0.0 %} to {{ tgt }}{% endif %}
{% endfor %}

Suggest reaction conditions to achieve these goals."""

        generator.set_transform_function(template)
        prompt = generator(reaction_metadata)

        assert "Chemical Reaction Optimization" in prompt
        for r in reaction_metadata["reactants"][0]:
            assert r in prompt
        assert reaction_metadata["products"][0] in prompt

    def test_jinja_multi_objective_with_length(
        self, generator: PromptGenerator
    ) -> None:
        """Test Jinja template #5: Using length filter and conditionals."""
        template = """Multi-objective optimization ({{ properties|length }} objectives):

{% if properties|length == 1 %}
Single objective: {{ objectives[0] }} {{ properties[0] }}
{% elif properties|length == 2 %}
Dual objectives:
{% for p, o in zip(properties, objectives) %}
  - {{ o }} {{ p }}
{% endfor %}
{% else %}
Multiple objectives ({{ properties|length }}):
{% for p, o in zip(properties, objectives) %}
  {{ loop.index }}. {{ o|upper }} {{ p }}
{% endfor %}
{% endif %}"""

        generator.set_transform_function(template)

        # Test with 3 objectives
        metadata = {
            "properties": ["A", "B", "C"],
            "objectives": ["maximize", "minimize", "equal"],
            "target": [1.0, 2.0, 3.0],
        }
        prompt = generator(metadata)
        assert "Multiple objectives (3)" in prompt
        assert "1. MAXIMIZE A" in prompt
        assert "2. MINIMIZE B" in prompt
        assert "3. EQUAL C" in prompt

    def test_jinja_docking_with_metadata(self, generator: PromptGenerator) -> None:
        """Test Jinja template #6: Docking task with pocket metadata."""
        template = """Design a drug candidate targeting:

{% for prop, obj, tgt in zip(properties, objectives, target) %}
**{{ prop }}**
  - Objective: {{ obj }}
  - Target score: {{ tgt }}
{% endfor %}

{% if docking_metadata %}
Binding pocket information:
{% for dock in docking_metadata %}
{% if dock.pdb_id is defined %}
  - PDB ID: {{ dock.pdb_id }}
{% endif %}
{% if dock.resolution is defined %}
  - Resolution: {{ dock.resolution }} Å
{% endif %}
{% endfor %}
{% endif %}

Output the SMILES of your designed molecule."""

        metadata = {
            "properties": ["target_1"],
            "objectives": ["minimize"],
            "target": [-9.5],
            "docking_metadata": [{"pdb_id": "4XYZ", "resolution": 2.1}],
        }

        generator.set_transform_function(template)
        prompt = generator(metadata)

        assert "drug candidate" in prompt
        assert "target_1" in prompt
        assert "-9.5" in prompt
        assert "PDB ID: 4XYZ" in prompt
        assert "Resolution: 2.1" in prompt


# =============================================================================
# Batch Generation Tests
# =============================================================================


class TestBatchGeneration:
    """Tests for batch prompt generation."""

    def test_batch_generation_default(self, generator: PromptGenerator) -> None:
        """Test batch generation with default transform."""
        metadata_list = [
            {"properties": ["A"], "objectives": ["maximize"], "target": [1.0]},
            {"properties": ["B"], "objectives": ["minimize"], "target": [2.0]},
            {"properties": ["C"], "objectives": ["above"], "target": [0.5]},
        ]

        prompts = generator.generate_batch(metadata_list)

        assert len(prompts) == 3
        assert "A" in prompts[0]
        assert "B" in prompts[1]
        assert "C" in prompts[2]

    def test_batch_generation_with_callable(self, generator: PromptGenerator) -> None:
        """Test batch generation with callable transform."""

        def simple_transform(metadata: Dict[str, Any]) -> str:
            return f"Props: {metadata['properties']}"

        generator.set_transform_function(simple_transform)

        metadata_list = [
            {"properties": ["X"], "objectives": ["max"], "target": [0.0]},
            {"properties": ["Y"], "objectives": ["min"], "target": [0.0]},
        ]

        prompts = generator.generate_batch(metadata_list)

        assert len(prompts) == 2
        assert "Props: ['X']" in prompts[0]
        assert "Props: ['Y']" in prompts[1]

    def test_batch_generation_with_jinja(self, generator: PromptGenerator) -> None:
        """Test batch generation with Jinja template."""
        template = "Optimize: {{ properties | join(', ') }}"
        generator.set_transform_function(template)

        metadata_list = [
            {"properties": ["p1", "p2"], "objectives": ["a", "b"], "target": [0, 0]},
            {"properties": ["p3"], "objectives": ["c"], "target": [0]},
        ]

        prompts = generator.generate_batch(metadata_list)

        assert len(prompts) == 2
        assert "Optimize: p1, p2" in prompts[0]
        assert "Optimize: p3" in prompts[1]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_metadata_raises_error(self, generator: PromptGenerator) -> None:
        """Test that invalid metadata raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metadata"):
            generator({})

    def test_empty_properties_list(self, generator: PromptGenerator) -> None:
        """Test with empty properties list."""
        metadata: Dict[str, List[Any]] = {
            "properties": [],
            "objectives": [],
            "target": [],
        }
        prompt = generator(metadata)
        assert "Generate molecules" in prompt

    def test_switch_transform_functions(
        self, generator: PromptGenerator, mol_gen_metadata: Dict[str, Any]
    ) -> None:
        """Test switching between different transform functions."""
        # First callable
        generator.set_transform_function(lambda m: "Callable 1")
        assert generator(mol_gen_metadata) == "Callable 1"

        # Second callable
        generator.set_transform_function(lambda m: "Callable 2")
        assert generator(mol_gen_metadata) == "Callable 2"

        # Jinja template
        generator.set_transform_function("Template: {{ properties|length }}")
        assert "Template: 3" in generator(mol_gen_metadata)

    def test_special_characters_in_metadata(self, generator: PromptGenerator) -> None:
        """Test handling of special characters in metadata."""
        metadata = {
            "properties": ["prop<with>special&chars"],
            "objectives": ["maximize"],
            "target": [0.0],
        }
        prompt = generator(metadata)
        assert "prop<with>special&chars" in prompt


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_mol_gen_workflow(self) -> None:
        """Test complete molecular generation workflow."""
        generator = PromptGenerator()

        template = """You are a computational chemist AI assistant.

Task: Design molecules with the following optimization objectives:
{% for prop, obj, tgt in zip(properties, objectives, target) %}
{{ loop.index }}. {{ obj|capitalize }} {{ prop }}{% if obj not in ['maximize', 'minimize'] %} (target: {{ tgt }}){% endif %}
{% endfor %}

Instructions:
- Provide valid SMILES strings
- Ensure drug-likeness (Lipinski's rules)
- Consider synthetic accessibility

Your response should be in the format:
<think>reasoning</think>
<answer>SMILES</answer>"""

        generator.set_transform_function(template)

        metadata = {
            "properties": ["binding_affinity", "logP", "molecular_weight"],
            "objectives": ["minimize", "below", "below"],
            "target": [0.0, 5.0, 500.0],
        }

        prompt = generator(metadata)

        assert "computational chemist" in prompt
        assert "Minimize binding_affinity" in prompt
        assert "Below logP (target: 5.0)" in prompt
        assert "Below molecular_weight (target: 500.0)" in prompt
        assert "<think>" in prompt
        assert "<answer>" in prompt

    def test_full_reaction_workflow(self) -> None:
        """Test complete reaction prediction workflow."""
        generator = PromptGenerator()

        def reaction_prompt(metadata: Dict[str, Any]) -> str:
            return f"""Reaction Prediction Task

Given reactants: {" + ".join(metadata["reactants"])}
Reaction type: {metadata["reaction_type"]}

Predict:
1. Main product SMILES
2. Expected yield ({metadata["objectives"][0]} target)
3. Reaction conditions

Constraints:
- Properties to optimize: {", ".join(metadata["properties"])}
- Targets: {metadata["target"]}"""

        generator.set_transform_function(reaction_prompt)

        metadata = {
            "properties": ["yield", "purity"],
            "objectives": ["maximize", "above"],
            "target": [0.0, 0.95],
            "reactants": ["c1ccccc1Br", "CC(=O)O"],
            "reaction_type": "Suzuki coupling",
        }

        prompt = generator(metadata)

        assert "Reaction Prediction Task" in prompt
        assert "c1ccccc1Br + CC(=O)O" in prompt
        assert "Suzuki coupling" in prompt
        assert "yield, purity" in prompt

from __future__ import annotations
import  pandas as pd
import  re
import  unicodedata
import  yaml
from    dataclasses  import dataclass, field
from    pathlib      import Path
from    typing       import Any, Dict, List, Optional, Sequence, Union
# velari-data
from    velari_data.registry import RecordRegistry


class SkillValidator(object):
    """Checks a skill against the Agent Skills rules (agentskills.io, as `skills-ref` applies them); each raises `ValueError`.

    Two rules come from Anthropic's platform docs, not the open spec, and are marked below: reserved words in a name, and XML
    tags in a description.
    """
    MAX_NAME_LENGTH           = 64
    MAX_DESCRIPTION_LENGTH    = 1024
    MAX_COMPATIBILITY_LENGTH  = 500

    @staticmethod
    def name(value: Any) -> None:
        """Check a skill name, after stripping it and applying NFKC normalization.

        Args:
            value (Any): The name to check.

        Raises:
            ValueError: If it is empty, over 64 characters, not lowercase, starts or ends with a hyphen, has consecutive hyphens,
                has a character that is not a letter, digit or hyphen, or (platform rule) contains `anthropic` or `claude`.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("skill name must not be empty")
        name = unicodedata.normalize("NFKC", value.strip())
        if len(name) > SkillValidator.MAX_NAME_LENGTH:
            raise ValueError(f"skill name must be at most {SkillValidator.MAX_NAME_LENGTH} characters, got {len(name)}")
        if name != name.lower():
            raise ValueError(f"skill name must be lowercase, got {name!r}")
        if name.startswith("-") or name.endswith("-"):
            raise ValueError(f"skill name cannot start or end with a hyphen, got {name!r}")
        if "--" in name:
            raise ValueError(f"skill name cannot contain consecutive hyphens, got {name!r}")
        if not all(char.isalnum() or char == "-" for char in name):
            raise ValueError(f"skill name may only contain letters, digits and hyphens, got {name!r}")
        if any(word in name for word in ("anthropic", "claude")):
            raise ValueError(f"skill name must not contain anthropic or claude, got {name!r}")

    @staticmethod
    def description(value: Any) -> None:
        """Check a skill description.

        Args:
            value (Any): The description to check.

        Raises:
            ValueError: If it is empty, longer than 1,024 characters, or (platform rule) contains XML tags.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("skill description must not be empty")
        if len(value) > SkillValidator.MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"skill description must be at most {SkillValidator.MAX_DESCRIPTION_LENGTH} characters, got {len(value)}")
        if re.search(r"<[a-zA-Z/][^>]*>", value):
            raise ValueError("skill description must not contain XML tags")

    @staticmethod
    def directory(name: str, path: Path) -> None:
        """Check that a skill's folder is named after the skill, comparing NFKC-normalized names.

        Args:
            name (str): The skill's name.
            path (Path): The skill's folder.

        Raises:
            ValueError: If the folder's name is not the skill's name.
        """
        if unicodedata.normalize("NFKC", path.name) != unicodedata.normalize("NFKC", name.strip()):
            raise ValueError(f"skill folder {path.name!r} must be named after the skill {name!r}")

    @staticmethod
    def compatibility(value: Any) -> None:
        """Check the optional `compatibility` field, which describes the environment a skill needs.

        Args:
            value (Any): The field's value; `None` means the field is absent.

        Raises:
            ValueError: If it is not a string or is longer than 500 characters.
        """
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError(f"skill compatibility must be a string, got {type(value).__name__}")
        if len(value) > SkillValidator.MAX_COMPATIBILITY_LENGTH:
            raise ValueError(f"skill compatibility must be at most {SkillValidator.MAX_COMPATIBILITY_LENGTH} characters, got {len(value)}")

    @staticmethod
    def text(field_name: str, value: Any) -> None:
        """Check an optional text field such as `license`.

        Args:
            field_name (str): The field's name, for the error message.
            value (Any): The field's value; `None` means the field is absent.

        Raises:
            ValueError: If it is present but not a string.
        """
        if value is not None and not isinstance(value, str):
            raise ValueError(f"skill {field_name} must be a string, got {type(value).__name__}")

    @staticmethod
    def metadata(value: Any) -> None:
        """Check the `metadata` mapping.

        Args:
            value (Any): The mapping to check.

        Raises:
            ValueError: If it is not a dict of strings to strings.
        """
        if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            raise ValueError("skill metadata must map strings to strings")


@dataclass(frozen=True)
class Skill(object):
    """A skill: a directory with a `SKILL.md` that gives an agent a specialty on demand.

    Args:
        name (str): The skill's name, which is also its folder's name; see `SkillValidator.name` for the rules.
        description (str): What the skill does and when to use it; see `SkillValidator.description` for the rules.
        path (Path): The skill's directory.
        license (Optional[str]): A license name, or the name of a bundled license file.
        compatibility (Optional[str]): The environment the skill needs, at most 500 characters.
        allowed_tools (Optional[str]): The tools the skill may use, space-separated (the spec's `allowed-tools`, experimental).
        metadata (Dict[str, str]): Extra key-value properties, such as an author or a version.

    Raises:
        ValueError: If a field or the folder's name breaks the Agent Skills rules.

    Examples:
        Read a skill's name and description from its `SKILL.md` frontmatter:

        >>> skill = Skill.from_dir("_build/agent-skills/packages/architect/skills/architecture-spec")
        >>> skill.name
        'architecture-spec'
    """
    name:           str
    description:    str
    path:           Path
    license:        Optional[str]   = field(default=None)
    compatibility:  Optional[str]   = field(default=None)
    allowed_tools:  Optional[str]   = field(default=None)
    metadata:       Dict[str, str]  = field(default_factory=dict)

    def __post_init__(self) -> None:
        SkillValidator.name(self.name)
        SkillValidator.description(self.description)
        SkillValidator.directory(self.name, self.path)
        SkillValidator.text("license", self.license)
        SkillValidator.compatibility(self.compatibility)
        SkillValidator.text("allowed-tools", self.allowed_tools)
        SkillValidator.metadata(self.metadata)

    @classmethod
    def from_dir(cls, path: Union[str, Path]) -> Skill:
        """Build a skill from the YAML frontmatter of `<path>/SKILL.md`; fields beyond the spec's are ignored.

        Args:
            path (Union[str, Path]): The skill's directory.

        Returns:
            Skill: With the frontmatter's fields, and `path` as a `Path`.

        Raises:
            FileNotFoundError: If the directory has no `SKILL.md`.
            ValueError: If the file has no YAML frontmatter, or the skill breaks the Agent Skills rules.
        """
        skill_file = Path(path) / "SKILL.md"
        meta = next(yaml.safe_load_all(skill_file.read_text()), None)   # the first YAML document is the frontmatter
        if not isinstance(meta, dict):
            raise ValueError(f"{skill_file} has no YAML frontmatter")
        raw = meta.get("metadata")
        raw = {} if raw is None else raw   # only an absent field defaults; other non-mappings reach the validator
        metadata = {str(key): str(value) for key, value in raw.items()} if isinstance(raw, dict) else raw   # stringified, as skills-ref does
        return cls(
            name          = meta.get("name", ""),
            description   = meta.get("description", ""),
            path          = skill_file.parent,
            license       = meta.get("license"),
            compatibility = meta.get("compatibility"),
            allowed_tools = meta.get("allowed-tools"),
            metadata      = metadata
        )

    def instructions(self) -> str:
        """Read the body of `SKILL.md`, everything after the frontmatter, for when the skill is triggered.

        Returns:
            str: The markdown body, without surrounding blank lines.

        Raises:
            FileNotFoundError: If the skill's `SKILL.md` is gone.

        Examples:
            >>> skill = Skill.from_dir("_build/agent-skills/packages/architect/skills/diagramming")
            >>> skill.instructions().splitlines()[0]
            '# Diagramming'
        """
        parts = re.split(r"^---[ \t]*$", (self.path / "SKILL.md").read_text(), maxsplit=2, flags=re.MULTILINE)   # fences may have trailing spaces
        return parts[2].strip() if len(parts) == 3 else ""

    def resources(self) -> List[Path]:
        """List the files bundled with the skill, other than its `SKILL.md`, relative to its folder.

        Returns:
            List[Path]: Sorted; hidden files such as `.DS_Store` are left out, and the list is empty when there are none.

        Examples:
            >>> skill = Skill.from_dir("_build/agent-skills/packages/finance/skills/write-invoice")
            >>> skill.resources()
            [PosixPath('references/{name}_{client}_{date}.xlsx')]
        """
        files = (path.relative_to(self.path) for path in self.path.rglob("*") if path.is_file())
        return sorted(path for path in files if path != Path("SKILL.md") and not any(part.startswith(".") for part in path.parts))


class SkillRegistry(RecordRegistry):
    """The skills found under folders, as a table with one row per skill and its attributes and metadata as columns.

    Args:
        *roots (Union[str, Path]): Folders to scan at any depth; a skill is a folder holding a `SKILL.md`. Hidden folders are skipped.

    Raises:
        FileNotFoundError: If a root is not a folder.
        ValueError: If a skill breaks the Agent Skills rules (the message names its folder), or two skills share a name.

    Examples:
        >>> registry = SkillRegistry("_build/agent-skills")
        >>> registry.list()[:3]
        ['architecture-spec', 'brand-guidelines', 'diagramming']
    """
    META_PREFIX      = "meta."
    COLUMNS          = ("name", "description", "path", "location", "license", "compatibility", "allowed_tools")
    CATALOG_COLUMNS  = ("name", f"{META_PREFIX}category", f"{META_PREFIX}tags", "description")

    def __init__(self, *roots: Union[str, Path]) -> None:
        df_skills = pd.DataFrame([self._row(skill) for skill in self._scan(roots)])
        meta_columns = sorted(set(df_skills.columns) - set(self.COLUMNS))
        super().__init__(df_skills.reindex(columns=[*self.COLUMNS, *meta_columns]).sort_values("name"))

    @staticmethod
    def _scan(roots: Sequence[Union[str, Path]]) -> List[Skill]:
        """Load every skill under the roots; raises as the class docstring says."""
        skills: Dict[str, Skill] = {}
        for root in map(Path, roots):
            if not root.is_dir():
                raise FileNotFoundError(f"skills folder not found: {root}")
            for skill_file in sorted(root.rglob("SKILL.md")):
                if any(part.startswith(".") for part in skill_file.relative_to(root).parts):
                    continue   # hidden folders such as .git
                try:
                    skill = Skill.from_dir(skill_file.parent)
                except ValueError as exc:
                    raise ValueError(f"{skill_file.parent}: {exc}") from exc
                if skill.name in skills:
                    raise ValueError(f"duplicate skill {skill.name!r}: {skills[skill.name].path} and {skill.path}")
                skills[skill.name] = skill
        return list(skills.values())

    def _row(self, skill: Skill) -> Dict[str, Any]:
        """One table row: the skill's attributes, then a `meta.<key>` entry per metadata key."""
        return {
            "name"           : skill.name,
            "description"    : skill.description,
            "path"           : str(skill.path),
            "location"       : str(skill.path.resolve() / "SKILL.md"),
            "license"        : skill.license,
            "compatibility"  : skill.compatibility,
            "allowed_tools"  : skill.allowed_tools,
            **{f"{self.META_PREFIX}{key}": value for key, value in skill.metadata.items()},
        }

    def get(self, name: str) -> Skill:
        """Get a skill, rebuilt from its row and validated again.

        Args:
            name (str): The skill's name.

        Returns:
            Skill: The skill, with its `meta.*` columns folded back into `metadata`.

        Raises:
            KeyError: If no skill is named `name`.
        """
        record = super().get(name)
        metadata = {
            str(column).removeprefix(self.META_PREFIX): value
            for column, value in record.items()
            if str(column).startswith(self.META_PREFIX) and value is not None
        }
        return Skill(
            name           = record["name"],
            description    = record["description"],
            path           = Path(record["path"]),
            license        = record["license"],
            compatibility  = record["compatibility"],
            allowed_tools  = record["allowed_tools"],
            metadata       = metadata,
        )

    def catalog_text(self, names: Optional[Sequence[str]] = None) -> str:
        """The skills as an `<available_skills>` block for a prompt: name, category, tags and description.

        Category and tags come from a skill's `metadata` and are left out when it has none; see `RecordRegistry.to_xml`.

        Args:
            names (Optional[Sequence[str]]): Only these skills, in table order; `None` renders every skill.

        Returns:
            str: The XML block; an empty string when there are no skills, so no empty catalog is shown.

        Raises:
            KeyError: If a name in `names` is not a skill in the registry.

        Examples:
            >>> registry = SkillRegistry("_build/agent-skills")
            >>> text = registry.catalog_text(names=["write-invoice"])
            >>> print("\\n".join(text.splitlines()[:3]))
            <available_skills>
              <skill>
                <name>write-invoice</name>
        """
        return self.to_xml(self.CATALOG_COLUMNS, root="available_skills", row="skill", names=names)

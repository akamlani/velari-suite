"""Tests for velari_harness.core.schemas.skill."""

import unicodedata
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest


def _skill_dir(tmp_path, frontmatter, folder="refund-policy"):
    skill_dir = tmp_path / folder
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n# Body\n")
    return skill_dir


def test_from_dir_reads_name_description_and_path_from_the_frontmatter(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill_dir = _skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions. Use when a customer asks about returns.")

    skill = Skill.from_dir(skill_dir)

    assert (skill.name, skill.description, skill.path) == ("refund-policy", "Answer refund questions. Use when a customer asks about returns.", skill_dir)


@pytest.mark.parametrize("frontmatter", [
    "name: Refund-Policy\ndescription: Answer refund questions.",
    "name: -refund\ndescription: Answer refund questions.",
    "name: refund-\ndescription: Answer refund questions.",
    "name: refund--policy\ndescription: Answer refund questions.",
    "name: refund_policy\ndescription: Answer refund questions.",
    "name: refund policy\ndescription: Answer refund questions.",
    f"name: {'a' * 65}\ndescription: Answer refund questions.",
    "name: claude-helper\ndescription: Answer refund questions.",
    "name: anthropic-notes\ndescription: Answer refund questions.",
    "name: refund-policy\ndescription: ''",
    f"name: refund-policy\ndescription: {'x' * 1025}",
    "name: refund-policy\ndescription: Answer <b>refund</b> questions.",
    "description: Answer refund questions.",
    "name: refund-policy",
    "name: refund-policy\ndescription: Answer refund questions.\ncompatibility: " + "x" * 501,
    "name: refund-policy\ndescription: Answer refund questions.\ncompatibility: [git, docker]",
])
def test_from_dir_invalid_name_description_or_compatibility_raises_valueerror(tmp_path, frontmatter):
    from velari_harness.core.schemas.skill import Skill

    with pytest.raises(ValueError):
        Skill.from_dir(_skill_dir(tmp_path, frontmatter))


def test_from_dir_folder_not_named_after_the_skill_raises_valueerror(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    with pytest.raises(ValueError):
        Skill.from_dir(_skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions.", folder="returns"))


def test_from_dir_accepts_unicode_lowercase_letters(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill = Skill.from_dir(_skill_dir(tmp_path, "name: café-notes\ndescription: Take notes at the café.", folder="café-notes"))

    assert skill.name == "café-notes"


def test_from_dir_matches_a_decomposed_folder_name_to_the_composed_skill_name(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    folder = unicodedata.normalize("NFD", "café-notes")

    assert Skill.from_dir(_skill_dir(tmp_path, "name: café-notes\ndescription: Take notes at the café.", folder=folder)).name == "café-notes"


def test_from_dir_ignores_frontmatter_fields_it_does_not_know(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill = Skill.from_dir(_skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions.\nuser-invocable: false\nwhen_to_use: Refunds."))

    assert skill.name == "refund-policy"


def test_from_dir_without_frontmatter_raises_valueerror(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill_dir = tmp_path / "refund-policy"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Refund policy\nNo frontmatter here.\n")

    with pytest.raises(ValueError):
        Skill.from_dir(skill_dir)


def test_from_dir_without_skill_file_raises_filenotfounderror(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    with pytest.raises(FileNotFoundError):
        Skill.from_dir(tmp_path)


def test_skill_is_immutable(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill = Skill.from_dir(_skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions."))

    with pytest.raises(FrozenInstanceError):
        setattr(skill, "name", "other")


def test_instructions_returns_the_body_without_the_frontmatter(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill = Skill.from_dir(_skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions."))

    assert skill.instructions() == "# Body"


def test_instructions_keeps_a_later_horizontal_rule_in_the_body(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill_dir = _skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions.")
    (skill_dir / "SKILL.md").write_text("---\nname: refund-policy\ndescription: Answer refund questions.\n---\n# Refunds\n\n---\n\nWithin 30 days.\n")

    assert Skill.from_dir(skill_dir).instructions() == "# Refunds\n\n---\n\nWithin 30 days."


def test_resources_lists_bundled_files_relative_and_sorted_without_skill_md_or_hidden_files(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    skill_dir = _skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions.")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "policy.md").write_text("30 days")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "check.py").write_text("print('ok')")
    (skill_dir / ".DS_Store").write_text("")

    assert Skill.from_dir(skill_dir).resources() == [Path("references/policy.md"), Path("scripts/check.py")]


def test_resources_is_empty_for_a_skill_with_only_skill_md(tmp_path):
    from velari_harness.core.schemas.skill import Skill

    assert Skill.from_dir(_skill_dir(tmp_path, "name: refund-policy\ndescription: Answer refund questions.")).resources() == []


def _skill_at(parent, name, description="Answer refund questions.", extra=""):
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n# Body\n")
    return skill_dir


def test_skill_registry_holds_one_row_per_skill_sorted_by_name(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    _skill_at(tmp_path, "refund-policy")
    _skill_at(tmp_path, "billing-faq")

    registry = SkillRegistry(tmp_path)

    assert registry.list() == ["billing-faq", "refund-policy"]
    assert list(registry.to_dataframe().columns) == ["name", "description", "path", "location", "license", "compatibility", "allowed_tools"]


def test_skill_registry_scans_nested_folders_and_merges_roots_as_str_or_path(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    _skill_at(tmp_path / "project" / "support" / "skills", "refund-policy")
    _skill_at(tmp_path / "personal", "billing-faq")

    assert SkillRegistry(str(tmp_path / "project"), tmp_path / "personal").list() == ["billing-faq", "refund-policy"]


def test_skill_registry_skips_hidden_folders(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    _skill_at(tmp_path / ".git" / "hooks", "refund-policy")

    assert len(SkillRegistry(tmp_path)) == 0


def test_skill_registry_of_an_empty_folder_has_no_rows_and_keeps_its_columns(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    registry = SkillRegistry(tmp_path)

    assert (len(registry), list(registry.to_dataframe().columns)[:2]) == (0, ["name", "description"])


def test_skill_registry_duplicate_name_across_roots_raises_valueerror_naming_both_folders(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    first = _skill_at(tmp_path / "project", "refund-policy")
    second = _skill_at(tmp_path / "personal", "refund-policy")

    with pytest.raises(ValueError, match="duplicate skill") as error:
        SkillRegistry(tmp_path / "project", tmp_path / "personal")

    assert str(first) in str(error.value) and str(second) in str(error.value)


def test_skill_registry_invalid_skill_raises_valueerror_naming_its_folder(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    bad = _skill_at(tmp_path, "refund-policy", description="''")

    with pytest.raises(ValueError, match="description") as error:
        SkillRegistry(tmp_path)

    assert str(bad) in str(error.value)


def test_skill_registry_missing_root_raises_filenotfounderror(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    with pytest.raises(FileNotFoundError):
        SkillRegistry(tmp_path / "missing")


def test_skill_registry_get_rebuilds_the_skill_with_its_optional_fields_and_metadata(tmp_path):
    from velari_harness.core.schemas.skill import Skill, SkillRegistry

    skill_dir = _skill_at(tmp_path, "refund-policy", extra="license: Apache-2.0\nmetadata:\n  author: support-team\n")

    registry = SkillRegistry(tmp_path)

    assert registry.get("refund-policy") == Skill.from_dir(skill_dir)
    assert registry.get("refund-policy").instructions() == "# Body"


def test_skill_registry_metadata_keys_become_meta_columns(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    _skill_at(tmp_path, "refund-policy", extra="metadata:\n  author: support-team\n")
    _skill_at(tmp_path, "billing-faq")

    df_skills = SkillRegistry(tmp_path).to_dataframe()

    assert df_skills["meta.author"].tolist()[1] == "support-team" and df_skills["meta.author"].isna().tolist()[0]


def test_skill_registry_get_unknown_name_raises_keyerror(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    with pytest.raises(KeyError):
        SkillRegistry(tmp_path).get("missing")


def test_skill_registry_list_filters_by_a_column_and_the_base_class_gives_len_contains_and_iteration(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    _skill_at(tmp_path, "refund-policy", extra="license: MIT\n")
    _skill_at(tmp_path, "billing-faq")

    registry = SkillRegistry(tmp_path)

    assert (registry.list(license="MIT"), len(registry), "billing-faq" in registry, list(registry)) == (["refund-policy"], 2, True, ["billing-faq", "refund-policy"])


def test_catalog_text_renders_name_category_tags_and_description_leaving_out_what_a_skill_lacks(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    _skill_at(tmp_path, "refund-policy", extra="metadata:\n  category: support\n  tags: refunds, returns\n")
    _skill_at(tmp_path, "billing-faq")

    assert SkillRegistry(tmp_path).catalog_text() == (
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>billing-faq</name>\n"
        "    <description>Answer refund questions.</description>\n"
        "  </skill>\n"
        "  <skill>\n"
        "    <name>refund-policy</name>\n"
        "    <category>support</category>\n"
        "    <tags>refunds, returns</tags>\n"
        "    <description>Answer refund questions.</description>\n"
        "  </skill>\n"
        "</available_skills>"
    )


def test_catalog_text_with_names_renders_only_those_skills_and_an_unknown_name_raises_keyerror(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    for name in ("refund-policy", "billing-faq"):
        _skill_at(tmp_path, name)
    registry = SkillRegistry(tmp_path)

    assert "billing-faq" not in registry.catalog_text(names=["refund-policy"])
    with pytest.raises(KeyError):
        registry.catalog_text(names=["missing"])


def test_catalog_text_without_skills_is_empty(tmp_path):
    from velari_harness.core.schemas.skill import SkillRegistry

    assert SkillRegistry(tmp_path).catalog_text() == ""

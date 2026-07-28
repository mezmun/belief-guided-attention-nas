"""
This module saves and loads the evaluated architecture archive.

The archive is stored as JSON because architecture encodings contain nested
lists and dictionaries. Files are written atomically to reduce the risk of a
partially written state after an interrupted run.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .archive import ArchiveEntry, EvaluatedArchitectureArchive
from .encoder import ArchitectureEncoder, ArchitectureEncoding


class ArchiveStorage:
    """Save and restore an EvaluatedArchitectureArchive object."""

    SCHEMA_VERSION = 1

    @classmethod
    def save_archive(
        cls,
        archive: EvaluatedArchitectureArchive,
        path: Path | str,
    ) -> Path:
        """Write the complete archive to a JSON file using an atomic replace."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": cls.SCHEMA_VERSION,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_summary": archive.summary(),
            "entries": archive.to_records(),
        }

        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)
            file_handle.flush()
            os.fsync(file_handle.fileno())

        temporary_path.replace(output_path)
        return output_path

    @classmethod
    def load_archive(
        cls,
        path: Path | str,
        encoder: Optional[ArchitectureEncoder] = None,
    ) -> EvaluatedArchitectureArchive:
        """Load an archive from JSON and validate the stored schema."""

        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Archive file was not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        cls._validate_payload(payload)
        archive = EvaluatedArchitectureArchive(encoder=encoder)

        for raw_entry in payload["entries"]:
            entry = cls._entry_from_dict(raw_entry)
            if archive.contains(entry.architecture_id):
                raise ValueError(
                    "The archive file contains a repeated architecture_id: "
                    f"{entry.architecture_id}"
                )
            # Storage is part of the same package and restores the exact state.
            archive._entries[entry.architecture_id] = entry

        return archive

    @classmethod
    def _validate_payload(cls, payload: Any) -> None:
        """Check the minimum fields required for a valid archive file."""

        if not isinstance(payload, dict):
            raise TypeError("Archive JSON must contain an object at the top level")

        version = payload.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported archive schema version: {version}. "
                f"Expected: {cls.SCHEMA_VERSION}"
            )

        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise TypeError("Archive JSON field 'entries' must be a list")

    @staticmethod
    def _entry_from_dict(data: Dict[str, Any]) -> ArchiveEntry:
        """Rebuild one ArchiveEntry from its stored dictionary."""

        if not isinstance(data, dict):
            raise TypeError("Each archive entry must be a dictionary")

        encoding_data = data.get("encoding")
        if not isinstance(encoding_data, dict):
            raise TypeError("Archive entry field 'encoding' must be a dictionary")

        encoding_field_names = {field.name for field in fields(ArchitectureEncoding)}
        missing_encoding_fields = encoding_field_names.difference(encoding_data)
        if missing_encoding_fields:
            raise ValueError(
                "Stored architecture encoding is missing fields: "
                f"{sorted(missing_encoding_fields)}"
            )

        encoding = ArchitectureEncoding(
            **{name: encoding_data[name] for name in encoding_field_names}
        )

        return ArchiveEntry(
            architecture_id=str(data["architecture_id"]),
            architecture_string=str(data["architecture_string"]),
            encoding=encoding,
            fitness_mean=float(data["fitness_mean"]),
            fitness_last=float(data["fitness_last"]),
            fitness_best=float(data["fitness_best"]),
            fitness_worst=float(data["fitness_worst"]),
            evaluation_count=int(data["evaluation_count"]),
            first_generation=int(data["first_generation"]),
            last_generation=int(data["last_generation"]),
            first_individual_id=str(data["first_individual_id"]),
            last_individual_id=str(data["last_individual_id"]),
            run_ids=ArchiveStorage._string_list(data.get("run_ids", [])),
            sources=ArchiveStorage._string_list(data.get("sources", [])),
            fitness_history=[float(value) for value in data.get("fitness_history", [])],
        )

    @staticmethod
    def _string_list(values: Iterable[Any]) -> list[str]:
        """Convert stored values into a clean list of strings."""

        return [str(value) for value in values]


def _run_self_test() -> None:
    """Save and load a small archive in a temporary local file."""

    from tempfile import TemporaryDirectory

    encoding = ArchitectureEncoding(
        architecture_id="arch_1",
        architecture_string="[ca-densenet]-[pool]",
        individual_id="indi0001",
        length=2,
        module_sequence=["ca-densenet", "pool"],
        base_sequence=["densenet", "pool"],
        attention_sequence=["ca", "none"],
        base_attention_pairs=["ca-densenet", "none-pool"],
        module_counts={"ca-densenet": 1, "pool": 1},
        base_counts={"densenet": 1, "pool": 1},
        attention_counts={"ca": 1, "none": 1},
        module_bigrams=["ca-densenet->pool"],
        pair_bigrams=["ca-densenet->none-pool"],
        numeric_summary={"length": 2.0, "attention_density": 0.5},
        unit_records=[],
    )

    archive = EvaluatedArchitectureArchive()
    archive.add_encoding(encoding, fitness=0.82, generation=0)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "archive.json"
        ArchiveStorage.save_archive(archive, path)
        restored = ArchiveStorage.load_archive(path)

    assert len(restored) == 1
    assert restored.get("arch_1").architecture_string == encoding.architecture_string
    assert abs(restored.get("arch_1").fitness_mean - 0.82) < 1e-12
    print("Archive storage self-test passed.")


if __name__ == "__main__":
    _run_self_test()

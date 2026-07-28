"""
This module converts an Individual object into deterministic architecture features.

The encoder reads the existing Individual.units structure without changing the
population classes. The produced features will later be used by the archive,
similarity, belief, and uncertainty modules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ArchitectureEncoding:
    """Store a deterministic and serializable representation of one architecture."""

    architecture_id: str
    architecture_string: str
    individual_id: str
    length: int
    module_sequence: List[str]
    base_sequence: List[str]
    attention_sequence: List[str]
    base_attention_pairs: List[str]
    module_counts: Dict[str, int]
    base_counts: Dict[str, int]
    attention_counts: Dict[str, int]
    module_bigrams: List[str]
    pair_bigrams: List[str]
    numeric_summary: Dict[str, float]
    unit_records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Return the encoding as a plain dictionary."""

        return asdict(self)


class ArchitectureEncoder:
    """Extract stable architecture features from the existing Individual object."""

    VERSION = "1.0"

    _TYPE_INFO: Dict[int, Tuple[str, str, str]] = {
        1: ("resnet", "resnet", "none"),
        2: ("pool", "pool", "none"),
        3: ("densenet", "densenet", "none"),
        4: ("inception", "inception", "none"),
        5: ("inception-se", "inception", "se"),
        6: ("inception-cbam", "inception", "cbam"),
        7: ("inception-ca", "inception", "ca"),
        8: ("se-resnet", "resnet", "se"),
        9: ("cbam-resnet", "resnet", "cbam"),
        10: ("ca-resnet", "resnet", "ca"),
        11: ("se-densenet", "densenet", "se"),
        12: ("cbam-densenet", "densenet", "cbam"),
        13: ("ca-densenet", "densenet", "ca"),
        14: ("inception-eca", "inception", "eca"),
        15: ("eca-resnet", "resnet", "eca"),
        16: ("eca-densenet", "densenet", "eca"),
    }

    def encode(self, individual: Any) -> ArchitectureEncoding:
        """Encode one Individual object without changing it."""

        units = getattr(individual, "units", None)
        if units is None:
            raise AttributeError("The individual does not contain a 'units' attribute")

        architecture_id, architecture_string = self._read_uuid(individual)
        unit_records = [self._encode_unit(unit, index) for index, unit in enumerate(units)]

        module_sequence = [record["module"] for record in unit_records]
        base_sequence = [record["base"] for record in unit_records]
        attention_sequence = [record["attention"] for record in unit_records]
        base_attention_pairs = [record["base_attention_pair"] for record in unit_records]

        encoding = ArchitectureEncoding(
            architecture_id=architecture_id,
            architecture_string=architecture_string,
            individual_id=str(getattr(individual, "id", "unknown")),
            length=len(unit_records),
            module_sequence=module_sequence,
            base_sequence=base_sequence,
            attention_sequence=attention_sequence,
            base_attention_pairs=base_attention_pairs,
            module_counts=dict(Counter(module_sequence)),
            base_counts=dict(Counter(base_sequence)),
            attention_counts=dict(Counter(attention_sequence)),
            module_bigrams=self._make_bigrams(module_sequence),
            pair_bigrams=self._make_bigrams(base_attention_pairs),
            numeric_summary=self._build_numeric_summary(unit_records),
            unit_records=unit_records,
        )
        return encoding

    def encode_many(self, individuals: Iterable[Any]) -> List[ArchitectureEncoding]:
        """Encode several Individual objects in their current order."""

        return [self.encode(individual) for individual in individuals]

    def _encode_unit(self, unit: Any, position: int) -> Dict[str, Any]:
        """Convert one architecture unit into a serializable record."""

        unit_type = int(getattr(unit, "type"))
        if unit_type not in self._TYPE_INFO:
            raise ValueError(f"Unsupported unit type: {unit_type}")

        module, base, attention = self._TYPE_INFO[unit_type]
        raw_in_channel = self._optional_number(unit, "in_channel")
        max_input_channel = self._optional_number(unit, "max_input_channel")

        effective_in_channel = raw_in_channel
        if raw_in_channel is not None and max_input_channel is not None:
            effective_in_channel = min(raw_in_channel, max_input_channel)

        out_channel = self._read_out_channel(unit)
        pool_type = self._read_pool_type(unit)

        return {
            "position": position,
            "number": self._optional_number(unit, "number"),
            "type_id": unit_type,
            "module": module,
            "base": base,
            "attention": attention,
            "base_attention_pair": f"{attention}-{base}",
            "amount": self._optional_number(unit, "amount"),
            "raw_in_channel": raw_in_channel,
            "in_channel": effective_in_channel,
            "out_channel": out_channel,
            "channel_ratio": self._safe_ratio(out_channel, effective_in_channel),
            "k": self._optional_number(unit, "k"),
            "max_input_channel": max_input_channel,
            "reduction_ratio": self._optional_number(unit, "reduction_ratio"),
            "k_size": self._optional_number(unit, "k_size"),
            "pool_type": pool_type,
            "inception_type": getattr(unit, "inception_type", None),
            "out_1x1": self._optional_number(unit, "out_1x1"),
            "red_3x3": self._optional_number(unit, "red_3x3"),
            "out_3x3": self._optional_number(unit, "out_3x3"),
            "red_5x5": self._optional_number(unit, "red_5x5"),
            "out_5x5": self._optional_number(unit, "out_5x5"),
            "out_1x1pool": self._optional_number(unit, "out_1x1pool"),
        }

    @staticmethod
    def _read_uuid(individual: Any) -> Tuple[str, str]:
        """Read the existing architecture hash and architecture string."""

        uuid_method = getattr(individual, "uuid", None)
        if not callable(uuid_method):
            raise AttributeError("The individual does not provide a callable uuid() method")

        result = uuid_method()
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("individual.uuid() must return (architecture_id, architecture_string)")

        architecture_id, architecture_string = result
        return str(architecture_id), str(architecture_string)

    @staticmethod
    def _read_out_channel(unit: Any) -> Optional[float]:
        """Read the output channel value for standard and Inception units."""

        direct_value = ArchitectureEncoder._optional_number(unit, "out_channel")
        if direct_value is not None:
            return direct_value

        branch_names = ("out_1x1", "out_3x3", "out_5x5", "out_1x1pool")
        branch_values = [ArchitectureEncoder._optional_number(unit, name) for name in branch_names]
        if all(value is not None for value in branch_values):
            return float(sum(value for value in branch_values if value is not None))

        return None

    @staticmethod
    def _read_pool_type(unit: Any) -> Optional[str]:
        """Convert the existing pool value into a stable label."""

        if int(getattr(unit, "type")) != 2:
            return None

        value = getattr(unit, "max_or_avg", None)
        if value is None:
            return None
        return "max" if float(value) < 0.5 else "avg"

    @staticmethod
    def _optional_number(obj: Any, name: str) -> Optional[float]:
        """Read a numeric attribute and return None when it is not available."""

        value = getattr(obj, name, None)
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Return a safe numeric ratio for channel comparison."""

        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator / denominator)

    @staticmethod
    def _make_bigrams(values: Sequence[str]) -> List[str]:
        """Create ordered transition labels from a sequence."""

        return [f"{left}->{right}" for left, right in zip(values, values[1:])]

    def _build_numeric_summary(self, records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        """Build compact numeric features for later similarity calculations."""

        attention_count = sum(record["attention"] != "none" for record in records)
        pool_count = sum(record["base"] == "pool" for record in records)
        max_pool_count = sum(record["pool_type"] == "max" for record in records)
        avg_pool_count = sum(record["pool_type"] == "avg" for record in records)

        return {
            "length": float(len(records)),
            "attention_count": float(attention_count),
            "attention_density": self._safe_density(attention_count, len(records)),
            "pool_count": float(pool_count),
            "max_pool_count": float(max_pool_count),
            "avg_pool_count": float(avg_pool_count),
            "amount_mean": self._mean_present(records, "amount"),
            "amount_sum": self._sum_present(records, "amount"),
            "in_channel_mean": self._mean_present(records, "in_channel"),
            "out_channel_mean": self._mean_present(records, "out_channel"),
            "channel_ratio_mean": self._mean_present(records, "channel_ratio"),
            "growth_rate_mean": self._mean_present(records, "k"),
            "reduction_ratio_mean": self._mean_present(records, "reduction_ratio"),
            "eca_kernel_mean": self._mean_present(records, "k_size"),
        }

    @staticmethod
    def _safe_density(count: int, total: int) -> float:
        """Return a safe ratio for count-based architecture features."""

        if total == 0:
            return 0.0
        return float(count / total)

    @staticmethod
    def _present_values(records: Sequence[Dict[str, Any]], key: str) -> List[float]:
        """Collect available numeric values from unit records."""

        values: List[float] = []
        for record in records:
            value = record.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _mean_present(self, records: Sequence[Dict[str, Any]], key: str) -> float:
        """Return the mean of available values, or zero when none exist."""

        values = self._present_values(records, key)
        return float(mean(values)) if values else 0.0

    def _sum_present(self, records: Sequence[Dict[str, Any]], key: str) -> float:
        """Return the sum of available values, or zero when none exist."""

        return float(sum(self._present_values(records, key)))


def _run_self_test() -> None:
    """Run a small local test without importing the population module."""

    class FakeUnit:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FakeIndividual:
        def __init__(self) -> None:
            self.id = "test_0001"
            self.units = [
                FakeUnit(type=13, number=0, amount=4, k=20, max_input_channel=128,
                         in_channel=64, out_channel=144, reduction_ratio=16),
                FakeUnit(type=2, number=1, max_or_avg=0.75),
                FakeUnit(type=9, number=2, amount=3, in_channel=144,
                         out_channel=256, reduction_ratio=16),
            ]

        def uuid(self) -> Tuple[str, str]:
            return "test_hash", "[ca-densenet]-[pool]-[cbam-resnet]"

    encoder = ArchitectureEncoder()
    result = encoder.encode(FakeIndividual())

    assert result.architecture_id == "test_hash"
    assert result.length == 3
    assert result.module_sequence == ["ca-densenet", "pool", "cbam-resnet"]
    assert result.attention_sequence == ["ca", "none", "cbam"]
    assert result.module_bigrams == ["ca-densenet->pool", "pool->cbam-resnet"]
    assert result.numeric_summary["attention_count"] == 2.0

    print("Architecture encoder self-test passed.")
    print(result.to_dict())


if __name__ == "__main__":
    _run_self_test()

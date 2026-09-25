"""
Load german.data and return structured records + human-readable English translations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import config


@dataclass
class Record:
    """A single loan application record."""
    idx: int                          # original line number 1..1000
    attrs: List[str] = field(default_factory=list)   # 20 raw attribute values
    label: str = ""                   # "1" = Good, "2" = Bad

    # ── Convenience accessors ──
    @property
    def attr1_checking(self) -> str:   return self.attrs[0]
    @property
    def attr2_duration(self) -> int:   return int(self.attrs[1])
    @property
    def attr3_history(self) -> str:    return self.attrs[2]
    @property
    def attr4_purpose(self) -> str:    return self.attrs[3]
    @property
    def attr5_amount(self) -> int:     return int(self.attrs[4])
    @property
    def attr6_savings(self) -> str:    return self.attrs[5]
    @property
    def attr7_employment(self) -> str: return self.attrs[6]
    @property
    def attr8_installment_pct(self) -> int: return int(self.attrs[7])
    @property
    def attr9_personal(self) -> str:   return self.attrs[8]
    @property
    def attr10_debtors(self) -> str:   return self.attrs[9]
    @property
    def attr11_residence(self) -> int: return int(self.attrs[10])
    @property
    def attr12_property(self) -> str:  return self.attrs[11]
    @property
    def attr13_age(self) -> int:       return int(self.attrs[12])
    @property
    def attr14_other_install(self) -> str: return self.attrs[13]
    @property
    def attr15_housing(self) -> str:   return self.attrs[14]
    @property
    def attr16_credits(self) -> int:   return int(self.attrs[15])
    @property
    def attr17_job(self) -> str:       return self.attrs[16]
    @property
    def attr18_dependents(self) -> int: return int(self.attrs[17])
    @property
    def attr19_phone(self) -> str:     return self.attrs[18]
    @property
    def attr20_foreign(self) -> str:   return self.attrs[19]

    def human_readable(self, attr_idx: int) -> str:
        """Translate the value at attr_idx (1-based) into a human-readable English description."""
        raw = self.attrs[attr_idx - 1]
        meta = config.ATTR_DICT[attr_idx]
        if meta["type"] == "numerical":
            unit = {
                2: " months", 5: " DM", 8: "%", 11: " years",
                13: " years", 16: "", 18: "",
            }.get(attr_idx, "")
            return f"{raw}{unit}"
        return meta["values"].get(raw, raw)


def load_data(path=None) -> List[Record]:
    """Parse german.data and return a list of 1000 Records."""
    path = path or config.DATA_FILE
    records: List[Record] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 21 columns: 20 attributes + 1 label
            records.append(Record(idx=i, attrs=parts[:20], label=parts[20]))
    return records

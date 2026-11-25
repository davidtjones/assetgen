"""What a library material is, in words: a `Description` (kind, pattern, finish, colour, detail).

The kind and pattern come from `vocabulary.yaml` and must match exactly for two materials to
be the same; the finish words are compared by meaning (`match.py`). The detail only steers
what is drawn: an agent's detail sentence wanders ("the lowest courses look the same as the wall
above"), and compared, it kept a footing from matching the wall it said it matched. The colour
words are kept for people and are not drawn: a material is drawn in its natural colour and
tinted to each building's own colour when it is used, so colour never stops a match. (Asked for
"dark blue-grey" slate, klein drew it bright teal, which every use would have had to undo.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

VOCABULARY: dict[str, dict[str, dict]] = yaml.safe_load((Path(__file__).parent / "vocabulary.yaml").read_text())

# Added to every material's words: one bare surface, not a building, in its natural colour.
SURFACE = ("natural muted colours, a flat patch about two metres across seen square on, the same "
           "surface repeating evenly with nothing else in it")
# Never drawn into a material, whatever it is.
NOT_A_SURFACE = ("window, door, opening, roof edge, eaves, chimney, corner, building, facade, house, "
                 "sky, grass, ground line")


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


@dataclass(frozen=True)
class Description:
    """A material in words. `finish` is its surface and wear ("rough-hewn, weathered, lichen in
    the joints"), `colour` its colour in words, `detail` anything else in one short sentence."""

    kind: str
    pattern: str
    finish: str
    colour: str = ""
    detail: str = ""

    @classmethod
    def of(cls, data: dict) -> Description:
        """A description from a dict, checked against the vocabulary. Raises ValueError."""
        kind, pattern = _clean(data.get("kind")).lower(), _clean(data.get("pattern")).lower()
        if kind not in VOCABULARY:
            raise ValueError(f"no kind `{kind}`; the kinds are {', '.join(VOCABULARY)}")
        if pattern not in VOCABULARY[kind]:
            raise ValueError(f"{kind} has no pattern `{pattern}`; its patterns are {', '.join(VOCABULARY[kind])}")
        finish = _clean(data.get("finish"))
        if not finish:
            raise ValueError("a description needs its finish: the surface and its wear, in a few words")
        return cls(kind, pattern, finish, _clean(data.get("colour")), _clean(data.get("detail")))

    @property
    def info(self) -> dict:
        """The pattern's entry in the vocabulary: `draw`, `runs`, and `course` or `tile`."""
        return VOCABULARY[self.kind][self.pattern]

    def words(self) -> str:
        """What klein is asked to photograph."""
        parts = [f"close-up photograph of {self.info['draw']}", self.finish, self.detail, SURFACE]
        return ", ".join(p for p in parts if p)

    def text(self) -> str:
        """The words compared between materials: the finish."""
        return self.finish

    def as_dict(self) -> dict:
        return asdict(self)


def choices() -> list[str]:
    """Every kind and pattern as `kind: pattern`, the form an agent picks from."""
    return [f"{kind}: {pattern}" for kind, patterns in VOCABULARY.items() for pattern in patterns]


def from_choice(choice: str, **rest) -> Description:
    kind, _, pattern = choice.partition(":")
    return Description.of({"kind": kind, "pattern": pattern, **rest})


def schema() -> dict:
    """The JSON schema an agent answers a description in: `material` is one of `choices()`."""
    return {
        "type": "object",
        "properties": {
            "material": {"type": "string", "enum": choices(),
                         "description": "the kind of material and its pattern"},
            "finish": {"type": "string",
                       "description": "its surface and wear in a few words, no colour: e.g. 'rough-hewn, weathered, lichen in the joints'"},
            "colour": {"type": "string", "description": "its colour in a few words"},
            "detail": {"type": "string",
                       "description": "anything else about the surface itself in one short sentence, or empty; "
                                      "never what is round it or next to it"},
        },
        "required": ["material", "finish", "colour", "detail"],
        "additionalProperties": False,
    }

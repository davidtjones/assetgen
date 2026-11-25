"""Whether the library already has a material for a description.

A material matches when its kind and pattern are the description's, and its free words mean
about the same: the cosine similarity of their embeddings is at least `MATCH`. Colour never
stops a match: every material is stored colour-neutral (`make.py`) and tinted to each use's
own colour.

`MATCH` is how cohesive the world looks. Set it loose, and one stone serves every wall and the
world has no variety; set it tight, and every building draws its own, the world has no
cohesion, and the library fills with near twins.
"""

from __future__ import annotations

import numpy as np

from textures.library.words import Description

# From bge-small on short finishes: a paraphrase ("rough-hewn, weathered, lichen in the joints"
# and "weathered rough stone, moss between the joints") scores about 0.83, unrelated finishes
# ("smooth polished, freshly cut") about 0.6.
MATCH = 0.78


def tint(mean: list[float], colour: list[float] | None) -> list[float]:
    """The linear-RGB multiplier that gives a material of mean albedo `mean` the mean `colour`
    (white when no colour is asked for)."""
    if colour is None:
        return [1.0, 1.0, 1.0]
    return [round(float(c / max(m, 1e-4)), 4) for c, m in zip(colour, mean)]


def rank(desc: Description, embedding: np.ndarray, colour: list[float] | None, candidates: list[dict]) -> list[dict]:
    """Every candidate (current versions with their `embedding`) of the description's kind and
    pattern, best first, each with its similarity, its tint for `colour`, and whether it matches."""
    out = []
    for c in candidates:
        if (c["kind"], c["pattern"]) != (desc.kind, desc.pattern) or c["review"] == "rejected":
            continue
        similarity = round(float(np.dot(embedding, c["embedding"])), 4)
        out.append({"name": c["name"], "version": c["version"], "review": c["review"], "similarity": similarity,
                    "tint": tint(c["albedo_mean"], colour), "tile_metres": c["tile_metres"],
                    "matches": similarity >= MATCH})
    return sorted(out, key=lambda r: (r["matches"], r["review"] == "approved", r["similarity"]), reverse=True)

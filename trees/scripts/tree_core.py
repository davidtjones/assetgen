"""Tree model: grow a skeleton, then skin it as two meshes, bark and foliage.

A tree is grown once as a **skeleton**: a trunk, and levels of branches leaving
it, each a bent, tapering line (`grow`). How a species grows is a `Habit`: a
table of numbers, not code. The skeleton is then skinned at a level of detail
(`skin`):

- **bark**: a tube round every branch, textured with a tiling bark image.
- **foliage**: alpha-cut cards hung on the outer wood, textured from the
  species' card atlas (`tree_textures.py`).

They are separate meshes with separate materials, so the game can recolour the
leaves of a tree, swap them, or leave them off, without touching its wood.
Both levels of detail come from the one skeleton, so their outlines agree.

Vertex data, beyond position / normal / tangent / UV:

- `COLOR_0`: baked occlusion. Each vertex is darkened by how much foliage lies
  between it and the sky, so a crown is dark inside and underneath and bright
  on its sunlit shell. Foliage adds a small per-card variation. It carries no
  hue: leaf colour belongs to the game. On foliage only the red channel is
  the shade; the others describe the card (see `skin`).
- `TEXCOORD_1`: wind. `x` is how far the vertex flutters, 0 at the ground
  rising to 1 at loose leaf tips; `y` is a per-branch phase so branches do not
  move in step.

Foliage normals are bent toward "outward from the bough, outward from the
crown" instead of each card's own face, so a crown shades as rounded masses
and not as a pile of flat planes; the cards' normal maps put the leaf-scale
variation back.

Units are metres, y up, base of the trunk at the origin.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import tree_wood
from tree_textures import BARK_METRES, card_rect

UP = np.array([0.0, 1.0, 0.0])
GOLDEN = 2.39996


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else UP.copy()


def _perp(d):
    """A unit vector perpendicular to `d`, level if it can be."""
    side = np.cross(d, UP)
    if np.linalg.norm(side) < 1e-4:
        side = np.cross(d, np.array([1.0, 0.0, 0.0]))
    return _unit(side)


def _rot(v, axis, ang):
    axis = _unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


# --- how a species grows -------------------------------------------------------------

@dataclass
class Level:
    """One order of branching: the branches that leave a parent."""
    count: int                      # per parent
    angle: tuple                    # degrees away from the parent's direction: at its base, at its tip
    length: float                   # of the parent's length (first level: of the tree's height)
    radius: float = 0.5             # of the parent's radius where the branch leaves it
    start: float = 0.25             # the first leaves this far along the parent, 0..1
    end: float = 0.96
    lift: float = 0.0               # bend toward the sky (+) or the ground (-) over the branch
    tip_lift: float = 0.0           # further bend toward the sky near the tip
    wander: float = 0.12            # random bend
    segs: int = 4
    flat: float = 0.0               # 0 leaves all round the parent, 1 only to its two sides
    whorl: int = 0                  # leaves in rings of this many (conifers)
    spread: float = 8.0             # random degrees on `angle`
    taper: float = 0.5              # how much shorter branches get toward the parent's tip
    # How the parent ends. 0: in its own tip, the branches all leaving its side
    # (a spruce's leader). 2 or more: it splits into that many of the `count`,
    # which carry it on, as an oak's trunk divides into limbs. Together they
    # are as thick as the parent was (the pipe rule), and they open like a vase.
    fork: int = 0
    fork_angle: tuple = (26, 40)    # degrees a fork ends up away from the parent's direction


@dataclass
class Foliage:
    kind: str = "spray"             # spray (broadleaf), bough (spruce), tuft (pine), none
    size: float = 1.2               # card length, metres
    aspect: float = 0.9             # card width over length
    spacing: float = 0.5            # metres of twig per cluster
    hang: float = 0.0               # 0 reaches up and out, 1 hangs
    reach: int = 1                  # how many of the outermost levels carry foliage
    mid: float = 3.0                # mid detail keeps one cluster in this many, grown to cover


@dataclass
class Roots:
    """Surface roots: each leaves the trunk as a buttress, runs out along the
    ground half buried, and dives under. Lengths are in trunk radii."""
    count: int = 5
    reach: float = 6.0              # how far a root runs before it is under
    radius: float = 0.40            # where it leaves the trunk
    rise: float = 1.6               # how far up the trunk its buttress starts
    fork: float = 0.6               # chance a root splits in two on its way out
    wander: float = 0.30            # random turn per step, radians


@dataclass
class Crown:
    """A crown grown by space colonisation instead of from `levels`: points
    are scattered through the space the crown may fill, and wood grows toward
    the points nearest it, using each up as it arrives. Every part of the
    space gets reached once, so limbs divide as they go and do not crowd."""
    radius: float                   # of the space at its widest, metres
    base: float = 0.27              # the space starts this far up the tree
    low: float = 0.7                # under 1 puts the widest part low, over 1 high
    points: int = 420               # more make a twiggier crown
    step: float = 0.5               # metres of wood per round of growth
    reach: float = 3.2              # a point pulls on wood within this of it, metres
    clear: float = 1.0              # and is used up when wood comes this near
    up: float = 0.12                # pull toward the sky
    pipe: float = 1.9               # a limb's radius to this power is the sum of its children's (2: equal area)
    fork: float = 0.7               # a child this thick beside the thickest splits the limb; thinner is a side branch
    limber: float = 0.045           # wind weight gained per metre of branch


@dataclass
class Habit:
    height: float
    radius: float                   # trunk radius at the ground
    levels: list
    foliage: Foliage
    trunk: float = 1.0              # the trunk ends this far up the tree
    shape: str = "round"            # how first-level length varies up the crown
    lean: float = 0.02              # trunk wander
    flare: float = 1.6              # root flare
    top: float = 0.03               # trunk radius at its top, metres
    taper: float = 0.85             # 1 a straight cone; lower keeps the trunk thick higher up
    trunk_segs: int = 14
    twig: float = 0.012             # thinnest wood kept, metres
    roots: Roots | None = field(default_factory=Roots)
    # Trunk, limbs and roots as one surface with collars where they meet
    # (`tree_wood`), instead of tubes laid through each other. (near, mid)
    # triangles for it, or None for tubes.
    solid: tuple | None = None
    # Wood this many times as thick as the thinnest twig, or thinner, gets
    # no mesh: it is under the leaves. 0 keeps everything.
    hidden: float = 0.0
    # Grow the crown by space colonisation, and leave `levels` empty.
    crown: Crown | None = None


def _profile(shape: str, t: float) -> float:
    """First-level branch length, 0..1, `t` of the way up the crown."""
    if shape == "cone":             # spruce: longest at the bottom
        return 0.10 + 0.90 * (1 - t) ** 0.75
    if shape == "column":           # birch, aspen: a narrow oval
        return 0.35 + 0.65 * math.sin(math.pi * (0.12 + 0.88 * t) ** 0.8) ** 0.8
    if shape == "umbrella":         # pine: widest high up
        return 0.45 + 0.55 * math.sin(math.pi * min(t * 0.8 + 0.25, 1.0))
    return 0.55 + 0.45 * math.sin(math.pi * (0.1 + 0.8 * t))   # round


# --- skeleton ------------------------------------------------------------------------

@dataclass
class Branch:
    pts: np.ndarray                 # k x 3
    radii: np.ndarray               # k
    sway: np.ndarray                # k, wind weight
    level: int                      # 0 is the trunk
    phase: float
    limb: int                       # which first-level branch it belongs to; -1 the trunk
    length: float
    forks: bool = False             # ends by splitting, not in a tip

    def at(self, t):
        """Point, direction and radius `t` of the way along."""
        k = len(self.pts) - 1
        x = min(max(t, 0.0), 1.0) * k
        i = min(int(x), k - 1)
        f = x - i
        return (self.pts[i] + (self.pts[i + 1] - self.pts[i]) * f, _unit(self.pts[i + 1] - self.pts[i]),
                float(self.radii[i] + (self.radii[i + 1] - self.radii[i]) * f),
                float(self.sway[i] + (self.sway[i + 1] - self.sway[i]) * f))


@dataclass
class Skeleton:
    habit: Habit
    branches: list = field(default_factory=list)
    # Where foliage hangs: (point, direction, limb, sway, phase, branch length, branch index).
    buds: list = field(default_factory=list)
    roots: list = field(default_factory=list)     # branches too, but still and leafless
    wood: object = None                           # `tree_wood.Wood`, made once for both details
    twig: float = 0.0                             # radius of the thinnest wood grown


def grow(habit: Habit, rng) -> Skeleton:
    sk = Skeleton(habit, twig=habit.twig)
    h = habit.height * habit.trunk
    n = habit.trunk_segs
    ts = np.linspace(0, 1, n + 1)
    drift = np.cumsum(rng.normal(0, habit.lean, (n + 1, 2)), 0) * h / n * 2.0
    drift -= drift[0]
    pts = np.stack([drift[:, 0], ts * h, drift[:, 1]], 1)
    radii = habit.top + (habit.radius - habit.top) * (1 - ts) ** habit.taper
    # The trunk does not flutter (the game rocks the whole tree by height
    # instead). Flutter is weight times a motion picked by phase, so wood
    # stays joined only where both agree: a limb starts at weight 0 on the
    # trunk, and everything growing from it shares its phase.
    sway = np.zeros(n + 1)
    if habit.crown:
        _colonise(sk, pts, sway, rng)
    else:
        trunk = Branch(pts, radii, sway, 0, 0.0, -1, h)
        sk.branches.append(trunk)
        _children(sk, trunk, 0, rng, -1)
    if habit.roots:
        # Their own stream of numbers, so adding roots leaves the crown as it was.
        _roots(sk, np.random.default_rng([int(rng.integers(1 << 30)), 11]))
    return sk


def _colonise(sk: Skeleton, trunk: np.ndarray, trunk_sway: np.ndarray, rng):
    """Grow the crown from the top of `trunk` into the space `habit.crown`
    describes, then read the result off as branches and buds."""
    from scipy.spatial import cKDTree
    habit, c = sk.habit, sk.habit.crown
    bottom, top = c.base * habit.height, habit.height
    # The space: round in plan, an egg in profile, standing over the trunk's top.
    points = []
    while len(points) < c.points:
        q = rng.uniform([-c.radius, bottom, -c.radius], [c.radius, top, c.radius], (c.points * 2, 3))
        t = (q[:, 1] - bottom) / (top - bottom)
        wide = c.radius * np.sqrt(np.maximum(1 - np.abs(2 * t ** c.low - 1) ** 2.5, 0))
        points += list(q[np.hypot(q[:, 0], q[:, 2]) < wide] + [trunk[-1][0], 0, trunk[-1][2]])
    points = np.array(points[: c.points])

    pos, parent = [p for p in trunk], [-1] + list(range(len(trunk) - 1))
    for _ in range(300):
        tree = cKDTree(np.array(pos))
        gap, near = tree.query(points)
        points, gap, near = points[gap > c.clear], gap[gap > c.clear], near[gap > c.clear]
        pulled = gap < c.reach
        if not pulled.any():
            break
        at = np.array(pos)[near[pulled]]
        pull = points[pulled] - at
        pull /= np.linalg.norm(pull, axis=1, keepdims=True)
        total = np.zeros((len(pos), 3))
        np.add.at(total, near[pulled], pull)
        grown = 0
        for n in np.unique(near[pulled]):
            d = _unit(_unit(total[n]) + UP * c.up + rng.normal(0, 0.05, 3))
            new = pos[n] + d * c.step
            if tree.query(new)[0] > c.step * 0.4:       # not where wood already is
                pos.append(new)
                parent.append(int(n))
                grown += 1
        if not grown:
            break
    pos, parent = np.array(pos), np.array(parent)
    count, bole = len(pos), len(trunk)
    kids = [[] for _ in range(count)]
    for n in range(1, count):
        kids[parent[n]].append(n)
    # Take the zigzag out of the new wood.
    for _ in range(2):
        eased = pos.copy()
        for n in range(bole, count):
            ahead = pos[kids[n]].mean(0) if kids[n] else pos[n]
            eased[n] = 0.5 * pos[n] + 0.25 * pos[parent[n]] + 0.25 * ahead
        pos = eased

    # Radii by the pipe rule, from the twig ends down. Twigs are as thick as
    # makes the trunk come out right where its lowest branch leaves it: below
    # that it is the habit's trunk, above it the rule's.
    tips = sum(1 for n in range(count) if not kids[n])
    first = next((n for n in range(bole) if any(k >= bole for k in kids[n])), bole - 1)
    wanted = habit.top + (habit.radius - habit.top) * (1 - first / (bole - 1)) ** habit.taper
    twig = sk.twig = max(wanted / tips ** (1 / c.pipe), habit.twig * 0.6)
    area = np.zeros(count)
    for n in range(count - 1, -1, -1):
        area[n] = sum(area[k] for k in kids[n]) if kids[n] else twig ** c.pipe
    radius = area ** (1 / c.pipe)
    t = np.linspace(0, 1, first + 1)
    radius[:first + 1] = radius[first] + (habit.radius - radius[first]) * (1 - t) ** habit.taper
    sway = np.zeros(count)                      # filled in branch by branch below

    # Branches: at each node the thickest child carries the branch on and the
    # rest leave it, unless two are near enough in size to be a fork.
    home = np.zeros(count, int)                 # which branch each node is in
    todo = [([0], 0, -1, 0.0)]                  # (nodes so far, level, limb, phase)
    limbs = 0
    while todo:
        nodes, level, limb, phase = todo.pop()
        forks = False
        while True:
            n = nodes[-1]
            ks = sorted(kids[n], key=lambda k: -radius[k])
            if not ks:
                break
            forks = len(ks) > 1 and radius[ks[1]] >= c.fork * radius[ks[0]]
            for k in ks if forks else ks[1:]:
                if limb < 0:
                    todo.append(([n, k], level + 1, limbs, float(rng.random())))
                    limbs += 1
                else:
                    todo.append(([n, k], level + 1, limb, phase))
            if forks:
                break
            nodes.append(ks[0])
        idx = np.array(nodes)
        if level:
            # Wind weight grows along a limb from nothing where it leaves the
            # trunk. The trunk, as far up as it runs, stays at nothing.
            for a, b in zip(nodes, nodes[1:]):
                sway[b] = min(sway[a] + c.limber * np.linalg.norm(pos[b] - pos[a]), 0.7)
        radii = radius[idx].copy()
        if level:
            radii[0] = radii[1]                 # it starts as thick as itself, not as its parent
        length = float(np.linalg.norm(np.diff(pos[idx], axis=0), axis=1).sum())
        home[idx[1:] if level else idx] = len(sk.branches)
        sk.branches.append(Branch(pos[idx], radii, sway[idx], level, phase, limb, length, forks))
    # Leaves on the thin outer wood.
    if habit.foliage.kind != "none":
        for n in range(bole, count):
            if radius[n] <= twig * 1.5:
                b = sk.branches[home[n]]
                sk.buds.append((pos[n], _unit(pos[n] - pos[parent[n]]), b.limb, sway[n], b.phase, b.length, home[n]))


def _roots(sk: Skeleton, rng):
    spec, trunk = sk.habit.roots, sk.habit.radius
    spin = rng.uniform(0, 2 * math.pi)
    for k in range(spec.count):
        size = rng.uniform(0.65, 1.0)
        heading = spin + 2 * math.pi * (k + rng.normal(0, 0.16)) / spec.count
        start = np.array([math.cos(heading), math.sin(heading)]) * trunk * 0.35
        _root(sk, rng, start, heading, spec.reach * trunk * size * rng.uniform(0.8, 1.2),
              spec.radius * trunk * size, spec.rise * trunk * size, spec.fork)


def _root(sk: Skeleton, rng, start, heading, reach, r0, rise, fork):
    """One root from `start` (x, z) toward `heading`, `reach` metres long."""
    spec = sk.habit.roots
    n = 10
    at, run = np.array(start, float), 0.0
    phase = rng.uniform(0, 2 * math.pi)
    pts, radii, headings = [], [], []
    for i in range(n + 1):
        s = (i / n) ** 1.4                      # points crowd toward the trunk, where it curves
        if i > 1:
            heading += rng.normal(0, spec.wander)
        at = at + np.array([math.cos(heading), math.sin(heading)]) * (reach * s - run)
        run = reach * s
        r = max(r0 * (1 - 0.6 * s), 0.012)        # still thick where it goes under
        y = rise * math.exp(-run / max(rise * 0.6, 1e-3)) if rise > 0 else 0.0   # the buttress
        y += r * (0.15 + 0.3 * math.sin(phase + 6 * s) * min(1.0, 3 * s))       # half buried, in and out
        dive = max(0.0, (s - 0.4) / 0.6)
        y -= (r + 0.3) * dive * dive
        pts.append([at[0], y, at[1]])
        radii.append(r)
        headings.append(heading)
    sk.roots.append(Branch(np.array(pts), np.array(radii), np.zeros(n + 1), 1, 0.0, -1, reach))
    if r0 > 0.04 and rng.random() < fork:
        j = n // 2 + int(rng.integers(0, 2))
        turn = rng.uniform(0.45, 0.8) * (1 if rng.random() < 0.5 else -1)
        _root(sk, rng, [pts[j][0], pts[j][2]], headings[j] + turn, reach * 0.5, radii[j] * 0.8, 0.0, 0.0)


def _room(taken, start, direction, length) -> float:
    """How much open space a branch from `start` along `direction` would grow
    into: metres from its middle and its end to the nearest wood already
    there, less for one that points at the ground."""
    if taken is None:
        return 0.0
    reach = start + np.outer([0.5, 1.0], direction) * length
    gap = np.sqrt(((taken[None] - reach[:, None]) ** 2).sum(-1).min(1))
    return float(min(gap[0] * 1.5, gap[1])) + min(direction[1], 0.0) * length


def _children(sk: Skeleton, parent: Branch, li: int, rng, limb: int):
    habit = sk.habit
    last = li >= len(habit.levels)
    if li >= len(habit.levels) - habit.foliage.reach or last:
        # Outer wood carries foliage: buds along its outer part.
        if parent.level > 0 and habit.foliage.kind != "none":
            first = 0.25 if last else 0.55
            count = max(1, int(round(parent.length * (1 - first) / habit.foliage.spacing)))
            for k in range(count):
                t = first + (1 - first) * (k + 1) / count
                p, d, _, sw = parent.at(t)
                sk.buds.append((p, d, parent.limb, sw, parent.phase, parent.length, len(sk.branches) - 1))
    if last:
        return
    lv = habit.levels[li]
    forks = min(lv.fork, lv.count)
    sides = lv.count - forks
    # A branch that will itself fork stays thick to its end.
    tip = 0.62 if li + 1 < len(habit.levels) and habit.levels[li + 1].fork else 0.2
    rings = max(1, sides // lv.whorl) if lv.whorl else 0
    spin = rng.uniform(0, 2 * math.pi)
    side = 1

    def taken():
        # The rest of the tree so far. Each branch's whole subtree is grown
        # before its next sibling, so later ones see where earlier ones went.
        others = [b.pts for b in sk.branches if b is not parent]
        return np.concatenate(others) if others else None

    if forks:
        parent.forks = True
        p, d, r, sw = parent.at(1.0)
        share = rng.uniform(0.85, 1.15, forks)
        share /= share.sum()
        length = lv.length * (habit.height if parent.level == 0 else parent.length)
        opening = math.radians(rng.uniform(*lv.fork_angle))
        # Which way round the split faces. A leaning limb that splits in two
        # does it side by side, not one fork over the other; otherwise the
        # split turns to whichever way leaves its forks the most room.
        if forks == 2 and abs(d[1]) < 0.85:
            turn = rng.normal(0, 0.25)
        else:
            wood = taken()
            turns = rng.uniform(0, 2 * math.pi) + np.arange(8) * (2 * math.pi / forks / 8)
            turn = max(turns, key=lambda a: min(
                _room(wood, p, _unit(d * math.cos(opening) + _rot(_perp(d), d, a + 2 * math.pi * k / forks)
                                     * math.sin(opening)), length) for k in range(forks)))
        for k in range(forks):
            radial = _rot(_perp(d), d, turn + 2 * math.pi * (k + rng.normal(0, 0.06)) / forks)
            ang = opening * rng.uniform(0.9, 1.1)
            # Leaves along the parent and bends away: no corner at the split.
            direction = _unit(d * math.cos(ang * 0.4) + radial * math.sin(ang * 0.4))
            child = _branch(p, direction, length * rng.uniform(0.9, 1.1), max(r * math.sqrt(share[k]), habit.twig),
                            lv, parent.level + 1, sw,
                            rng, limb if limb >= 0 else sides + k, habit.twig, tip, (radial, ang * 0.6))
            if parent.level:
                child.phase = parent.phase
            sk.branches.append(child)
            _children(sk, child, li + 1, rng, child.limb)
    for k in range(sides):
        if lv.whorl:
            ring = k // lv.whorl
            if k % lv.whorl == 0:
                spin = rng.uniform(0, 2 * math.pi)
            t = lv.start + (lv.end - lv.start) * ((ring + 0.5) / rings) ** 1.15 + rng.normal(0, 0.004)
            az = spin + 2 * math.pi * ((k % lv.whorl) + rng.normal(0, 0.12)) / lv.whorl
        else:
            t = lv.start + (lv.end - lv.start) * (k + rng.uniform(0.15, 0.85)) / sides
            az = spin + k * GOLDEN + rng.normal(0, 0.3)
        p, d, r, sw = parent.at(t)
        tc = (t - lv.start) / max(lv.end - lv.start, 1e-6)
        ang = math.radians(lv.angle[0] + (lv.angle[1] - lv.angle[0]) * tc + rng.normal(0, lv.spread))
        if lv.flat > 0 and abs(d[1]) < 0.9:
            # To the parent's sides, alternating, as boughs lie in a plane.
            side = -side
            level = _perp(d) * side
            radial = _unit(_rot(level, d, rng.normal(0, (1 - lv.flat) * 1.2)))
        else:
            radial = _rot(_perp(d), d, az)
        if parent.level == 0:
            length = lv.length * habit.height * _profile(habit.shape, tc)
        else:
            length = lv.length * parent.length * (1 - lv.taper * tc)
        length *= 0.85 + 0.3 * rng.random()
        if not lv.whorl and not (lv.flat > 0 and abs(d[1]) < 0.9):
            # Round the parent by the golden angle, then turned toward the
            # most open space near that, so branches do not bunch on one side.
            wood = taken()
            radial = max((_rot(radial, d, off) for off in (0.0, 0.6, -0.6, 1.2, -1.2, 1.9, -1.9)),
                         key=lambda v: _room(wood, p, _unit(d * math.cos(ang) + v * math.sin(ang)), length))
        direction = _unit(d * math.cos(ang) + radial * math.sin(ang))
        # No thicker than its length can carry.
        r0 = max(min(lv.radius * r, length * (0.055 if parent.level == 0 else 0.038)), habit.twig)
        child = _branch(p, direction, length, r0, lv, parent.level + 1, sw, rng,
                        limb if limb >= 0 else k, habit.twig, tip)
        if parent.level:
            child.phase = parent.phase
        sk.branches.append(child)
        _children(sk, child, li + 1, rng, child.limb)


def _branch(p, d, length, r0, lv: Level, level, sway0, rng, limb, twig, tip=0.2, splay=None) -> Branch:
    """`tip`: its radius at the end, of `r0`. `splay`: (direction, radians) to
    bend toward over its first half."""
    pts, radii, sway = [np.asarray(p, float)], [r0], [sway0]
    for i in range(lv.segs):
        f = (i + 1) / lv.segs
        d = d + UP * ((lv.lift + lv.tip_lift * f) / lv.segs) + rng.normal(0, lv.wander, 3) / math.sqrt(lv.segs)
        if splay and f <= 0.5:
            d = d + splay[0] * (math.tan(splay[1]) / max(lv.segs // 2, 1))
        d = _unit(d)
        pts.append(pts[-1] + d * length / lv.segs)
        radii.append(max(r0 * (1 - (1 - tip) * f), twig * 0.6))
        sway.append(min(sway0 + (0.16 + 0.07 * level) * f, 0.7))
    return Branch(np.array(pts), np.array(radii), np.array(sway), level, float(rng.random()), limb, length)


# --- meshes --------------------------------------------------------------------------

class Mesh:
    """Triangles under construction."""

    def __init__(self):
        self.pos, self.nrm, self.tan, self.uv, self.uv1, self.col, self.idx = [], [], [], [], [], [], []

    def extend(self, pos, nrm, tan, uv, uv1, col, idx):
        base = sum(len(p) for p in self.pos)
        self.pos.append(pos)
        self.nrm.append(nrm)
        self.tan.append(tan)
        self.uv.append(uv)
        self.uv1.append(uv1)
        self.col.append(col)
        self.idx.append(np.asarray(idx, np.int64) + base)

    def done(self):
        if not self.pos:
            return None
        cat = lambda xs: np.concatenate(xs).astype(np.float32)  # noqa: E731
        col = cat(self.col)
        return {"positions": cat(self.pos), "normals": cat(self.nrm), "tangents": cat(self.tan),
                "uvs": cat(self.uv), "uv1": cat(self.uv1),
                "colors": np.concatenate([col, np.ones((len(col), 1), np.float32)], 1),
                "indices": np.concatenate(self.idx).astype(np.uint32)}


def _tube(mesh: Mesh, b: Branch, sides: int, step: int, rng_phase: float, flare: float = 1.0, cap=False):
    """Skin one branch. `step` keeps every `step`-th point of its line."""
    keep = list(range(0, len(b.pts), step))
    if keep[-1] != len(b.pts) - 1:
        keep.append(len(b.pts) - 1)
    pts, radii, sway = b.pts[keep].copy(), b.radii[keep].copy(), b.sway[keep]
    k = len(pts)
    # The texture goes a whole number of times round, chosen at the thick end.
    wraps = max(1, int(round(2 * math.pi * radii[0] / BARK_METRES[0])))
    side = _perp(pts[1] - pts[0])
    ang = np.linspace(0, 2 * math.pi, sides + 1)
    rings, vs = [], []
    run = 0.0
    for i in range(k):
        d = _unit(pts[min(i + 1, k - 1)] - pts[max(i - 1, 0)])
        side = _unit(side - d * np.dot(side, d))      # parallel transport: no twist
        other = np.cross(d, side)
        if i:
            run += np.linalg.norm(pts[i] - pts[i - 1])
        r = np.full(sides + 1, radii[i])
        if flare > 1.0:
            # Root flare: the trunk widens toward the ground into a few buttresses.
            y = max(pts[i][1], 0.0)
            near = math.exp(-y / (radii[0] * 2.2))
            r = r * (1 + (flare - 1) * near * (1 + 0.32 * np.cos(ang * 5 + rng_phase) + 0.2 * np.cos(ang * 3 + 2 * rng_phase)))
            r[-1] = r[0]
        ring = pts[i] + np.outer(np.cos(ang) * r, side) + np.outer(np.sin(ang) * r, other)
        rings.append(ring)
        vs.append(run / BARK_METRES[1])
    rings = np.array(rings)                               # k x (sides+1) x 3
    # Normals from the surface itself, so flare and taper shade correctly.
    along = np.gradient(rings, axis=0) if k > 2 else np.broadcast_to(rings[-1] - rings[0], rings.shape)
    around = np.roll(rings[:, :-1], -1, 1) - np.roll(rings[:, :-1], 1, 1)
    around = np.concatenate([around, around[:, :1]], 1)
    nrm = np.cross(around, along)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    tan = around / np.maximum(np.linalg.norm(around, axis=-1, keepdims=True), 1e-9)
    # v runs up the branch, which is down the image: the bitangent is flipped.
    tan = np.concatenate([tan, -np.ones((k, sides + 1, 1))], -1)
    u = np.broadcast_to(np.linspace(0, wraps, sides + 1), (k, sides + 1))
    v = np.broadcast_to(np.array(vs)[:, None], (k, sides + 1))
    uv1 = np.stack([np.broadcast_to(sway[:, None], (k, sides + 1)), np.full((k, sides + 1), b.phase)], -1)
    idx = []
    w = sides + 1
    for i in range(k - 1):
        for s in range(sides):
            a, c = i * w + s, (i + 1) * w + s
            idx += [a, a + 1, c, c, a + 1, c + 1]
    n = k * w
    pos, nrm, tan = rings.reshape(n, 3), nrm.reshape(n, 3), tan.reshape(n, 4)
    uvs, uv1 = np.stack([u, v], -1).reshape(n, 2), uv1.reshape(n, 2)
    if cap:
        # A broken top: close the tube with a fan.
        d = _unit(pts[-1] - pts[-2])
        top = n - w
        centre = pts[-1] + d * radii[-1] * 0.4
        pos = np.concatenate([pos, rings[-1], centre[None]])
        nrm = np.concatenate([nrm, np.broadcast_to(d, (w + 1, 3))])
        tan = np.concatenate([tan, np.broadcast_to(np.append(side, 1.0), (w + 1, 4))])
        ring_uv = np.stack([np.cos(ang), np.sin(ang)], -1) * radii[-1] / BARK_METRES[0]
        uvs = np.concatenate([uvs, ring_uv, [[0.0, 0.0]]])
        uv1 = np.concatenate([uv1, uv1[top:], uv1[-1:]])
        for s in range(sides):
            idx += [n + s, n + s + 1, n + w]
        n += w + 1
    mesh.extend(pos, nrm, tan, uvs, uv1, np.ones((n, 3)), idx)


def _card(mesh: Mesh, base, along, right, length, width, rect, sway, phase, segments, curl, shade, cards,
          narrows=True):
    """A foliage card rooted at `base`, growing `length` along `along` and
    `width` across `right`. With more than one segment it bends by `curl`
    radians toward the ground. Records the card in `cards` for shading.
    `narrows` is whether the game may narrow the card away when it is seen
    edge-on: right for a spray of leaves, wrong for a flat bough, which from
    the side should be the thin layer it is."""
    u0, v0, u1, v1 = card_rect(rect)
    along, right = _unit(along), _unit(right - along * np.dot(right, along))
    # Bend about `right`, whichever way takes the tip down.
    sign = -1.0 if _rot(along, right, 0.3)[1] > _rot(along, right, -0.3)[1] else 1.0
    p, d = np.asarray(base, float), along
    pos, uv, uv1, dirs = [], [], [], []
    for i in range(segments + 1):
        t = i / segments
        if i:
            d = _rot(d, right, sign * curl / segments)
            p = p + d * (length / segments)
        v = v1 + (v0 - v1) * t
        for sx, u in ((-0.5, u0), (0.5, u1)):
            pos.append(p + right * (width * sx))
            uv.append((u, v))
            uv1.append((sway[0] + (sway[1] - sway[0]) * t, phase))
            dirs.append(d)
    idx = []
    for i in range(segments):
        a = i * 2
        idx += [a, a + 1, a + 2, a + 2, a + 1, a + 3]
    n = len(pos)
    cards.append((len(mesh.pos), np.array(dirs), right, shade, width if narrows else 0.0))
    mesh.extend(np.array(pos), np.zeros((n, 3)), np.zeros((n, 4)), np.array(uv), np.array(uv1), np.ones((n, 3)), idx)


def _foliage(sk: Skeleton, detail: float, rng) -> tuple[Mesh, list, np.ndarray]:
    """Hang cards on the skeleton's buds. Returns the mesh, the card records and
    each card's limb."""
    f = sk.habit.foliage
    mesh, cards, limbs = Mesh(), [], []
    near = detail > 0.7
    buds = sk.buds
    scale = 1.0
    if not near and f.kind != "bough":
        order = rng.permutation(len(buds))
        buds = [buds[i] for i in sorted(order[: max(1, int(len(buds) / f.mid))])]
        scale = f.mid ** 0.5
    centre = np.mean([b[0] for b in sk.buds], 0) if sk.buds else np.zeros(3)
    if f.kind == "bough":
        # One card along each whole branch, and sprays hanging under the long ones.
        done = set()
        for p, d, limb, sw, phase, length, bi in sk.buds:
            if bi in done:
                continue
            done.add(bi)
            b = sk.branches[bi]
            start, d0, _, s0 = b.at(0.0)
            tip = b.pts[-1]
            d0 = _unit(tip - start)
            side = _perp(d0)
            ln = np.linalg.norm(tip - start)
            wide = ln * f.aspect * (1.0 if near else 1.2)
            _card(mesh, start - d0 * 0.12, _unit(d0 + UP * 0.12), side, ln * 1.1, wide, int(rng.integers(2)),
                  (s0, s0 + 0.55), phase, 3 if near else 1, 0.34, 1.0 + rng.normal(0, 0.06), cards, narrows=False)
            limbs.append(limb)
            if ln > 1.1:
                for sgn in ((-1, 1) if near else (1,)):
                    down = _rot(-UP, d0, sgn * math.radians(22 + 12 * rng.random()))
                    _card(mesh, start + d0 * ln * 0.14 + UP * 0.04, d0, down, ln * 0.9, ln * 0.6 * (1.0 if near else 1.25),
                          2 + int(rng.integers(2)), (s0, s0 + 0.55), phase, 1, 0.0, 0.92 + rng.normal(0, 0.06), cards)
                    limbs.append(limb)
        return mesh, cards, np.array(limbs)
    for p, d, limb, sw, phase, length, bi in buds:
        size = f.size * (0.8 + 0.4 * rng.random()) * scale
        out = _unit((p - centre) * np.array([1.0, 0.4, 1.0]))
        lean = _unit(d * 0.7 + out * 0.5 + UP * (0.35 - 1.3 * f.hang) + rng.normal(0, 0.25, 3))
        side = _perp(lean)
        sway = (min(sw + 0.25, 0.85), min(sw + 0.55, 1.0))
        if f.kind == "tuft":
            # Needles stand round the shoot: crossed cards.
            lean = _unit(d * 0.6 + UP * 0.7 + rng.normal(0, 0.25, 3))
            spin = rng.uniform(0, math.pi)
            planes = 3 if near else 2
            for k in range(planes):
                right = _rot(_perp(lean), lean, spin + k * math.pi / planes)
                _card(mesh, p - lean * size * 0.45, lean, right, size, size * f.aspect, int(rng.integers(4)),
                      sway, phase, 1, 0.0, 1.0 + rng.normal(0, 0.07), cards)
                limbs.append(limb)
            continue
        # A spray: one card lying open to the sky, one standing on edge through
        # it, so it has body from the side and from below.
        for roll in (rng.normal(0, 0.45), math.pi / 2 + rng.normal(0, 0.45)):
            right = _rot(side, lean, roll)
            _card(mesh, p - lean * size * 0.18, lean, right, size, size * f.aspect, int(rng.integers(4)),
                  sway, phase, 2 if near else 1, 0.35 + 0.5 * f.hang, 1.0 + rng.normal(0, 0.08), cards)
            limbs.append(limb)
    return mesh, cards, np.array(limbs)


def _occlusion(points: np.ndarray, weights: np.ndarray, where: np.ndarray, cell: float) -> np.ndarray:
    """How open to the sky each of `where` is, 0..1: foliage (`points`, each
    worth `weights` square metres) is binned into a grid and rays are marched
    through it toward the sky."""
    if len(points) == 0:
        return np.ones(len(where))
    lo = points.min(0) - cell * 2
    shape = np.ceil((points.max(0) + cell * 2 - lo) / cell).astype(int) + 1
    grid = np.zeros(shape)
    np.add.at(grid, tuple(((points - lo) / cell).astype(int).T), weights / cell ** 2)
    from scipy import ndimage
    grid = ndimage.uniform_filter(grid, 3, mode="constant")
    rays = [UP] + [_unit([math.cos(a), 1.1, math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
    rays += [_unit([math.cos(a), 0.25, math.sin(a)]) for a in np.linspace(0.5, 2 * math.pi + 0.5, 4, endpoint=False)]
    total = np.zeros(len(where))
    for ray in rays:
        depth = np.zeros(len(where))
        for step in range(1, 14):
            q = ((where + ray * (step * cell) - lo) / cell).astype(int)
            ok = ((q >= 0) & (q < shape)).all(1)
            depth[ok] += grid[q[ok, 0], q[ok, 1], q[ok, 2]]
        total += np.exp(-0.22 * depth * cell)
    return total / len(rays)


def skin(sk: Skeleton, detail: float, seed: int, shaded_by=None):
    """The skeleton's bark and foliage meshes at `detail` (1 near, lower for
    mid distance), as dicts of arrays for `gltf_io.write_meshes`, and its
    crown: where its foliage is and how much, for shading. Foliage is `None`
    for a bare tree.

    Pass the near skin's crown as `shaded_by` when skinning a lower detail, so both
    are shaded by the same foliage and one is not brighter than the other."""
    habit = sk.habit
    rng = np.random.default_rng([seed, 7])
    near = detail > 0.7
    bark = Mesh()
    deepest = len(habit.levels)

    def fused(b):
        return habit.solid and (b.level == 0 or b.radii[0] >= 0.06)

    for b in sk.branches:
        if fused(b):
            continue
        if b.level == 0:
            _tube(bark, b, 12 if near else 6, 1 if near else 2, seed * 1.7, flare=habit.flare,
                  cap=habit.top > 0.05 and habit.trunk >= 1.0)
            continue
        if not near and (b.level >= max(deepest, 2) or b.radii[0] < habit.twig * 2.5):
            continue
        if b.radii[0] <= sk.twig * habit.hidden:
            continue
        thick = b.radii[0]
        sides = (8 if thick > 0.12 else 6 if thick > 0.06 else 4 if thick > 0.03 else 3) if near else 3
        _tube(bark, b, sides, 1 if near and thick > 0.03 else 2, 0.0)
    rooted = sum(len(p) for p in bark.pos)
    for b in sk.roots:
        if not habit.solid and (near or b.radii[0] > 0.08):
            _tube(bark, b, (8 if b.radii[0] > 0.1 else 6) if near else 4, 1 if near else 2, 0.0)
    whole = sum(len(p) for p in bark.pos)
    if habit.solid:
        if sk.wood is None:
            sk.wood = tree_wood.Wood(
                [tree_wood.limb(b, BARK_METRES[0], flare=(habit.flare, habit.radius) if b.level == 0 else None,
                                blend=0.02 if b.level == 0 else None)
                 for b in [b for b in sk.branches if fused(b)] + sk.roots])
        pos, nrm, tan, uv, uv1, idx = sk.wood.mesh(habit.solid[0 if near else 1], BARK_METRES[1])
        bark.extend(pos, nrm, tan, uv, uv1, np.ones((len(pos), 3)), idx)
    leaves, cards, limbs = _foliage(sk, detail, rng)

    # Sink the foot so a slope does not show a gap under the trunk.
    bark_out = bark.done()
    foot = bark_out["positions"][:, 1] < 0.02
    foot[rooted:] = False
    bark_out["positions"][foot, 1] -= 0.5
    # Roots get a keel instead: their undersides reach down, so one that
    # runs out over a dip does not float.
    under = bark_out["normals"][:, 1] < -0.3
    under[:rooted] = False
    under[whole:] = False
    bark_out["positions"][under, 1] -= 0.4
    leaf_out = leaves.done()
    if leaf_out is None:
        return bark_out, None, None

    pos = leaf_out["positions"]
    count = len(cards)
    sizes = np.array([len(p) for p in leaves.pos])
    owner = np.repeat(np.arange(count), sizes)
    centres = np.array([p.mean(0) for p in leaves.pos])
    areas = np.array([np.linalg.norm(p[-1] - p[0]) * np.linalg.norm(p[1] - p[0]) for p in leaves.pos])
    crown = centres.mean(0)
    spread = np.maximum(centres.std(0), 0.3)
    # Each bough is its own rounded mass.
    clump = np.zeros_like(centres)
    for limb in np.unique(limbs):
        mine = limbs == limb
        clump[mine] = centres[mine].mean(0)
    out_crown = (pos - crown) / spread
    out_clump = pos - clump[owner]
    norm = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)  # noqa: E731
    outward = norm(0.55 * norm(out_clump) + 0.45 * norm(out_crown) + np.array([0.0, 0.3, 0.0]))
    dirs = np.concatenate([c[1] for c in cards])
    rights = np.repeat(np.array([c[2] for c in cards]), sizes, 0)
    face = norm(np.cross(rights, dirs))
    face *= np.sign(np.sum(face * outward, -1, keepdims=True) + 1e-9)
    nrm = norm(0.4 * face + 0.6 * outward)
    # The tangent is the card's true across direction, not squared up to the
    # bent normal: the game narrows a card along it (see below).
    hand = np.sign(np.sum(np.cross(nrm, rights) * dirs, -1) + 1e-9)
    leaf_out["normals"] = nrm.astype(np.float32)
    leaf_out["tangents"] = np.concatenate([rights, hand[:, None]], 1).astype(np.float32)

    cell = max(habit.height / 36.0, 0.12)
    own = (centres, areas)
    if shaded_by is not None:
        # Shaded by the near skin's foliage. Its cards are larger than the
        # near ones they stand for, so sample nearer each card's middle,
        # where those would be.
        centres, areas = shaded_by
        where = pos + (own[0][owner] - pos) * 0.45 + nrm * cell * 0.5
    else:
        where = pos + nrm * cell * 0.5
    open_leaf = _occlusion(centres, areas, where, cell)
    shade = np.repeat(np.array([c[3] for c in cards]), sizes)
    light = (0.30 + 0.70 * open_leaf ** 0.8) * shade
    # Foliage vertex colour is not a colour. Red is the shade. Green and blue
    # are the x and z of the card's own face normal (turned to point up), and
    # alpha is how far the vertex sits from the card's midline, signed along
    # the tangent: with these the game narrows a card to nothing as it turns
    # edge-on to the camera, where it would otherwise show as a smear.
    flat = norm(np.cross(rights, dirs))
    flat *= np.where(flat[:, 1:2] < 0, -1.0, 1.0)
    widths = np.repeat(np.array([c[4] for c in cards]), sizes)
    side = np.tile(np.array([-0.5, 0.5]), len(pos) // 2)
    leaf_out["colors"] = np.stack([light, flat[:, 0], flat[:, 2], side * widths], 1).astype(np.float32)
    open_bark = _occlusion(centres, areas, bark_out["positions"] + bark_out["normals"] * cell * 0.5, cell)
    bark_out["colors"][:, :3] = (0.45 + 0.55 * open_bark)[:, None]
    return bark_out, leaf_out, own


# --- species -------------------------------------------------------------------------

def _spruce():
    # A leader with rings of branches: the one shape `levels` still grows.
    return Habit(18.0, 0.30, [
        Level(162, (110, 55), 0.29, radius=0.11, start=0.09, end=0.985, lift=-0.25, tip_lift=0.55, wander=0.05,
              segs=2, whorl=6, spread=6),
    ], Foliage("bough", aspect=0.92), shape="cone", lean=0.006, flare=1.7, trunk_segs=16,
        roots=Roots(6, reach=8.0, radius=0.34, rise=1.3, fork=0.4), solid=(1500, 400))


def _pine():
    # A long bare trunk, and a crown that is widest near its top.
    return Habit(16.0, 0.27, [], Foliage("tuft", size=1.5, aspect=0.9, spacing=0.5, reach=1, mid=3.0),
                 trunk=0.78, lean=0.022, flare=1.4, top=0.10, roots=Roots(4, reach=5.5, radius=0.58, rise=1.7, fork=0.15),
                 solid=(3200, 700), hidden=1.05,
                 crown=Crown(radius=3.6, base=0.50, low=1.5, points=320, step=0.45, reach=3.0, clear=0.7, up=0.04))


def _oak():
    # A short trunk that divides into limbs, which divide again as they fill the crown.
    return Habit(13.0, 0.46, [], Foliage("spray", size=1.3, aspect=1.0, spacing=0.42, reach=1, mid=3.4),
                 trunk=0.30, shape="round", flare=1.7, top=0.33, taper=0.6, trunk_segs=7,
                 roots=Roots(6, reach=6.5, radius=0.44, rise=1.9, fork=0.8), solid=(5200, 1000),
                 crown=Crown(radius=6.0, points=680, clear=0.8, step=0.45), hidden=1.05)


def _birch():
    # A slim trunk well up into a narrow crown of fine, limber wood.
    return Habit(14.0, 0.17, [], Foliage("spray", size=0.95, aspect=0.9, spacing=0.5, hang=0.5, reach=1, mid=3.2),
                 trunk=0.70, flare=1.35, top=0.05, roots=Roots(4, reach=5.5, radius=0.58, rise=1.7, fork=0.15),
                 solid=(3000, 600), hidden=1.05,
                 crown=Crown(radius=2.1, base=0.34, low=0.85, points=460, step=0.36, reach=2.4, clear=0.48, up=0.3,
                             limber=0.07))


def _aspen():
    # Like the birch, but the crown starts higher and is narrower still.
    return Habit(15.0, 0.16, [], Foliage("spray", size=0.9, aspect=0.95, spacing=0.45, hang=0.12, reach=1, mid=3.2),
                 trunk=0.76, lean=0.012, flare=1.3, top=0.05,
                 roots=Roots(4, reach=5.5, radius=0.58, rise=1.7, fork=0.15), solid=(3000, 600), hidden=1.05,
                 crown=Crown(radius=2.0, base=0.42, low=0.9, points=380, step=0.36, reach=2.4, clear=0.5, up=0.25,
                             limber=0.07))


def _snag():
    return Habit(9.0, 0.26, [
        Level(7, (70, 40), 0.24, radius=0.45, start=0.35, end=0.95, lift=0.2, wander=0.25, segs=3),
        Level(2, (55, 40), 0.5, radius=0.55, start=0.4, lift=0.1, wander=0.3, segs=2),
    ], Foliage("none"), lean=0.03, flare=1.5, top=0.08, trunk_segs=8, twig=0.015, solid=(1800, 500))


def _bush():
    # A low dome on a stub of a stem. Too fine for the blended surface: tubes.
    return Habit(2.0, 0.05, [], Foliage("spray", size=0.62, aspect=1.0, spacing=0.26, hang=0.05, reach=1, mid=1.8),
                 trunk=0.12, flare=1.0, top=0.045, trunk_segs=2, twig=0.006, roots=None,
                 crown=Crown(radius=0.95, base=0.10, low=0.75, points=95, step=0.16, reach=0.8, clear=0.25, up=0.1,
                             limber=0.3))


@dataclass
class Species:
    name: str
    habit: object                   # () -> Habit
    seeds: list
    bark: str                       # a bark kind in `tree_textures`
    foliage: str | None             # a foliage kind in `tree_textures`, or None
    # Leaf colours the species can wear, sRGB 0..255; the first is its default.
    leaves: list = field(default_factory=list)


# Order is the species index in a world's `tree` lines; add new ones at the end.
SPECIES = [
    Species("spruce", _spruce, [1, 2, 3], "spruce", "spruce", [("green", (36, 74, 40))]),
    Species("pine", _pine, [1, 2, 3], "pine", "pine", [("green", (56, 96, 46))]),
    Species("oak", _oak, [1, 2, 3], "oak", "oak",
            [("green", (62, 104, 36)), ("amber", (196, 142, 40)), ("russet", (150, 78, 30))]),
    Species("birch", _birch, [1, 2, 3], "birch", "birch", [("green", (104, 146, 52)), ("gold", (216, 180, 44))]),
    Species("snag", _snag, [1, 2], "dead", None),
    Species("bush", _bush, [1, 2], "oak", "bush", [("green", (72, 114, 42)), ("red", (150, 46, 38))]),
    Species("aspen", _aspen, [1, 2, 3], "aspen", "aspen",
            [("green", (100, 144, 58)), ("gold", (230, 188, 36)), ("orange", (226, 128, 30)), ("red", (186, 52, 30))]),
]


TRIES = 10


def lopsided(sk: Skeleton) -> float:
    """How unbalanced a skeleton's crown is, 0 for even: how far its middle
    sits off the trunk, and how unevenly it fills the directions round it."""
    ends = np.array([b[0] for b in sk.buds] if sk.buds else [b.pts[-1] for b in sk.branches if b.level > 0])
    if len(ends) < 8:
        return 0.0
    flat = ends[:, [0, 2]]
    middle = flat.mean(0)
    radius = max(float(np.linalg.norm(flat - middle, axis=1).mean()), 1e-3)
    around = np.arctan2(flat[:, 1], flat[:, 0])
    filled = np.bincount(((around + math.pi) / (2 * math.pi) * 8).astype(int) % 8, minlength=8)
    return float(np.linalg.norm(middle)) / radius * 2.0 + float(filled.std() / filled.mean())


def build(species: Species, seed: int):
    """One variant: `(habit, near bark, near foliage, mid bark, mid foliage)`."""
    habit = species.habit()
    # Chance alone can put most of a crown on one side. Grow a few and keep
    # the best balanced.
    sk = min((grow(habit, np.random.default_rng([seed, len(species.name), k])) for k in range(TRIES)),
             key=lopsided)
    near_bark, near_leaves, crown = skin(sk, 1.0, seed)
    mid_bark, mid_leaves, _ = skin(sk, 0.45, seed, crown)
    return habit, near_bark, near_leaves, mid_bark, mid_leaves

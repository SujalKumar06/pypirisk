"""Score a whole project from its requirements and the dependency graph.

The project score is the *weakest link*: the highest dependency risk after discounting
by depth, ``max(risk * depth_decay ** (depth - 1))``. Unlike
``1 - prod(1 - risk)``, it does not approach 1 just because a project has many
dependencies, so large projects are not all graded F. The scan also reports the expected
number of dependencies abandoned within 18 months (the sum of calibrated
probabilities) and the paths that contribute most.
"""

from collections import deque
from dataclasses import dataclass, field

import pandas as pd
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

MAX_REQUIREMENT_LINES = 500


@dataclass
class ParsedRequirements:
    """Result of parsing a requirements file.

    Attributes:
        packages: Canonical names of the declared packages, in file order.
        errors: Lines that could not be parsed, with the reason.
    """

    packages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_requirements(text: str) -> ParsedRequirements:
    """Parse a ``requirements.txt`` (or a plain list of names).

    Comments, blank lines and pip options (``-r``, ``-e``, ``--index-url``) are skipped;
    they do not name a PyPI package.

    Args:
        text: File contents.

    Returns:
        The declared packages and the unparseable lines.

    Raises:
        ValueError: If the file has more than :data:`MAX_REQUIREMENT_LINES` lines.
    """
    lines = text.splitlines()
    if len(lines) > MAX_REQUIREMENT_LINES:
        raise ValueError(f"At most {MAX_REQUIREMENT_LINES} lines are accepted, got {len(lines)}")
    parsed = ParsedRequirements()
    for number, raw in enumerate(lines, start=1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            name = canonicalize_name(Requirement(line).name)
        except InvalidRequirement as error:
            parsed.errors.append(f"line {number}: {error}")
            continue
        if name not in parsed.packages:
            parsed.packages.append(name)
    return parsed


@dataclass(frozen=True)
class Dependency:
    """A package reached from the project's requirements.

    Attributes:
        name: Package name.
        depth: 1 for declared requirements, 2 for their dependencies, and so on.
        path: Packages from a declared requirement down to this one.
    """

    name: str
    depth: int
    path: tuple[str, ...]


def resolve(direct: list[str], edges: pd.DataFrame, max_depth: int) -> list[Dependency]:
    """Find every package reachable from the declared requirements.

    Breadth-first search, so each package is reported at its shallowest depth.

    Args:
        direct: Declared requirements.
        edges: ``source -> target`` edges of the scoring snapshot's graph.
        max_depth: Deepest level to follow.

    Returns:
        One entry per reachable package.
    """
    children = edges.groupby("source")["target"].apply(sorted).to_dict()
    found: dict[str, Dependency] = {}
    queue = deque(Dependency(name, 1, (name,)) for name in direct)
    while queue:
        dependency = queue.popleft()
        if dependency.name in found:
            continue
        found[dependency.name] = dependency
        if dependency.depth < max_depth:
            for child in children.get(dependency.name, []):
                if child not in found:
                    queue.append(Dependency(child, dependency.depth + 1, (*dependency.path, child)))
    return list(found.values())


@dataclass
class ProjectRisk:
    """Outcome of a project scan.

    Attributes:
        score: Weakest-link score between 0 and 1.
        expected_abandoned: Expected number of dependencies abandoned within the window.
        contributions: Scored dependencies with ``risk``, ``weight`` and ``weighted``
            risk, highest weighted first.
        unscored: Reachable packages without a risk score.
    """

    score: float
    expected_abandoned: float
    contributions: list[dict]
    unscored: list[str]


def project_risk(
    dependencies: list[Dependency], risks: dict[str, float], depth_decay: float
) -> ProjectRisk:
    """Combine package risks into a project score.

    Args:
        dependencies: Output of :func:`resolve`.
        risks: Risk per scored package.
        depth_decay: Weight multiplier per level below the declared requirements.

    Returns:
        The project's score, expected abandoned count, contributions and gaps.
    """
    contributions = []
    unscored = []
    for dependency in dependencies:
        risk = risks.get(dependency.name)
        if risk is None:
            unscored.append(dependency.name)
            continue
        weight = depth_decay ** (dependency.depth - 1)
        contributions.append(
            {
                "package": dependency.name,
                "depth": dependency.depth,
                "path": list(dependency.path),
                "risk": float(risk),
                "weight": weight,
                "weighted": float(risk) * weight,
            }
        )
    contributions.sort(key=lambda item: item["weighted"], reverse=True)
    return ProjectRisk(
        score=contributions[0]["weighted"] if contributions else 0.0,
        expected_abandoned=sum(item["risk"] for item in contributions),
        contributions=contributions,
        unscored=unscored,
    )

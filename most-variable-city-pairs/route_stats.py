from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Tuple

from shapely import frechet_distance, hausdorff_distance
from shapely.geometry import LineString


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, value: float) -> None:
        if self.count == 0:
            self.min_value = value
            self.max_value = value
        else:
            self.min_value = value if value < self.min_value else self.min_value
            self.max_value = value if value > self.max_value else self.max_value
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return sqrt(self.m2 / self.count)

    def min(self) -> float:
        return 0.0 if self.min_value is None else self.min_value

    def max(self) -> float:
        return 0.0 if self.max_value is None else self.max_value


def summarize_reference_distances(
    distances: Iterable[Tuple[float, float]],
) -> dict[str, float]:
    hausdorff_stats = RunningStats()
    frechet_stats = RunningStats()

    for hausdorff_value, frechet_value in distances:
        hausdorff_stats.update(hausdorff_value)
        frechet_stats.update(frechet_value)

    pairwise_count = hausdorff_stats.count
    hausdorff_variance = (
        0.0 if hausdorff_stats.count == 0 else hausdorff_stats.m2 / hausdorff_stats.count
    )
    frechet_variance = (
        0.0 if frechet_stats.count == 0 else frechet_stats.m2 / frechet_stats.count
    )
    return {
        "pairwise_count": pairwise_count,
        "hausdorff_avg_distance": hausdorff_stats.mean,
        "hausdorff_std_distance": hausdorff_stats.std(),
        "hausdorff_var_distance": hausdorff_variance,
        "hausdorff_min_distance": hausdorff_stats.min(),
        "hausdorff_max_distance": hausdorff_stats.max(),
        "frechet_avg_distance": frechet_stats.mean,
        "frechet_std_distance": frechet_stats.std(),
        "frechet_var_distance": frechet_variance,
        "frechet_min_distance": frechet_stats.min(),
        "frechet_max_distance": frechet_stats.max(),
    }


def compute_pairwise_stats(
    routes: Iterable[List[Tuple[float, float]]],
) -> dict[str, float]:
    line_strings = [LineString(route) for route in routes]
    distances = []

    for i in range(len(line_strings)):
        line_i = line_strings[i]
        for j in range(i + 1, len(line_strings)):
            line_j = line_strings[j]
            distances.append(
                (
                    hausdorff_distance(line_i, line_j),
                    frechet_distance(line_i, line_j),
                )
            )

    return summarize_reference_distances(distances)


def compute_reference_distances(
    reference_route: List[Tuple[float, float]],
    other_routes: Iterable[List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    reference_line = LineString(reference_route)
    distances: List[Tuple[float, float]] = []

    for route in other_routes:
        line = LineString(route)
        distances.append(
            (
                hausdorff_distance(reference_line, line),
                frechet_distance(reference_line, line),
            )
        )

    return distances


def compute_reference_stats(
    reference_route: List[Tuple[float, float]],
    other_routes: Iterable[List[Tuple[float, float]]],
) -> dict[str, float]:
    distances = compute_reference_distances(reference_route, other_routes)
    return summarize_reference_distances(distances)

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

print(str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import random
from dataclasses import dataclass
from math import cos, radians
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from route_io import RouteLoadStats, load_routes_by_pair, load_waypoint_coords
from route_stats import compute_reference_distances, summarize_reference_distances
from utils.haversine import haversine_distance

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm isn't installed
    tqdm = None


MIN_ROUTES_DEFAULT = 10
MAX_ROUTES_PER_PAIR = 35
ROUTE_COLUMN_DEFAULT = "real_waypoints"
FALLBACK_COLUMNS_DEFAULT = ("real_full_waypoints",)
FRECHET_OUTLIER_STD_DEFAULT = 2.0
NM_PER_DEGREE = 60.0
DISTANCE_RELEGATION_THRESHOLD_DEFAULT = 1.35  # Times the reference route length
MAX_REFERENCE_ATTEMPTS = 10
REFERENCE_GCD_MULTIPLIER = 1.3

WAYPOINT_COORDS: Dict[str, Tuple[float, float]] | None = None
ROUTE_COLUMN: str | None = None
FALLBACK_COLUMNS: Tuple[str, ...] = FALLBACK_COLUMNS_DEFAULT
MIN_ROUTES: int = MIN_ROUTES_DEFAULT
FRECHET_OUTLIER_STD: float = FRECHET_OUTLIER_STD_DEFAULT
OUTLIER_FILTER_ENABLED: bool = True
DISTANCE_RELEGATION_THRESHOLD: float = DISTANCE_RELEGATION_THRESHOLD_DEFAULT


@dataclass
class FileResult:
    results: dict[tuple[str, str], dict[str, float]]
    skipped_pairs: int
    skipped_pairs_after_filter: int
    skipped_pairs_reference_too_long: int
    pair_count: int
    load_stats: RouteLoadStats
    resolved_column: str | None
    outliers_relegated: int


def init_worker(
    graph_path: str,
    route_column: str,
    fallback_columns: Iterable[str],
    min_routes: int,
    frechet_outlier_std: float,
    outlier_filter_enabled: bool,
    distance_relegation_threshold: float,
) -> None:
    global WAYPOINT_COORDS
    global ROUTE_COLUMN
    global FALLBACK_COLUMNS
    global MIN_ROUTES
    global FRECHET_OUTLIER_STD
    global OUTLIER_FILTER_ENABLED
    global DISTANCE_RELEGATION_THRESHOLD
    WAYPOINT_COORDS = load_waypoint_coords(Path(graph_path))
    ROUTE_COLUMN = route_column
    FALLBACK_COLUMNS = tuple(fallback_columns)
    MIN_ROUTES = min_routes
    FRECHET_OUTLIER_STD = frechet_outlier_std
    OUTLIER_FILTER_ENABLED = outlier_filter_enabled
    DISTANCE_RELEGATION_THRESHOLD = distance_relegation_threshold
    random.seed()


def compute_route_length(route: List[Tuple[float, float]]) -> float:
    if len(route) < 2:
        return 0.0
    total = 0.0
    prev_lon, prev_lat = route[0]
    for lon, lat in route[1:]:
        total += haversine_distance(prev_lat, prev_lon, lat, lon)
        prev_lon, prev_lat = lon, lat
    return total


def compute_great_circle_nm(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
) -> float:
    origin_lon, origin_lat = origin
    dest_lon, dest_lat = destination
    return haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)


def mean_latitude(route: List[Tuple[float, float]]) -> float:
    if not route:
        return 0.0
    return sum(lat for _, lat in route) / len(route)


def project_route_nm(
    route: List[Tuple[float, float]],
    lat_ref: float,
) -> List[Tuple[float, float]]:
    scale_lon = NM_PER_DEGREE * cos(radians(lat_ref))
    scale_lat = NM_PER_DEGREE
    return [(lon * scale_lon, lat * scale_lat) for lon, lat in route]


def process_csv(csv_path: str) -> FileResult:
    if WAYPOINT_COORDS is None or ROUTE_COLUMN is None:
        raise RuntimeError("Worker not initialized with waypoint coordinates.")

    path = Path(csv_path)
    routes_by_pair, load_stats, resolved_column = load_routes_by_pair(
        path,
        WAYPOINT_COORDS,
        ROUTE_COLUMN,
        FALLBACK_COLUMNS,
    )

    results: dict[tuple[str, str], dict[str, float]] = {}
    skipped_pairs = 0
    skipped_pairs_after_filter = 0
    skipped_pairs_reference_too_long = 0
    outliers_relegated = 0
    for pair, routes in routes_by_pair.items():
        if len(routes) > MAX_ROUTES_PER_PAIR:
            routes = random.sample(routes, k=MAX_ROUTES_PER_PAIR)
        if len(routes) < MIN_ROUTES:
            skipped_pairs += 1
            continue
        origin_coord = WAYPOINT_COORDS.get(pair[0])
        destination_coord = WAYPOINT_COORDS.get(pair[1])
        if origin_coord is None or destination_coord is None:
            skipped_pairs += 1
            continue
        gcd_nm = compute_great_circle_nm(origin_coord, destination_coord)
        reference_route = None
        reference_index = -1
        for _ in range(MAX_REFERENCE_ATTEMPTS):
            candidate_index = random.randrange(len(routes))
            candidate_route = routes[candidate_index]
            if compute_route_length(candidate_route) <= gcd_nm * REFERENCE_GCD_MULTIPLIER:
                reference_route = candidate_route
                reference_index = candidate_index
                break
        if reference_route is None:
            skipped_pairs_reference_too_long += 1
            continue
        comparison_routes = routes[:reference_index] + routes[reference_index + 1 :]
        lat_ref = mean_latitude(reference_route)
        reference_projected = project_route_nm(reference_route, lat_ref)
        comparison_projected = [
            project_route_nm(route, lat_ref) for route in comparison_routes
        ]
        distances = compute_reference_distances(
            reference_projected,
            comparison_projected,
        )
        stats = summarize_reference_distances(distances)
        filtered_routes = comparison_routes
        filtered_distances = distances
        if OUTLIER_FILTER_ENABLED and distances:
            mean = stats["frechet_avg_distance"]
            std = stats["frechet_std_distance"]
            threshold = FRECHET_OUTLIER_STD * std
            reference_length = compute_route_length(reference_route)
            filtered_routes = []
            filtered_distances = []
            for route, (hausdorff_value, frechet_value) in zip(
                comparison_routes, distances
            ):
                route_length = compute_route_length(route)
                if route_length > reference_length * DISTANCE_RELEGATION_THRESHOLD:
                    outliers_relegated += 1
                    continue
                if abs(frechet_value - mean) <= threshold:
                    filtered_routes.append(route)
                    filtered_distances.append((hausdorff_value, frechet_value))
                else:
                    outliers_relegated += 1
            if len(filtered_routes) + 1 < MIN_ROUTES:
                skipped_pairs_after_filter += 1
                continue
            stats = summarize_reference_distances(filtered_distances)
        stats["route_count"] = len(filtered_routes) + 1
        stats["origin"] = pair[0]
        stats["destination"] = pair[1]
        results[pair] = stats

    return FileResult(
        results=results,
        skipped_pairs=skipped_pairs,
        skipped_pairs_after_filter=skipped_pairs_after_filter,
        skipped_pairs_reference_too_long=skipped_pairs_reference_too_long,
        pair_count=len(routes_by_pair),
        load_stats=load_stats,
        resolved_column=resolved_column,
        outliers_relegated=outliers_relegated,
    )


def iter_csv_files(input_dir: Path) -> List[Path]:
    # return [Path("matched_filtered_data_by_origin_region/LF.csv")] # TODO: Remove this
    return sorted(input_dir.glob("*.csv"))


def write_outputs(
    output_dir: Path,
    rows: List[dict[str, float]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "route_hausdorff_var.csv"
    json_path = output_dir / "route_hausdorff_var_summary.json"

    headers = [
        "origin",
        "destination",
        "route_count",
        "pairwise_count",
        "hausdorff_avg_distance",
        "hausdorff_std_distance",
        "hausdorff_var_distance",
        "hausdorff_min_distance",
        "hausdorff_max_distance",
        "frechet_avg_distance",
        "frechet_std_distance",
        "frechet_var_distance",
        "frechet_min_distance",
        "frechet_max_distance",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise Hausdorff/Frechet distance stats for city-pair routes."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("matched_filtered_data_by_origin_region"),
        help="Directory of matched route CSV files.",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("data/graphs/ats_fra_nodes_only.gml"),
        help="GML graph containing waypoint lat/lon attributes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/most-variable-city-pairs"),
        help="Directory for output CSV/summary files.",
    )
    parser.add_argument(
        "--route-column",
        default=ROUTE_COLUMN_DEFAULT,
        help="CSV column name containing waypoint sequences.",
    )
    parser.add_argument(
        "--min-routes",
        type=int,
        default=MIN_ROUTES_DEFAULT,
        help="Minimum number of routes required to compute stats for a city pair.",
    )
    parser.add_argument(
        "--frechet-outlier-std",
        type=float,
        default=FRECHET_OUTLIER_STD_DEFAULT,
        help="Standard deviation multiplier for Frechet outlier filtering.",
    )
    parser.add_argument(
        "--distance-relegation-threshold",
        type=float,
        default=DISTANCE_RELEGATION_THRESHOLD_DEFAULT,
        help=(
            "Max allowed route length multiplier (e.g. 1.5 = 150 percent of reference) before relegation."
        ),
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="Disable Frechet outlier filtering.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Number of worker processes to use.",
    )
    args = parser.parse_args()

    csv_files = iter_csv_files(args.input_dir)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {args.input_dir}")

    results: Dict[tuple[str, str], dict[str, float]] = {}
    total_skipped_pairs = 0
    total_skipped_pairs_after_filter = 0
    total_skipped_pairs_reference_too_long = 0
    total_pairs_seen = 0
    total_missing_tokens = 0
    total_skipped_routes = 0
    total_outliers_relegated = 0
    resolved_columns: List[str] = []
    duplicate_pairs = 0

    progress = tqdm if tqdm is not None else None

    with Pool(
        processes=args.processes,
        initializer=init_worker,
        initargs=(
            str(args.graph_path),
            args.route_column,
            FALLBACK_COLUMNS_DEFAULT,
            args.min_routes,
            args.frechet_outlier_std,
            not args.no_outlier_filter,
            args.distance_relegation_threshold,
        ),
    ) as pool:
        iterator = pool.imap_unordered(process_csv, [str(path) for path in csv_files])
        if progress is not None:
            iterator = progress(iterator, total=len(csv_files), desc="Processing CSVs")
        for file_result in iterator:
            total_skipped_pairs += file_result.skipped_pairs
            total_skipped_pairs_after_filter += file_result.skipped_pairs_after_filter
            total_skipped_pairs_reference_too_long += (
                file_result.skipped_pairs_reference_too_long
            )
            total_pairs_seen += file_result.pair_count
            total_missing_tokens += file_result.load_stats.missing_waypoint_tokens
            total_skipped_routes += (
                file_result.load_stats.skipped_missing_endpoints
                + file_result.load_stats.skipped_too_short
            )
            total_outliers_relegated += file_result.outliers_relegated
            if file_result.resolved_column:
                resolved_columns.append(file_result.resolved_column)
            for pair, stats in file_result.results.items():
                if pair in results:
                    duplicate_pairs += 1
                    continue
                results[pair] = stats

    rows = list(results.values())
    rows.sort(key=lambda row: row.get("hausdorff_std_distance", 0.0), reverse=True)

    summary = {
        "files_processed": len(csv_files),
        "total_pairs_seen": total_pairs_seen,
        "pairs_reported": len(rows),
        "pairs_skipped_insufficient_routes": total_skipped_pairs,
        "pairs_skipped_after_outlier_filter": total_skipped_pairs_after_filter,
        "pairs_skipped_reference_too_long": total_skipped_pairs_reference_too_long,
        "routes_skipped": total_skipped_routes,
        "missing_waypoint_tokens": total_missing_tokens,
        "duplicate_pairs": duplicate_pairs,
        "resolved_route_columns": sorted(set(resolved_columns)),
        "min_routes": args.min_routes,
        "max_routes_per_pair": MAX_ROUTES_PER_PAIR,
        "outlier_filter_enabled": not args.no_outlier_filter,
        "frechet_outlier_std": args.frechet_outlier_std,
        "distance_relegation_threshold": args.distance_relegation_threshold,
        "outliers_relegated": total_outliers_relegated,
        "route_column_requested": args.route_column,
        "distance_unit": "nautical_miles",
        "distance_projection": "equirectangular_mean_lat",
        "distance_definition": (
            "distance=Hausdorff (reference), frechet_distance=Frechet (reference, shapely), units=nautical_miles"
        ),
    }

    write_outputs(args.output_dir, rows, summary)


if __name__ == "__main__":
    main()

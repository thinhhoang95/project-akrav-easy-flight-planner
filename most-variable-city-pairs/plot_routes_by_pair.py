from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

print(str(Path(__file__).resolve().parent.parent))


import argparse
import random
from pathlib import Path
from math import cos, radians

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from route_io import load_routes_for_pair, load_waypoint_coords
from route_stats import compute_reference_distances, summarize_reference_distances
from utils.haversine import haversine_distance





ROUTE_COLUMN_DEFAULT = "real_waypoints"
FALLBACK_COLUMNS_DEFAULT = ("real_full_waypoints",)
MAX_ROUTES_DEFAULT = 40
FRECHET_OUTLIER_STD_DEFAULT = 2.0
DISTANCE_RELEGATION_THRESHOLD_DEFAULT = 1.35 # Times the reference route length
NM_PER_DEGREE = 60.0


def iter_csv_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.csv"))


def collect_routes(
    csv_files: list[Path],
    waypoint_coords: dict[str, tuple[float, float]],
    origin: str,
    destination: str,
    route_column: str,
    fallback_columns: tuple[str, ...],
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    all_routes: list[list[tuple[float, float]]] = []
    resolved_columns: list[str] = []
    for csv_path in csv_files:
        routes, _, resolved = load_routes_for_pair(
            csv_path,
            waypoint_coords,
            route_column,
            fallback_columns,
            origin,
            destination,
        )
        if resolved:
            resolved_columns.append(resolved)
        if routes:
            all_routes.extend(routes)
    return all_routes, resolved_columns


def plot_routes(
    routes: list[list[tuple[float, float]]],
    relegated_routes: list[list[tuple[float, float]]],
    origin: str,
    destination: str,
    total_routes: int,
    relegated_count: int,
) -> None:
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    title_suffix = (
        f" (relegated {relegated_count})" if relegated_count > 0 else ""
    )
    ax.set_title(
        f"Routes from {origin} to {destination} "
        f"(showing {len(routes)} of {total_routes}{title_suffix})"
    )
    ax.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="none")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.coastlines(linewidth=0.5)

    colors = plt.cm.viridis([i / max(len(routes) - 1, 1) for i in range(len(routes))])
    for route, color in zip(routes, colors):
        lons = [coord[0] for coord in route]
        lats = [coord[1] for coord in route]
        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=1.2,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )

    if relegated_routes:
        for route in relegated_routes:
            lons = [coord[0] for coord in route]
            lats = [coord[1] for coord in route]
            ax.plot(
                lons,
                lats,
                color="gray",
                linewidth=1.0,
                alpha=0.5,
                linestyle="--",
                transform=ccrs.PlateCarree(),
            )

    if routes:
        start = routes[0][0]
        end = routes[0][-1]
        ax.scatter(
            [start[0]],
            [start[1]],
            color="black",
            marker="o",
            s=40,
            label="Origin",
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            [end[0]],
            [end[1]],
            color="black",
            marker="x",
            s=50,
            label="Destination",
            transform=ccrs.PlateCarree(),
        )
        ax.legend(loc="best")

    if routes or relegated_routes:
        all_routes = routes + relegated_routes
        all_lons = [lon for route in all_routes for lon, _ in route]
        all_lats = [lat for route in all_routes for _, lat in route]
        if all_lons and all_lats:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
            lon_span = max(max_lon - min_lon, 1.0)
            lat_span = max(max_lat - min_lat, 1.0)
            padding = max(lon_span, lat_span) * 0.1
            ax.set_extent(
                [
                    max(-180.0, min_lon - padding),
                    min(180.0, max_lon + padding),
                    max(-90.0, min_lat - padding),
                    min(90.0, max_lat + padding),
                ],
                crs=ccrs.PlateCarree(),
            )

    gridlines = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gridlines.top_labels = False
    gridlines.right_labels = False
    plt.tight_layout()
    plt.show()


def filter_outlier_routes(
    routes: list[list[tuple[float, float]]],
    frechet_outlier_std: float,
    distance_relegation_threshold: float,
    enabled: bool,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]], int]:
    if not enabled or len(routes) < 2:
        return routes, [], 0

    def compute_route_length(route: list[tuple[float, float]]) -> float:
        if len(route) < 2:
            return 0.0
        total = 0.0
        prev_lon, prev_lat = route[0]
        for lon, lat in route[1:]:
            total += haversine_distance(prev_lat, prev_lon, lat, lon)
            prev_lon, prev_lat = lon, lat
        return total

    def mean_latitude(route: list[tuple[float, float]]) -> float:
        if not route:
            return 0.0
        return sum(lat for _, lat in route) / len(route)

    def project_route_nm(
        route: list[tuple[float, float]],
        lat_ref: float,
    ) -> list[tuple[float, float]]:
        scale_lon = NM_PER_DEGREE * cos(radians(lat_ref))
        scale_lat = NM_PER_DEGREE
        return [(lon * scale_lon, lat * scale_lat) for lon, lat in route]

    reference_index = random.randrange(len(routes))
    reference_route = routes[reference_index]
    reference_length = compute_route_length(reference_route)
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
    if not distances:
        return routes, [], 0

    stats = summarize_reference_distances(distances)
    mean = stats["frechet_avg_distance"]
    std = stats["frechet_std_distance"]
    threshold = frechet_outlier_std * std
    filtered_routes = [reference_route]
    relegated_routes: list[list[tuple[float, float]]] = []
    relegated_routes_count = 0

    for route, (_, frechet_value) in zip(comparison_routes, distances):
        route_length = compute_route_length(route)
        if route_length > reference_length * distance_relegation_threshold:
            relegated_routes_count += 1
            relegated_routes.append(route)
            continue
        if abs(frechet_value - mean) <= threshold:
            filtered_routes.append(route)
        else:
            relegated_routes_count += 1
            relegated_routes.append(route)

    return filtered_routes, relegated_routes, relegated_routes_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot randomly selected routes for a city pair."
    )
    parser.add_argument("--origin", required=True, help="Origin waypoint identifier.")
    parser.add_argument(
        "--destination", required=True, help="Destination waypoint identifier."
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
        "--route-column",
        default=ROUTE_COLUMN_DEFAULT,
        help="CSV column name containing waypoint sequences.",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=MAX_ROUTES_DEFAULT,
        help="Maximum number of routes to plot.",
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
        "--show-relegated-routes",
        action="store_true",
        help="Plot relegated routes using dotted lines.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    csv_files = iter_csv_files(args.input_dir)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {args.input_dir}")

    waypoint_coords = load_waypoint_coords(args.graph_path)
    routes, resolved_columns = collect_routes(
        csv_files,
        waypoint_coords,
        args.origin,
        args.destination,
        args.route_column,
        FALLBACK_COLUMNS_DEFAULT,
    )
    if not routes:
        resolved_info = (
            f"Resolved columns: {sorted(set(resolved_columns))}"
            if resolved_columns
            else "No route column found."
        )
        raise SystemExit(
            f"No routes found for {args.origin}->{args.destination}. {resolved_info}"
        )

    if args.seed is not None:
        random.seed(args.seed)
    original_total_routes = len(routes)
    routes, relegated_routes, relegated_routes_count = filter_outlier_routes(
        routes,
        args.frechet_outlier_std,
        args.distance_relegation_threshold,
        not args.no_outlier_filter,
    )
    random.shuffle(routes)
    total_routes = len(routes)
    if len(routes) > args.max_routes:
        routes = random.sample(routes, k=args.max_routes)

    plot_routes(
        routes,
        relegated_routes if args.show_relegated_routes else [],
        args.origin,
        args.destination,
        original_total_routes,
        relegated_routes_count,
    )


if __name__ == "__main__":
    main()

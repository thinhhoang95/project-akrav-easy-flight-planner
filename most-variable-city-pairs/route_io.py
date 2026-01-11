from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx


@dataclass
class RouteLoadStats:
    skipped_too_short: int = 0
    skipped_missing_endpoints: int = 0
    missing_waypoint_tokens: int = 0


def load_waypoint_coords(graph_path: Path) -> Dict[str, Tuple[float, float]]:
    graph = nx.read_gml(graph_path)
    coords: Dict[str, Tuple[float, float]] = {}
    for node, data in graph.nodes(data=True):
        # In the GML graph the node identifier is the waypoint name.
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is None or lon is None:
            continue
        coords[str(node)] = (float(lon), float(lat))
    return coords


def resolve_route_column(
    headers: Iterable[str] | None,
    preferred: str,
    fallback: Iterable[str],
) -> str | None:
    if not headers:
        return None
    header_set = set(headers)
    if preferred in header_set:
        return preferred
    for candidate in fallback:
        if candidate in header_set:
            return candidate
    return None


def load_routes_by_pair(
    csv_path: Path,
    waypoint_coords: Dict[str, Tuple[float, float]],
    route_column: str,
    fallback_columns: Iterable[str],
) -> tuple[dict[tuple[str, str], list[list[tuple[float, float]]]], RouteLoadStats, str | None]:
    routes_by_pair: dict[tuple[str, str], list[list[tuple[float, float]]]] = defaultdict(list)
    stats = RouteLoadStats()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        resolved_column = resolve_route_column(reader.fieldnames, route_column, fallback_columns)
        if resolved_column is None:
            return routes_by_pair, stats, None
        for row in reader:
            raw = row.get(resolved_column, "")
            if not raw:
                stats.skipped_too_short += 1
                continue
            tokens = raw.strip().split()
            if len(tokens) < 2:
                stats.skipped_too_short += 1
                continue
            origin = tokens[0]
            destination = tokens[-1]
            if origin not in waypoint_coords or destination not in waypoint_coords:
                stats.skipped_missing_endpoints += 1
                continue

            coords: List[Tuple[float, float]] = []
            missing_tokens = 0
            for token in tokens:
                coord = waypoint_coords.get(token)
                if coord is None:
                    missing_tokens += 1
                    continue
                coords.append(coord)
            if len(coords) < 2:
                stats.skipped_too_short += 1
                continue
            stats.missing_waypoint_tokens += missing_tokens
            routes_by_pair[(origin, destination)].append(coords)

    return routes_by_pair, stats, resolved_column


def load_routes_for_pair(
    csv_path: Path,
    waypoint_coords: Dict[str, Tuple[float, float]],
    route_column: str,
    fallback_columns: Iterable[str],
    origin: str,
    destination: str,
) -> tuple[list[list[tuple[float, float]]], RouteLoadStats, str | None]:
    routes: list[list[tuple[float, float]]] = []
    stats = RouteLoadStats()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        resolved_column = resolve_route_column(reader.fieldnames, route_column, fallback_columns)
        if resolved_column is None:
            return routes, stats, None
        for row in reader:
            raw = row.get(resolved_column, "")
            if not raw:
                stats.skipped_too_short += 1
                continue
            tokens = raw.strip().split()
            if len(tokens) < 2:
                stats.skipped_too_short += 1
                continue
            row_origin = tokens[0]
            row_destination = tokens[-1]
            if row_origin != origin or row_destination != destination:
                continue
            if row_origin not in waypoint_coords or row_destination not in waypoint_coords:
                stats.skipped_missing_endpoints += 1
                continue

            coords: List[Tuple[float, float]] = []
            missing_tokens = 0
            for token in tokens:
                coord = waypoint_coords.get(token)
                if coord is None:
                    missing_tokens += 1
                    continue
                coords.append(coord)
            if len(coords) < 2:
                stats.skipped_too_short += 1
                continue
            stats.missing_waypoint_tokens += missing_tokens
            routes.append(coords)

    return routes, stats, resolved_column

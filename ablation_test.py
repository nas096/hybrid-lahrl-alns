from functools import lru_cache
import csv
from alns_helpers import (
    FSMVRP_State,
    greedy_repair,
    _route_dist,
    _best_vehicle,
    _vehicle_swap,
    _two_opt,
    _or_opt
)
from problemdef import get_random_problems
import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import SegmentedRouletteWheel

from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed

try:
    from alns.stop import MaxRuntime
except Exception:
    MaxRuntime = None

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


_ = get_random_problems


@lru_cache(maxsize=100000)
def fast_best_vehicle(load, route_dist, capacities, fixed_costs, var_costs):
    best_v, best_c = -1, float('inf')
    for vt in range(len(capacities)):
        if capacities[vt] >= load:
            c = fixed_costs[vt] + route_dist * var_costs[vt]
            if c < best_c:
                best_c, best_v = c, vt
    return best_v, best_c


def _best_vehicle(load, route_dist, data):
    """Wrapper to map the data dict to the fast cached function."""

    return fast_best_vehicle(
        load,
        route_dist,
        tuple(data['capacity']),
        tuple(data['fixed_cost']),
        tuple(data['var_cost'])
    )


INSTANCES = [
    "003", "004", "005", "006", "013", "014",
    "015", "016", "017", "018", "019", "020"
]

METHODS = [
    "v1_bbox_divideL",
    "v2_maxcoord_divideL",
    "v3_cap_to_unit_max3",
    "v4_fixed100_cap_scale",
    "v5_cap_to_range_0p5_3"
]

BEST_KNOWN_MAP = {
    "003": 961.03, "004": 6437.33, "005": 1007.05, "006": 6516.47,
    "013": 2406.36, "014": 9119.03, "015": 2586.37, "016": 2720.43,
    "017": 1734.53, "018": 2369.65, "019": 8661.81, "020": 4029.61,
}
PAPER_REPORT_MAP = {
    "003": 979.22, "004": 6442.22, "005": 1008.59, "006": 6518.13,
    "013": 2496.12, "014": 9137.82, "015": 2644.28, "016": 2774.28,
    "017": 1786.66, "018": 2457.30, "019": 8760.59, "020": 4145.65,
}

TEST_TENSOR_SIZES = [20, 50, 75, 100, 200, 500, 1000]

TEST_TENSOR_BASELINES = {
    20: 64.13,
    50: 159.01,
    75: 214.08,
    100: 292.83,
    200: 614.13,
    500: 1601.60,
    1000: 3248.01,
}

ALNS_SEEDS = [
    1234,
    5678,
    9999,
    42,
    104729,
    77777,
    314159,
    271828,
    11235813,
    260706,
]


class TeeLogger:
    """Duplicates sys.stdout to both the terminal and a target log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def interactive_menu():
    """Provides a terminal UI for selecting datasets, methods, instances, and runs."""
    print("=" * 50)
    print(" Pure ALNS Evaluation Menu ".center(50))
    print("=" * 50)

    print("\nSelect Dataset:")
    print("  1. GoldenBC")
    print("  2. Test Tensors (data/test_tensor...)")
    try:
        ds_choice = int(input("\nEnter dataset choice (1-2): "))
        dataset = "Test Tensors" if ds_choice == 2 else "GoldenBC"
    except ValueError:
        print("Invalid selection. Defaulting to GoldenBC.")
        dataset = "GoldenBC"

    selected_methods = ["None"]
    selected_instances = []
    if dataset == "GoldenBC":
        print("\n" + "-" * 50)
        print("Select Normalization Method:")
        for i, m in enumerate(METHODS):
            print(f"  {i + 1}. {m}")
        print(f"  {len(METHODS) + 1}. ALL Methods")

        try:
            m_choice = int(input("\nEnter method choice (1-6): "))
            selected_methods = METHODS if m_choice == len(
                METHODS) + 1 else [METHODS[m_choice - 1]]
        except (ValueError, IndexError):
            print("Invalid selection. Defaulting to ALL methods.")
            selected_methods = METHODS

        print("\n" + "-" * 50)
        print("Select GoldenBC Instance(s):")
        for i, inst in enumerate(INSTANCES):
            print(f"  {i + 1}. Problem {inst}")
        print(f"  {len(INSTANCES) + 1}. ALL Instances")

        i_choice_str = input(
            f"\nEnter instance choice(s) separated by comma (e.g. 1, 3) or {len(INSTANCES) + 1} for ALL: ")
        try:
            choices = [int(x.strip())
                       for x in i_choice_str.split(',') if x.strip()]
            if len(INSTANCES) + 1 in choices or not choices:
                selected_instances = INSTANCES
            else:
                selected_instances = [INSTANCES[c - 1]
                                      for c in choices if 1 <= c <= len(INSTANCES)]
                selected_instances = list(dict.fromkeys(selected_instances))
                if not selected_instances:
                    selected_instances = INSTANCES
        except ValueError:
            selected_instances = INSTANCES

    else:

        print("\n" + "-" * 50)
        print("Select Test Tensor Size(s):")
        for i, size in enumerate(TEST_TENSOR_SIZES):
            print(f"  {i + 1}. {size} nodes")
        print(f"  {len(TEST_TENSOR_SIZES) + 1}. ALL Sizes")

        i_choice_str = input(
            f"\nEnter size choice(s) separated by comma (e.g. 1, 3) or {len(TEST_TENSOR_SIZES) + 1} for ALL: ")
        try:
            choices = [int(x.strip())
                       for x in i_choice_str.split(',') if x.strip()]
            if len(TEST_TENSOR_SIZES) + 1 in choices or not choices:
                selected_instances = TEST_TENSOR_SIZES
            else:
                selected_instances = [TEST_TENSOR_SIZES[c - 1]
                                      for c in choices if 1 <= c <= len(TEST_TENSOR_SIZES)]
                selected_instances = list(dict.fromkeys(selected_instances))
                if not selected_instances:
                    selected_instances = TEST_TENSOR_SIZES
        except ValueError:
            selected_instances = TEST_TENSOR_SIZES
        print("\n" + "-" * 50)
    print("\n" + "-" * 50)
    print("Select Number of Instances to Test per File:")
    try:
        num_to_test = int(
            input("\nEnter number of instances to run (e.g., 10, or press Enter for ALL): "))
        if num_to_test < 1:
            num_to_test = None
    except ValueError:
        print("Invalid selection. Defaulting to ALL instances.")
        num_to_test = None

    print("\n" + "-" * 50)
    print("Select Number of Runs (Fluctuation Gauge):")
    try:
        num_runs = int(input("\nEnter number of runs (default 1): "))
        if num_runs < 1:
            num_runs = 1
    except ValueError:
        print("Invalid selection. Defaulting to 1 run.")
        num_runs = 1

    return dataset, selected_methods, selected_instances, num_to_test, num_runs


def setup_logger(log_dir: str, log_level: str, timestamp: str) -> tuple[logging.Logger, Path]:
    resolved_dir = Path(log_dir).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / f"pure_alns_run_{timestamp}.log"

    logger = logging.getLogger("pure_alns_runner")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def _excepthook(exc_type, exc_value, exc_traceback):
        logger.exception(
            "Uncaught exception during pure ALNS run",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _excepthook
    return logger, log_path


def compute_L_bbox(raw_pt_path: str) -> float:
    """L = max(Δx, Δy) of the bbox (for v1, v3, v4, v6, v7)."""
    data = torch.load(raw_pt_path, map_location='cpu')
    depot_xy = data['depot_xy'].float()
    node_xy = data['node_xy'].float()
    if depot_xy.dim() == 2:
        depot_xy = depot_xy.unsqueeze(0)
    if node_xy.dim() == 2:
        node_xy = node_xy.unsqueeze(0)
    all_xy = torch.cat([depot_xy, node_xy], dim=1)
    span = all_xy.amax(dim=1) - all_xy.amin(dim=1)
    return float(span.amax(dim=1)[0].item())


def compute_L_maxcoord(raw_pt_path: str) -> float:
    """L = max(all_coords) for v2_maxcoord_divideL."""
    data = torch.load(raw_pt_path, map_location='cpu')
    depot_xy = data['depot_xy'].float()
    node_xy = data['node_xy'].float()
    if depot_xy.dim() == 2:
        depot_xy = depot_xy.unsqueeze(0)
    if node_xy.dim() == 2:
        node_xy = node_xy.unsqueeze(0)
    all_xy = torch.cat([depot_xy, node_xy], dim=1)
    return float(all_xy.amax().item())


def get_L_for_method(raw_pt_path: str, method: str) -> float:
    if method == "v2_maxcoord_divideL":
        return compute_L_maxcoord(raw_pt_path)
    if method == "v4_fixed100_cap_scale":
        return 100.0
    return compute_L_bbox(raw_pt_path)


class StagnationStop:
    """Stops ALNS after max iterations or too many iterations without a new best."""

    def __init__(self, max_iterations, patience, min_iterations=0, tolerance=1e-9):

        self.max_iterations = int(max_iterations)

        self.patience = int(patience)

        self.min_iterations = int(min_iterations)

        self.tolerance = float(tolerance)

        self.iterations = 0

        self.no_improve_iterations = 0

        self.best_objective = None

    def __call__(self, rng, best, current):
        """Return True when the ALNS run should stop."""

        self.iterations += 1

        best_objective = float(best.objective())

        if self.best_objective is None or best_objective < self.best_objective - self.tolerance:
            self.best_objective = best_objective
            self.no_improve_iterations = 0
        else:
            self.no_improve_iterations += 1

        if self.iterations >= self.max_iterations:
            return True

        return self.iterations >= self.min_iterations and self.no_improve_iterations >= self.patience


def parse_args():
    parser = argparse.ArgumentParser(description="Pure ALNS benchmark runner")
    parser.add_argument("--log_level", type=str, default="INFO")

    parser.add_argument("--reuse_pool", dest="reuse_pool", action="store_true", default=True,
                        help="Reuse one ProcessPoolExecutor per benchmark file for ALNS seeds.")

    parser.add_argument("--no_reuse_pool", dest="reuse_pool", action="store_false",
                        help="Create a fresh ProcessPoolExecutor for each instance.")
    return parser.parse_args()


def build_data_instances(data):
    batch_size = data["node_demand"].shape[0]
    instances = []
    for i in range(batch_size):
        demand = np.insert(data["node_demand"][i].numpy(), 0, 0.0)
        all_xy = np.vstack(
            [data["depot_xy"][i].numpy().squeeze(0), data["node_xy"][i].numpy()])
        dm = np.linalg.norm(all_xy[:, None, :] - all_xy[None, :, :], axis=-1)
        instances.append(
            {
                "all_xy": all_xy,
                "dist_matrix": dm,
                "demand": demand,
                "capacity": data["agent_capacity"][i].numpy(),
                "fixed_cost": data["agent_fixed_cost"][i].numpy(),
                "var_cost": data["agent_variable_cost"][i].numpy(),
            }
        )
    return instances


def _warm_destroy_bounds(n_customers):

    low_pct = 0.15 if n_customers >= 50 else 0.05
    high_pct = 0.40 if n_customers >= 50 else 0.15

    low = max(2, int(low_pct * n_customers))
    high = max(low, int(high_pct * n_customers))
    return low, high


def warm_random_removal(state, rng, **kwargs):
    new_state = state.copy()
    if not new_state.routes:
        return new_state
    n = len(state.data["demand"]) - 1
    low, high = _warm_destroy_bounds(n)
    k = int(rng.integers(low, high + 1))
    pool = [(cust, route_idx) for route_idx, (path, _, __)
            in enumerate(new_state.routes) for cust in path[1:-1]]
    if not pool:
        return new_state
    rng.shuffle(pool)
    for cust, route_idx in pool[: min(k, len(pool))]:
        if cust in new_state.routes[route_idx][0]:
            new_state.routes[route_idx][0].remove(cust)
            new_state.routes[route_idx][2] -= new_state.data["demand"][cust]
            new_state.unassigned.add(cust)
    new_state.routes = [
        route for route in new_state.routes if len(route[0]) > 2]
    return new_state


def warm_worst_removal(state, rng, p: float = 4.0, **kwargs):
    """
    Removes customers that add the most detour cost to their routes,
    using controlled randomness (y^p index) to prevent deterministic looping.
    Signature matches ALNS convention: (state, rng, **kwargs).
    """
    new_state = state.copy()
    if not new_state.routes:
        return new_state

    n = len(state.data["demand"]) - 1
    low = max(2, int(0.08 * n))
    high = max(low, int(0.22 * n))
    k = int(rng.integers(low, high + 1))

    adaptive_p = float(rng.uniform(max(1.5, p - 1.5), p + 1.0))

    per_route_limit = max(1, int(math.ceil(k / max(2, len(new_state.routes)))))

    dm = new_state.data["dist_matrix"]

    scored = []
    for ri, (path, v_type, load) in enumerate(new_state.routes):
        vc = new_state.data["var_cost"][v_type]
        cap = new_state.data["capacity"][v_type]
        util = load / cap if cap > 0 else 1.0
        for pos in range(1, len(path) - 1):
            cust = path[pos]
            detour = (
                dm[path[pos - 1], cust]
                + dm[cust, path[pos + 1]]
                - dm[path[pos - 1], path[pos + 1]]
            ) * vc

            slack_bonus = 1.0 + max(0.0, 1.0 - util)
            scored.append((detour * slack_bonus, cust, ri))

    if not scored:
        return new_state

    scored.sort(key=lambda x: x[0], reverse=True)

    removed = 0
    already_removed = set()
    removed_by_route = {}

    while removed < k and scored:
        y = rng.random()
        pop_index = int((y ** adaptive_p) * len(scored))
        pop_index = min(pop_index, len(scored) - 1)
        detour, cust, ri = scored.pop(pop_index)

        if (cust, ri) in already_removed:
            continue

        if removed_by_route.get(ri, 0) >= per_route_limit and any(
            removed_by_route.get(candidate_ri, 0) < per_route_limit for _, _, candidate_ri in scored
        ):
            continue

        if ri < len(new_state.routes) and cust in new_state.routes[ri][0]:
            new_state.routes[ri][0].remove(cust)
            new_state.routes[ri][2] -= new_state.data["demand"][cust]
            new_state.unassigned.add(cust)
            already_removed.add((cust, ri))
            removed_by_route[ri] = removed_by_route.get(ri, 0) + 1
            removed += 1

    new_state.routes = [r for r in new_state.routes if len(r[0]) > 2]
    return new_state


def h_route_removal(state, rng, **kwargs):
    """Destroys inefficient whole routes to force better fleet reassignment."""
    new_state = state.copy()
    if not new_state.routes:
        return new_state

    max_routes = max(3, int(0.20 * len(new_state.routes)) + 1)
    num_routes_to_kill = min(len(new_state.routes),
                             int(rng.integers(1, max_routes)))
    dm = new_state.data['dist_matrix']

    for _ in range(num_routes_to_kill):
        if not new_state.routes:
            break
        worst_ri = -1
        worst_score = -float('inf')

        for ri, (path, vt, load) in enumerate(new_state.routes):
            n_cust = len(path) - 2
            if n_cust <= 0:
                continue
            cap = new_state.data['capacity'][vt]
            util = load / cap if cap > 0 else 1.0
            route_dist = _route_dist(path, dm)
            route_cost = new_state.data['fixed_cost'][vt] + \
                route_dist * new_state.data['var_cost'][vt]

            cost_per_customer = route_cost / max(1, n_cust)
            cost_per_load = route_cost / max(load, 1e-9)
            slack_penalty = 1.0 + max(0.0, 1.0 - util)
            score = (0.65 * cost_per_customer + 0.35 *
                     cost_per_load) * slack_penalty

            if score > worst_score:
                worst_score = score
                worst_ri = ri

        if worst_ri != -1:

            for c in new_state.routes[worst_ri][0][1:-1]:
                new_state.unassigned.add(c)
            new_state.routes.pop(worst_ri)

    return new_state


def warm_shaw_d_removal(state, rng, p: float = 4.0, **kwargs):
    """
    Shaw removal: removes a cluster of customers based on both geographic
    distance and demand similarity.
    Uses controlled randomness (y^p index) to balance diversity vs. greediness.
    """
    new_state = state.copy()
    if not new_state.routes:
        return new_state

    n = len(state.data["demand"]) - 1
    low, high = _warm_destroy_bounds(n)
    k = int(rng.integers(low, high + 1))

    dm = new_state.data["dist_matrix"]

    pool = [
        (cust, ri)
        for ri, (path, _, __) in enumerate(new_state.routes)
        for cust in path[1:-1]
    ]
    if not pool:
        return new_state

    seed_idx = int(rng.integers(0, len(pool)))
    seed_cust, _ = pool[seed_idx]

    L = [item for item in pool if item[0] != seed_cust]

    max_dist = np.max(dm) if np.max(dm) > 0 else 1.0
    max_dem = np.max(state.data["demand"]) if np.max(
        state.data["demand"]) > 0 else 1.0

    def shaw_score(cust_a, cust_b):
        dist_val = dm[cust_a, cust_b] / max_dist
        dem_val = abs(state.data["demand"][cust_a] -
                      state.data["demand"][cust_b]) / max_dem

        return 0.5 * dist_val + 0.5 * dem_val

    L.sort(key=lambda item: shaw_score(seed_cust, item[0]))

    to_remove = [pool[seed_idx]]

    while len(to_remove) < k and L:
        y = rng.random()
        pop_index = int((y ** p) * len(L))
        pop_index = min(pop_index, len(L) - 1)
        to_remove.append(L.pop(pop_index))

    for cust, ri in to_remove:
        if ri < len(new_state.routes) and cust in new_state.routes[ri][0]:
            new_state.routes[ri][0].remove(cust)
            new_state.routes[ri][2] -= new_state.data["demand"][cust]
            new_state.unassigned.add(cust)

    new_state.routes = [r for r in new_state.routes if len(r[0]) > 2]
    return new_state


def vehicle_type_removal(state, rng, p: float = 3.0, **kwargs):
    """
    Improved Vehicle Type Removal:
    Identifies vehicle types that are suffering from poor capacity utilization
    (wasting fixed costs on empty space). It selects an inefficient type and
    removes its emptiest routes up to the standard 'k' destruction limit.
    """
    new_state = state.copy()
    if not new_state.routes:
        return new_state

    n = len(state.data["demand"]) - 1
    low, high = _warm_destroy_bounds(n)
    k = int(rng.integers(low, high + 1))

    vtype_utilization = {}
    vtype_routes = {}

    for ri, (path, vt, load) in enumerate(new_state.routes):
        cap = new_state.data["capacity"][vt]

        util = load / cap if cap > 0 else 1.0

        if vt not in vtype_utilization:
            vtype_utilization[vt] = []
            vtype_routes[vt] = []

        vtype_utilization[vt].append(util)
        vtype_routes[vt].append((util, ri))

    if not vtype_utilization:
        return new_state

    scored_types = []
    for vt, utils in vtype_utilization.items():
        avg_util = sum(utils) / len(utils)
        scored_types.append((avg_util, vt))

    scored_types.sort(key=lambda x: x[0])

    y = rng.random()
    pop_index = int((y ** p) * len(scored_types))
    pop_index = min(pop_index, len(scored_types) - 1)
    target_vt = scored_types[pop_index][1]

    routes_of_type = vtype_routes[target_vt]
    routes_of_type.sort(key=lambda x: x[0])

    removed_count = 0
    routes_to_pop = []

    for util, ri in routes_of_type:
        if removed_count >= k and removed_count > 0:
            break

        n_cust = len(new_state.routes[ri][0]) - 2
        removed_count += n_cust
        routes_to_pop.append(ri)

    for ri in sorted(routes_to_pop, reverse=True):
        path = new_state.routes[ri][0]
        for cust in path[1:-1]:
            new_state.unassigned.add(cust)
        new_state.routes.pop(ri)

    return new_state


def repair_with_cross_exchange(state, rng, **kwargs):
    """Greedy repair followed by segment swapping between different routes."""
    new_state = greedy_repair(state, rng, **kwargs)
    dm = new_state.data["dist_matrix"]

    improved = True
    max_loops = 20
    loops = 0

    while improved and loops < max_loops:
        improved = False
        loops += 1

        route_pairs = [(ri, rj) for ri in range(len(new_state.routes))
                       for rj in range(ri + 1, len(new_state.routes))]
        rng.shuffle(route_pairs)

        for ri, rj in route_pairs:
            if improved:
                break
            path_i, vt_i, load_i = new_state.routes[ri]
            path_j, vt_j, load_j = new_state.routes[rj]

            rd_i = _route_dist(path_i, dm)
            _, cost_i_old = _best_vehicle(load_i, rd_i, new_state.data)
            rd_j = _route_dist(path_j, dm)
            _, cost_j_old = _best_vehicle(load_j, rd_j, new_state.data)

            for len_i in [1, 2, 3]:
                if improved:
                    break
                if len(path_i) <= len_i + 1:
                    continue
                for pos_i in range(1, len(path_i) - len_i):
                    if improved:
                        break
                    seg_i = path_i[pos_i: pos_i + len_i]
                    dem_i = sum(new_state.data["demand"][c] for c in seg_i)

                    for len_j in [1, 2, 3]:
                        if improved:
                            break
                        if len(path_j) <= len_j + 1:
                            continue
                        for pos_j in range(1, len(path_j) - len_j):
                            seg_j = path_j[pos_j: pos_j + len_j]
                            dem_j = sum(
                                new_state.data["demand"][c] for c in seg_j)

                            new_load_i = load_i - dem_i + dem_j
                            new_load_j = load_j - dem_j + dem_i
                            max_cap = max(new_state.data["capacity"])
                            if new_load_i > max_cap or new_load_j > max_cap:
                                continue

                            a_i, d_i = path_i[pos_i - 1], path_i[pos_i + len_i]
                            dist_saved_i = dm[a_i, seg_i[0]] + \
                                dm[seg_i[-1], d_i] - dm[a_i, d_i]
                            insert_dist_i = dm[a_i, seg_j[0]] + \
                                dm[seg_j[-1], d_i] - dm[a_i, d_i]
                            new_rd_i = rd_i - dist_saved_i + insert_dist_i
                            new_vt_i, cost_i_new = _best_vehicle(
                                new_load_i, new_rd_i, new_state.data)

                            if new_vt_i == -1:
                                continue

                            a_j, d_j = path_j[pos_j - 1], path_j[pos_j + len_j]
                            dist_saved_j = dm[a_j, seg_j[0]] + \
                                dm[seg_j[-1], d_j] - dm[a_j, d_j]
                            insert_dist_j = dm[a_j, seg_i[0]] + \
                                dm[seg_i[-1], d_j] - dm[a_j, d_j]
                            new_rd_j = rd_j - dist_saved_j + insert_dist_j
                            new_vt_j, cost_j_new = _best_vehicle(
                                new_load_j, new_rd_j, new_state.data)

                            if new_vt_j == -1:
                                continue

                            gain = (cost_i_old + cost_j_old) - \
                                (cost_i_new + cost_j_new)

                            if gain > 0.01:
                                new_path_i = path_i[:pos_i] + \
                                    seg_j + path_i[pos_i + len_i:]
                                new_path_j = path_j[:pos_j] + \
                                    seg_i + path_j[pos_j + len_j:]

                                if len(new_path_i) > 4:
                                    new_path_i = _or_opt(
                                        _two_opt(new_path_i, dm), dm)
                                if len(new_path_j) > 4:
                                    new_path_j = _or_opt(
                                        _two_opt(new_path_j, dm), dm)

                                new_state.routes[ri] = [
                                    new_path_i, new_vt_i, new_load_i]
                                new_state.routes[rj] = [
                                    new_path_j, new_vt_j, new_load_j]
                                improved = True
                                break

    new_state.routes = [r for r in new_state.routes if len(r[0]) > 2]
    return _vehicle_swap(new_state)


def regret4_repair(state, rng, **kwargs):
    """
    Regret-4: Inserts customers one by one, prioritizing those with the highest
    regret cost across their top 4 insertion options.
    """
    new_state = state.copy()
    customers = list(new_state.unassigned)
    dm = new_state.data['dist_matrix']

    while customers:
        best_cust, best_ins, best_regret = None, None, -float('inf')

        for cust in customers:
            cd = new_state.data['demand'][cust]
            options = []
            max_fleet_capacity = max(new_state.data['capacity'])

            for ri, (path, vt, load) in enumerate(new_state.routes):
                new_load = load + cd
                if new_load > max_fleet_capacity:
                    continue

                old_rd = _route_dist(path, dm)
                old_vt, old_cost = _best_vehicle(load, old_rd, new_state.data)
                for pos in range(1, len(path)):
                    u, v = path[pos - 1], path[pos]
                    extra = dm[u, cust] + dm[cust, v] - dm[u, v]
                    new_rd = old_rd + extra
                    new_vt, new_cost = _best_vehicle(
                        new_load, new_rd, new_state.data)

                    if new_vt == -1:
                        continue
                    options.append(
                        (new_cost - old_cost, ri, pos, None, new_vt))

            for vt in range(len(new_state.data['capacity'])):
                if cd <= new_state.data['capacity'][vt]:
                    rd = 2 * dm[0, cust]
                    c = new_state.data['fixed_cost'][vt] + \
                        rd * new_state.data['var_cost'][vt]
                    options.append((c, None, None, vt, vt))

            if not options:
                continue

            options.sort(key=lambda x: x[0])

            c1 = options[0][0]

            c2 = options[1][0] if len(options) > 1 else c1 + 1e9
            c3 = options[2][0] if len(options) > 2 else c2 + 1e9
            c4 = options[3][0] if len(options) > 3 else c3 + 1e9

            regret = (c4 - c1) + (c3 - c1) + (c2 - c1)

            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_ins = options[0]

        if best_cust is None:
            break

        diff, ri, pos, new_route_vt, chosen_vt = best_ins
        cd = new_state.data['demand'][best_cust]

        if new_route_vt is not None:
            new_state.routes.append([[0, best_cust, 0], new_route_vt, cd])
        else:
            new_state.routes[ri][0].insert(pos, best_cust)
            new_state.routes[ri][2] += cd
            new_state.routes[ri][1] = chosen_vt

        new_state.unassigned.discard(best_cust)
        customers.remove(best_cust)

    return _vehicle_swap(new_state)


def repair_with_or_opt(state, rng, **kwargs):

    new_state = greedy_repair(state, rng, **kwargs)

    if rng.random() > 0.20:
        return _vehicle_swap(new_state)

    dm = new_state.data["dist_matrix"]

    improved = True

    max_loops = 30
    loops = 0

    while improved:
        improved = False
        loops += 1
        for ri in range(len(new_state.routes)):
            if improved:
                break
            path_i, vt_i, load_i = new_state.routes[ri]

            for seg_len in [1, 2]:
                if improved:
                    break
                for pos in range(1, len(path_i) - seg_len):
                    if improved:
                        break
                    seg = path_i[pos: pos + seg_len]
                    seg_demand = sum(new_state.data["demand"][c] for c in seg)

                    a = path_i[pos - 1]
                    d = path_i[pos + seg_len]
                    removal_dist_saved = dm[a, seg[0]
                                            ] + dm[seg[-1], d] - dm[a, d]

                    rd_i = _route_dist(path_i, dm)
                    _, cost_i_old = _best_vehicle(load_i, rd_i, new_state.data)
                    new_load_i = load_i - seg_demand
                    new_rd_i = rd_i - removal_dist_saved
                    new_vt_i, cost_i_new = _best_vehicle(
                        new_load_i, new_rd_i, new_state.data)

                    if new_vt_i == -1:
                        continue

                    for rj in range(len(new_state.routes)):
                        if improved:
                            break
                        if rj == ri:
                            continue
                        path_j, vt_j, load_j = new_state.routes[rj]
                        new_load_j = load_j + seg_demand
                        if new_load_j > max(new_state.data["capacity"]):
                            continue

                        rd_j = _route_dist(path_j, dm)
                        _, cost_j_old = _best_vehicle(
                            load_j, rd_j, new_state.data)

                        for ins_pos in range(1, len(path_j)):
                            u, v = path_j[ins_pos - 1], path_j[ins_pos]
                            insert_dist = dm[u, seg[0]] + \
                                dm[seg[-1], v] - dm[u, v]
                            new_rd_j = rd_j + insert_dist
                            new_vt_j, cost_j_new = _best_vehicle(
                                new_load_j, new_rd_j, new_state.data)

                            if new_vt_j == -1:
                                continue

                            gain = (cost_i_old - cost_i_new) + \
                                (cost_j_old - cost_j_new)

                            if gain > 0.01:
                                del path_i[pos: pos + seg_len]
                                new_state.routes[ri][2] = new_load_i
                                new_state.routes[ri][1] = new_vt_i

                                for offset, c in enumerate(seg):
                                    path_j.insert(ins_pos + offset, c)
                                new_state.routes[rj][2] = new_load_j
                                new_state.routes[rj][1] = new_vt_j

                                improved = True
                                break

        new_state.routes = [r for r in new_state.routes if len(r[0]) > 2]

    return _vehicle_swap(new_state)


def build_scratch_initial_state(inst, rng):
    empty_state = FSMVRP_State([], set(range(1, len(inst["demand"]))), inst)
    return regret4_repair(empty_state, rng)


def _selector_weight_list(selector, attr_names):
    """Reads selector weight arrays from public or private ALNS attributes."""
    for attr_name in attr_names:
        if hasattr(selector, attr_name):
            weights = getattr(selector, attr_name)
            if weights is not None:

                return np.asarray(weights, dtype=float).reshape(-1).tolist()
    return []


def extract_selector_weights(selector):
    """Extracts final destroy and repair operator weights from ALNS selector variants."""

    destroy_weights = _selector_weight_list(
        selector, ("destroy_weights", "d_weights", "_d_weights"))
    repair_weights = _selector_weight_list(
        selector, ("repair_weights", "r_weights", "_r_weights"))

    if destroy_weights or repair_weights:
        return {
            "destroy_weights": destroy_weights,
            "repair_weights": repair_weights,
        }

    unified_weights = _selector_weight_list(
        selector, ("weights", "operator_weights", "_weights", "_operator_weights"))
    return {"weights": unified_weights} if unified_weights else {}


def _set_selector_repair_weight(selector, repair_idx, weight):
    """Sets one repair operator's starting weight across known ALNS selector layouts."""

    for attr_name in ("repair_weights", "r_weights", "_r_weights"):
        if hasattr(selector, attr_name):
            weights = getattr(selector, attr_name)
            if weights is not None and len(weights) > repair_idx:
                weights[repair_idx] = weight

    for attr_name in ("weights", "operator_weights", "_weights", "_operator_weights"):
        if hasattr(selector, attr_name):
            weights = getattr(selector, attr_name)
            if weights is not None:
                arr = np.asarray(weights)
                if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > repair_idx:
                    weights[1][repair_idx] = weight


def noisy_regret_repair(state, rng, **kwargs):
    """Regret-4 Repair with +/- 20% noise injected into the evaluation costs."""
    new_state = state.copy()
    customers = list(new_state.unassigned)
    dm = new_state.data['dist_matrix']

    while customers:
        best_cust, best_ins, best_regret = None, None, -float('inf')

        for cust in customers:
            cd = new_state.data['demand'][cust]
            options = []
            max_fleet_capacity = max(new_state.data['capacity'])

            for ri, (path, vt, load) in enumerate(new_state.routes):
                new_load = load + cd
                if new_load > max_fleet_capacity:
                    continue

                old_rd = _route_dist(path, dm)
                old_vt, old_cost = _best_vehicle(load, old_rd, new_state.data)
                for pos in range(1, len(path)):
                    u, v = path[pos - 1], path[pos]
                    extra = dm[u, cust] + dm[cust, v] - dm[u, v]
                    new_rd = old_rd + extra
                    new_vt, new_cost = _best_vehicle(
                        new_load, new_rd, new_state.data)

                    if new_vt == -1:
                        continue

                    noise = rng.uniform(0.8, 1.2)
                    cost_diff = (new_cost - old_cost) * noise
                    options.append((cost_diff, ri, pos, None, new_vt))

            for vt in range(len(new_state.data['capacity'])):
                if cd <= new_state.data['capacity'][vt]:
                    rd = 2 * dm[0, cust]
                    c = new_state.data['fixed_cost'][vt] + \
                        rd * new_state.data['var_cost'][vt]
                    noise = rng.uniform(0.8, 1.2)
                    c_noisy = c * noise
                    options.append((c_noisy, None, None, vt, vt))

            if not options:
                continue

            options.sort(key=lambda x: x[0])
            c1 = options[0][0]
            c2 = options[1][0] if len(options) > 1 else c1 + 1e9
            c3 = options[2][0] if len(options) > 2 else c2 + 1e9
            c4 = options[3][0] if len(options) > 3 else c3 + 1e9

            regret = (c4 - c1) + (c3 - c1) + (c2 - c1)

            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_ins = options[0]

        if best_cust is None:
            break

        diff, ri, pos, new_route_vt, chosen_vt = best_ins
        cd = new_state.data['demand'][best_cust]

        if new_route_vt is not None:
            new_state.routes.append([[0, best_cust, 0], new_route_vt, cd])
        else:
            new_state.routes[ri][0].insert(pos, best_cust)
            new_state.routes[ri][2] += cd
            new_state.routes[ri][1] = chosen_vt

        new_state.unassigned.discard(best_cust)
        customers.remove(best_cust)

    return _vehicle_swap(new_state)


def neighbor_route_removal(state, rng, **kwargs):
    """
    Destroys a route and its closest geographic neighboring route.
    Highly optimized for dense 100-node instances to force large-scale local repacking.
    """
    new_state = state.copy()
    if len(new_state.routes) < 2:
        return new_state

    n = len(state.data["demand"]) - 1
    low, high = _warm_destroy_bounds(n)
    k = int(rng.integers(low, high + 1))

    dm = new_state.data["dist_matrix"]

    route_probs = np.array([1.0 / max(1, len(r[0]) - 2)
                           for r in new_state.routes])
    route_probs /= route_probs.sum()
    seed_ri = rng.choice(len(new_state.routes), p=route_probs)

    seed_path = new_state.routes[seed_ri][0][1:-1]
    if not seed_path:
        return new_state

    best_dist = float('inf')
    neighbor_ri = -1

    for ri, (path, _, _) in enumerate(new_state.routes):
        if ri == seed_ri:
            continue
        comp_path = path[1:-1]
        if not comp_path:
            continue

        sub_dm = dm[np.ix_(seed_path, comp_path)]
        dist = sub_dm.min()

        if dist < best_dist:
            best_dist = dist
            neighbor_ri = ri

    if neighbor_ri == -1:
        return new_state

    routes_to_kill = sorted([seed_ri, neighbor_ri], reverse=True)
    removed_count = 0

    for ri in routes_to_kill:
        if removed_count >= k and removed_count > 0:
            continue

        path = new_state.routes[ri][0]
        n_cust = len(path) - 2
        for cust in path[1:-1]:
            new_state.unassigned.add(cust)
        new_state.routes.pop(ri)
        removed_count += n_cust

    return new_state


def repair_with_regret_or_opt(state, rng, **kwargs):
    """
    Uses the strong regret repair, followed by aggressive intra-route and inter-route
    2-opt / Or-opt to strictly optimize the geometry. Excellent for late-stage 100-node packing.
    """
    new_state = noisy_regret_repair(state, rng, **kwargs)

    if rng.random() > 0.20:
        return new_state

    dm = new_state.data["dist_matrix"]
    improved = True
    loops = 0
    max_loops = 20

    while improved and loops < max_loops:
        improved = False
        loops += 1

        for ri, (path_i, vt_i, load_i) in enumerate(new_state.routes):
            if len(path_i) > 4:
                new_path = _two_opt(path_i, dm)
                if new_path != path_i:
                    new_state.routes[ri][0] = new_path
                    improved = True

        for ri in range(len(new_state.routes)):
            if improved:
                break
            path_i, vt_i, load_i = new_state.routes[ri]

            for seg_len in [1, 2]:
                if improved:
                    break
                for pos in range(1, len(path_i) - seg_len):
                    if improved:
                        break
                    seg = path_i[pos: pos + seg_len]
                    seg_demand = sum(new_state.data["demand"][c] for c in seg)

                    a = path_i[pos - 1]
                    d = path_i[pos + seg_len]
                    removal_dist_saved = dm[a, seg[0]
                                            ] + dm[seg[-1], d] - dm[a, d]

                    rd_i = _route_dist(path_i, dm)
                    _, cost_i_old = _best_vehicle(load_i, rd_i, new_state.data)
                    new_load_i = load_i - seg_demand
                    new_rd_i = rd_i - removal_dist_saved
                    new_vt_i, cost_i_new = _best_vehicle(
                        new_load_i, new_rd_i, new_state.data)

                    if new_vt_i == -1:
                        continue

                    for rj in range(len(new_state.routes)):
                        if improved:
                            break
                        if rj == ri:
                            continue
                        path_j, vt_j, load_j = new_state.routes[rj]
                        new_load_j = load_j + seg_demand
                        if new_load_j > max(new_state.data["capacity"]):
                            continue

                        rd_j = _route_dist(path_j, dm)
                        _, cost_j_old = _best_vehicle(
                            load_j, rd_j, new_state.data)

                        for ins_pos in range(1, len(path_j)):
                            u, v = path_j[ins_pos - 1], path_j[ins_pos]
                            insert_dist = dm[u, seg[0]] + \
                                dm[seg[-1], v] - dm[u, v]
                            new_rd_j = rd_j + insert_dist
                            new_vt_j, cost_j_new = _best_vehicle(
                                new_load_j, new_rd_j, new_state.data)

                            if new_vt_j == -1:
                                continue
                            gain = (cost_i_old - cost_i_new) + \
                                (cost_j_old - cost_j_new)

                            if gain > 0.01:
                                del path_i[pos: pos + seg_len]
                                new_state.routes[ri][2] = new_load_i
                                new_state.routes[ri][1] = new_vt_i

                                for offset, c in enumerate(seg):
                                    path_j.insert(ins_pos + offset, c)
                                new_state.routes[rj][2] = new_load_j
                                new_state.routes[rj][1] = new_vt_j

                                improved = True
                                break

    new_state.routes = [r for r in new_state.routes if len(r[0]) > 2]
    return _vehicle_swap(new_state)


def run_pure_alns(init_state, run_idx=0, seed=1234, get_config=False):

    rng = np.random.default_rng(seed + run_idx * 10000)
    init = init_state.copy()

    alns = ALNS(rng)

    n_customers = len(init_state.data["demand"]) - 0

    alns.add_destroy_operator(h_route_removal)
    alns.add_destroy_operator(warm_worst_removal)
    alns.add_destroy_operator(vehicle_type_removal)
    alns.add_destroy_operator(warm_shaw_d_removal)
    if n_customers >= 40:
        alns.add_destroy_operator(warm_random_removal)
        alns.add_destroy_operator(neighbor_route_removal)
    alns.add_repair_operator(repair_with_regret_or_opt)
    alns.add_repair_operator(noisy_regret_repair)
    alns.add_repair_operator(repair_with_cross_exchange)

    init_obj = float(init.objective())
    start_temp = 0.005 * init_obj
    end_temp = 0.0001 * init_obj

    total_iterations = 2000
    dynamic_seg_length = max(25, int(total_iterations / 25))

    cooling_step = (end_temp / start_temp) ** (1.0 / total_iterations)

    select_scores = [33, 9, 3, 0]
    select_decay = 0.94
    select = SegmentedRouletteWheel(select_scores, select_decay, dynamic_seg_length, len(
        alns.destroy_operators), len(alns.repair_operators))
    _set_selector_repair_weight(select, len(alns.repair_operators) - 1, 0.25)
    accept = SimulatedAnnealing(
        start_temp, end_temp, cooling_step, "exponential")

    stagnation_patience = max(150, int(0.30 * total_iterations))
    min_iterations = max(200, int(0.50 * total_iterations))
    stop = MaxRuntime(30 / len(ALNS_SEEDS))

    if get_config:
        def _get_name(op):

            base_op = op[0] if isinstance(op, tuple) else op

            if isinstance(base_op, str):
                return base_op
            return getattr(base_op, '__name__', base_op.__class__.__name__)

        return {
            "destroy_operators": [_get_name(op) for op in alns.destroy_operators],
            "repair_operators": [_get_name(op) for op in alns.repair_operators],
            "iterations": total_iterations,
            "stagnation_patience": stagnation_patience,
            "min_iterations": min_iterations,
            "start_temp": start_temp,
            "end_temp": end_temp,
            "cooling_step": cooling_step,
            "scores": select_scores,
            "decay": select_decay,
            "seg_length": dynamic_seg_length,
        }
    result = alns.iterate(init, select, accept, stop)

    total_iterations = sum(
        sum(counts) for counts in result.statistics.destroy_operator_counts.values())
    seed_stats = {
        "seed": seed,
        "iterations": total_iterations,
        "stop_iterations": getattr(stop, "iterations", total_iterations),
        "stagnation_patience": stagnation_patience,
        "no_improve_iterations": getattr(stop, "no_improve_iterations", 0),
        "destroy": {k: list(v) for k, v in result.statistics.destroy_operator_counts.items()},
        "repair": {k: list(v) for k, v in result.statistics.repair_operator_counts.items()}
    }
    seed_stats.update(extract_selector_weights(select))

    cost = float(result.best_state.objective())

    return cost, result.best_state, [seed_stats]


def _format_metric(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def _format_gap(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:+.2f}%"


def log_instance_table(rows, logger):
    header = f"{'Inst':>4} | {'Pure ALNS':>10} | {'Best-Known':>10} | {'Gap(Best)':>10} | {'Gap(Paper)':>10} | {'Time(s)':>8} | {'Total iterations':>6}"
    logger.info(header)
    logger.info("%s", "-" * len(header))
    for row in rows:
        gap_b = row.get("alns_gap_best", float("nan"))
        gap_p = row.get("alns_gap_paper", float("nan"))
        iters = row.get("best_iters", 0)

        row_string = (
            f"{row['instance']:>4} | "
            f"{_format_metric(row['alns_cost']):>10} | "
            f"{_format_metric(row['best_known']):>10} | "
            f"{_format_gap(gap_b):>10} | "
            f"{_format_gap(gap_p):>10} | "
            f"{_format_metric(row.get('time', float('nan'))):>8} | "
            f"{iters:>6}"
        )

        logger.info("%s", row_string)


def log_summary(rows, logger):
    alns = np.array([row["alns_cost"] for row in rows], dtype=float)
    gaps_best = np.array([row["alns_gap_best"] for row in rows], dtype=float)
    gaps_paper = np.array([row["alns_gap_paper"] for row in rows], dtype=float)
    times = np.array([row.get("time", float("nan"))
                     for row in rows], dtype=float)
    iters = np.array([row.get("best_iters", 0) for row in rows], dtype=float)

    logger.info("")
    logger.info("Summary")
    logger.info("-------")
    logger.info("Instances        : %d", len(rows))
    logger.info("Failed instances : %d", int(np.isnan(alns).sum()))
    logger.info("Avg ALNS cost    : %s",
                _format_metric(float(np.nanmean(alns)) if not np.isnan(alns).all() else float("nan")))
    logger.info("Avg Gap(Best)    : %s",
                _format_gap(float(np.nanmean(gaps_best)) if not np.isnan(gaps_best).all() else float("nan")))
    logger.info("Avg Gap(Paper)   : %s",
                _format_gap(float(np.nanmean(gaps_paper)) if not np.isnan(gaps_paper).all() else float("nan")))
    logger.info("Avg Time/Inst(s) : %s",
                _format_metric(float(np.nanmean(times)) if not np.isnan(times).all() else float("nan")))
    logger.info("Total Time(s)    : %s",
                _format_metric(float(np.nansum(times)) if not np.isnan(times).all() else float("nan")))
    logger.info("Avg Iterations   : %s",
                _format_metric(float(np.nanmean(iters)) if not np.isnan(iters).all() else float("nan")))


def export_operator_stats_csv(instance_id, method_type, run_idx, stats_list, log_dir):
    """Saves ALNS operator performance to a dedicated CSV file (Segmented Wheel Compatible)."""
    csv_path = Path(log_dir) / "operator_stats.csv"
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Instance", "Method", "Run", "Seed", "Phase",
                "Operator", "Best", "Better", "Accepted", "Rejected", "Total", "Final_Weight"
            ])

        for stat in stats_list:
            seed = stat["seed"]

            d_weights = stat.get("destroy_weights", [])
            r_weights = stat.get("repair_weights", [])
            unified_weights = stat.get("weights", [])

            if unified_weights and not d_weights:
                num_destroy = len(stat["destroy"])
                d_weights = unified_weights[:num_destroy]
                r_weights = unified_weights[num_destroy:]

            for idx, (op_name, counts) in enumerate(stat["destroy"].items()):
                best, better, acc, rej = counts
                total = sum(counts)
                weight = f"{d_weights[idx]:.4f}" if idx < len(
                    d_weights) else "N/A"
                writer.writerow([
                    instance_id, method_type, run_idx, seed, "Destroy",
                    op_name, best, better, acc, rej, total, weight
                ])

            for idx, (op_name, counts) in enumerate(stat["repair"].items()):
                best, better, acc, rej = counts
                total = sum(counts)
                weight = f"{r_weights[idx]:.4f}" if idx < len(
                    r_weights) else "N/A"
                writer.writerow([
                    instance_id, method_type, run_idx, seed, "Repair",
                    op_name, best, better, acc, rej, total, weight
                ])


def collect_alns_seed_results(executor, init_state, run_idx, active_seeds, hard_timeout_limit, logger, instance_number):
    """Runs one pure ALNS job per seed and returns the best completed seed result."""

    future_to_seed = {
        executor.submit(run_pure_alns, init_state, run_idx, seed): seed
        for seed in active_seeds
    }

    alns_cost_raw = float('inf')
    best_alns_state = None
    all_alns_stats = []
    best_alns_iters = 0

    try:

        completed_futures = as_completed(
            future_to_seed, timeout=hard_timeout_limit)
        for future in completed_futures:
            seed = future_to_seed[future]
            try:

                cost, b_state, stats = future.result()
                all_alns_stats.extend(stats)

                if cost < alns_cost_raw:
                    alns_cost_raw = cost
                    best_alns_state = b_state
                    best_alns_iters = stats[0].get(
                        "iterations", 0) if stats else 0
            except Exception as e:
                logger.error(
                    f"ALNS seed failed on Seed {seed} for Instance {instance_number}: {e}")
    except TimeoutError:

        for future, seed in future_to_seed.items():
            if not future.done():
                future.cancel()
                logger.error(
                    f"ALNS seed timed out on Seed {seed} for Instance {instance_number}")

    return alns_cost_raw, best_alns_state, all_alns_stats, best_alns_iters


def process_benchmark_file(pt_file: Path, raw_pt: Path, method: str, instance_id: str, args, logger, run_idx: int,
                           dataset: str = "GoldenBC", num_to_test: int = None):
    """Processes one benchmark tensor file with pure ALNS only."""
    logger.info("==================================================")
    logger.info("Processing benchmark file: %s", pt_file.name)

    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    instances = build_data_instances(data)
    if num_to_test is not None:
        instances = instances[:num_to_test]
    logger.info("Loaded %d instances (testing %d)",
                data["node_demand"].shape[0], len(instances))

    if dataset == "GoldenBC":
        L = get_L_for_method(str(raw_pt), method)
        best_known = BEST_KNOWN_MAP.get(instance_id)
        paper_cost = PAPER_REPORT_MAP.get(instance_id)
    else:
        L = 1.0
        best_known = TEST_TENSOR_BASELINES.get(int(instance_id))
        paper_cost = None

    logger.info(f"Scaling Factor L = {L:.4f}")
    if best_known:
        logger.info(f"Best-Known Cost  = {best_known:.2f}")
    if paper_cost:
        logger.info(f"Wan Report     = {paper_cost:.2f}")

    rows = []
    max_cores = len(ALNS_SEEDS)
    active_seeds = ALNS_SEEDS[:max_cores]
    hard_timeout_limit = 10000
    shared_executor = None

    if args.reuse_pool:
        logger.info(
            "Reusing one pure ALNS process pool with %d workers", max_cores)
        shared_executor = ProcessPoolExecutor(max_workers=max_cores)

    for i, inst in enumerate(instances):
        start_time = time.time()
        try:
            init_rng = np.random.default_rng(
                ALNS_SEEDS[0] + run_idx * 10000 + i)
            init_state = build_scratch_initial_state(inst, init_rng)
            init_cost_raw = float(init_state.objective())

            if shared_executor is None:
                with ProcessPoolExecutor(max_workers=max_cores) as executor:
                    alns_cost_raw, best_alns_state, all_alns_stats, best_alns_iters = collect_alns_seed_results(
                        executor, init_state, run_idx, active_seeds, hard_timeout_limit, logger, i + 1
                    )
            else:
                alns_cost_raw, best_alns_state, all_alns_stats, best_alns_iters = collect_alns_seed_results(
                    shared_executor, init_state, run_idx, active_seeds, hard_timeout_limit, logger, i + 1
                )

            log_directory = Path(logger.handlers[0].baseFilename).parent
            export_operator_stats_csv(
                instance_id, "PureALNS", run_idx, all_alns_stats, log_directory)

            if best_alns_state is not None:
                export_best_routes_csv(
                    instance_id, "PureALNS", run_idx, best_alns_state, log_directory)

            init_cost = init_cost_raw * L
            alns_cost = alns_cost_raw * L
            alns_gap_best = ((alns_cost - best_known) /
                             best_known * 100.0) if best_known else float('nan')
            alns_gap_paper = ((alns_cost - paper_cost) /
                              paper_cost * 100.0) if paper_cost else float('nan')

            elapsed = time.time() - start_time
            logger.info(
                "Instance %03d | Initial: %.4f | Pure ALNS: %.4f | Time: %.2fs",
                i + 1, init_cost, alns_cost, elapsed,
            )

            rows.append({
                "instance": instance_id,
                "file_name": pt_file.name,
                "initial_cost": init_cost,
                "alns_cost": alns_cost,
                "best_known": best_known,
                "paper_cost": paper_cost,
                "alns_gap_best": alns_gap_best,
                "alns_gap_paper": alns_gap_paper,
                "time": elapsed,
                "best_iters": best_alns_iters
            })
        except Exception:
            elapsed = time.time() - start_time
            logger.exception(
                "Instance %03d failed after %.2fs", i + 1, elapsed)
            rows.append({
                "instance": instance_id,
                "file_name": pt_file.name,
                "initial_cost": float("nan"),
                "alns_cost": float("nan"),
                "best_known": best_known,
                "paper_cost": paper_cost,
                "alns_gap_best": float("nan"),
                "alns_gap_paper": float("nan"),
                "time": elapsed,
                "best_iters": 0
            })

    if shared_executor is not None:
        shared_executor.shutdown(wait=True, cancel_futures=True)

    logger.info("")
    log_instance_table(rows, logger)
    log_summary(rows, logger)

    return rows


def export_best_routes_csv(instance_id, method_type, run_idx, best_state, log_dir):
    """Saves the exact route structures of the best found state to verify feasibility."""
    csv_path = Path(log_dir) / "best_routes.csv"
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Instance", "Method", "Run", "Total_Cost", "Route_Idx",
                "Vehicle_Type", "Load", "Capacity", "Overload_Amount", "Route_Cost", "Path"
            ])

        if best_state is None or not hasattr(best_state, 'routes'):
            return

        dm = best_state.data['dist_matrix']
        total_cost = float(best_state.objective())

        for ri, (path, vt, load) in enumerate(best_state.routes):
            capacity = best_state.data['capacity'][vt]
            overload = max(0.0, load - capacity)

            rd = sum(dm[path[i], path[i + 1]] for i in range(len(path) - 1))
            route_cost = best_state.data['fixed_cost'][vt] + \
                rd * best_state.data['var_cost'][vt]

            path_str = " -> ".join(map(str, path))

            writer.writerow([
                instance_id, method_type, run_idx, f"{total_cost:.4f}", ri,
                vt, f"{load:.6f}", f"{capacity:.6f}", f"{overload:.6f}", f"{route_cost:.4f}", path_str
            ])


def main():
    args = parse_args()

    dataset, selected_methods, selected_instances, num_to_test, num_runs = interactive_menu()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_annotation = "GoldenBC" if dataset == "GoldenBC" else "TestTensors"
    method_annotation = "Mixed" if len(
        selected_methods) > 1 else selected_methods[0]

    log_base_dir = Path(
        f"./logs/run_{timestamp}_{dataset_annotation}_{method_annotation}")
    log_base_dir.mkdir(parents=True, exist_ok=True)

    console_log_path = log_base_dir / f"console_output_{timestamp}.txt"
    sys.stdout = TeeLogger(str(console_log_path))

    logger, log_path = setup_logger(
        str(log_base_dir), args.log_level, timestamp)

    print(f"\n[SYSTEM] Logs will be saved to: {log_base_dir}")
    print(
        f"[SYSTEM] Console output is being duplicated to: {console_log_path}\n")

    class DummyConfigState:
        def __init__(self):

            self.data = {"demand": [0.0] * 100}

        def copy(self): return self

        def objective(self): return 1000.0

    config = run_pure_alns(DummyConfigState(), get_config=True)

    logger.info("==================================================")
    logger.info("           PURE ALNS CONFIGURATION                ")
    logger.info("==================================================")
    logger.info(f"Total Iterations : {config['iterations']}")
    logger.info(f"Stagnation Pat.  : {config['stagnation_patience']}")
    logger.info(f"Min Iterations   : {config['min_iterations']}")
    logger.info(f"Start Temp       : {config['start_temp']:.4f}")
    logger.info(f"End Temp         : {config['end_temp']:.4f}")
    logger.info(f"Cooling Step     : {config['cooling_step']:.6f}")
    logger.info("Selector         : SegmentedRouletteWheel")
    logger.info(f"  Scores         : {config['scores']}")
    logger.info(f"  Decay          : {config['decay']}")
    logger.info(f"  Seg Length     : {config['seg_length']}")
    logger.info("Destroy Operators:")
    for op in config['destroy_operators']:
        logger.info(f"  - {op}")
    logger.info("Repair Operators:")
    for op in config['repair_operators']:
        logger.info(f"  - {op}")
    logger.info("==================================================\n")

    all_runs_data = []
    global_start_time = time.time()

    for run_idx in range(num_runs):
        logger.info(
            f"\n{'=' * 60}\n STARTING RUN {run_idx + 1} OF {num_runs}\n{'=' * 60}")
        master_rows = []

        if dataset == "GoldenBC":
            for method in selected_methods:
                for instance_id in selected_instances:
                    pt_file = Path(
                        f"Benchmarks/GoldenBC/pt/variants/goldenbc_problem_{instance_id}__{method}.pt")
                    raw_pt = Path(
                        f"Benchmarks/GoldenBC/pt/by_problem/goldenbc_problem_{instance_id}.pt")

                    if not pt_file.exists() or not raw_pt.exists():
                        logger.error(f"File not found: {pt_file} or {raw_pt}")
                        continue

                    try:

                        file_rows = process_benchmark_file(pt_file, raw_pt, method, instance_id, args, logger, run_idx,
                                                           dataset, num_to_test)
                        master_rows.extend(file_rows)
                    except Exception as e:
                        logger.exception(
                            "Catastrophic failure processing file %s", pt_file.name)
        else:

            for size in selected_instances:
                pt_file = Path(f"data/test_tensor({size})_6_100_5678.pt")
                raw_pt = pt_file
                method = "None"
                instance_id = str(size)

                if not pt_file.exists():
                    logger.error(f"File not found: {pt_file}")
                    continue

                try:

                    file_rows = process_benchmark_file(pt_file, raw_pt, method, instance_id, args, logger, run_idx,
                                                       dataset, num_to_test)
                    master_rows.extend(file_rows)
                except Exception as e:
                    logger.exception(
                        "Catastrophic failure processing file %s", pt_file.name)

        all_runs_data.append(master_rows)

        if master_rows:
            logger.info("")
            logger.info("##################################################")
            logger.info(f"          SUMMARY FOR RUN {run_idx + 1}         ")
            logger.info("##################################################")
            log_summary(master_rows, logger)

    logger.info("\n==================================================")
    logger.info("All benchmark files and runs processed successfully.")

    if num_runs > 1 and all_runs_data:
        logger.info("\n" + "#" * 120)
        logger.info(
            "                         MULTI-RUN PURE ALNS SUMMARY (MEAN +- STD DEV)")
        logger.info("#" * 120)

        summary_map = {}
        for run_rows in all_runs_data:
            for row in run_rows:
                key = (row["instance"], row["file_name"])
                if key not in summary_map:
                    summary_map[key] = {
                        "alns": [],
                        "time": [],
                        "best_known": row.get("best_known"),
                        "paper_cost": row.get("paper_cost")
                    }

                if not math.isnan(row.get("alns_cost", float('nan'))):
                    summary_map[key]["alns"].append(row["alns_cost"])
                if not math.isnan(row.get("time", float('nan'))):
                    summary_map[key]["time"].append(row["time"])

        header = f"{'Inst':>4} | {'Method File':>35} | {'Pure ALNS (Mean +- SD)':>23} | {'Best ALNS':>11} | {'Avg Gap(B)':>10} | {'Avg Gap(P)':>10} | {'Time(s)':>18}"
        logger.info(header)
        logger.info("-" * len(header))

        for (inst, fname), metrics in summary_map.items():
            alns_mean = np.mean(
                metrics["alns"]) if metrics["alns"] else float('nan')
            alns_sd = np.std(metrics["alns"]) if metrics["alns"] else 0.0
            alns_best = np.min(
                metrics["alns"]) if metrics["alns"] else float('nan')
            time_mean = np.mean(
                metrics["time"]) if metrics["time"] else float('nan')
            time_sd = np.std(metrics["time"]) if metrics["time"] else 0.0

            bk = metrics["best_known"]
            pc = metrics["paper_cost"]
            overall_gap_best = ((alns_mean - bk) / bk) * \
                100.0 if not math.isnan(alns_mean) and bk else float('nan')
            overall_gap_paper = ((alns_mean - pc) / pc) * \
                100.0 if not math.isnan(alns_mean) and pc else float('nan')

            str_alns = f"{alns_mean:>9.2f} +- {alns_sd:<6.2f}" if not math.isnan(
                alns_mean) else f"{'nan':>23}"
            str_best = f"{alns_best:>11.2f}" if not math.isnan(
                alns_best) else f"{'nan':>11}"
            str_gap_b = f"{overall_gap_best:>9.2f}%" if not math.isnan(
                overall_gap_best) else f"{'nan':>10}"
            str_gap_p = f"{overall_gap_paper:>9.2f}%" if not math.isnan(
                overall_gap_paper) else f"{'nan':>10}"
            str_time = f"{time_mean:>8.2f} +- {time_sd:<5.2f}" if not math.isnan(
                time_mean) else f"{'nan':>18}"

            logger.info(
                "%s",
                f"{inst:>4} | {fname:>35} | {str_alns} | {str_best} | {str_gap_b} | {str_gap_p} | {str_time}"
            )

        logger.info("-" * len(header))
        run_averages = []
        for r_idx, run_rows in enumerate(all_runs_data):
            valid_costs = [r["alns_cost"] for r in run_rows if not math.isnan(
                r.get("alns_cost", float('nan')))]
            if valid_costs:
                run_averages.append((r_idx + 1, np.mean(valid_costs)))

        if run_averages:
            best_run_idx, best_run_avg = min(run_averages, key=lambda x: x[1])
            logger.info("")
            logger.info(
                f">>> BEST OVERALL RUN: Run {best_run_idx} achieved the lowest average pure ALNS cost ({best_run_avg:.2f}) <<<")
    global_elapsed = time.time() - global_start_time
    logger.info("\n" + "=" * 50)
    logger.info(
        f"TOTAL WALL-CLOCK TIME (ALL RUNS): {global_elapsed:.2f} seconds")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

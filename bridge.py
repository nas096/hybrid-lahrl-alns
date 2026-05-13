import math
import sys
from logging import Logger
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from alns_helpers import FSMVRP_State


def lahrl_solution_to_alns_state(
    extracted_solution: dict,
    data_instance: dict,
    logger: Logger | None = None,
    capacity_tolerance: float = 1e-5,
) -> FSMVRP_State:
    routes = []
    covered_customers = set()
    capacities = data_instance["capacity"]

    for route_index, route in enumerate(extracted_solution.get("routes", []), start=1):
        served_nodes = [int(node) for node in route.get("served_nodes", [])]
        vehicle_type = int(route["vehicle_type"])
        load = float(route.get("delivered_demand", 0.0))

        if vehicle_type < 0 or vehicle_type >= len(capacities):
            raise ValueError(
                f"Route {route_index} uses invalid vehicle_type={vehicle_type}; "
                f"valid range is [0, {len(capacities) - 1}]."
            )

        capacity = float(capacities[vehicle_type])
        overload = load - capacity
        if overload > capacity_tolerance:
            raise ValueError(
                f"Route {route_index} load {load:.6f} exceeds capacity {capacity:.6f} "
                f"for vehicle_type={vehicle_type}."
            )
        if overload > 0 and logger is not None:
            logger.warning(
                "Accepted small LA-HRL overload on route %d: vehicle_type=%d load=%.6f capacity=%.6f delta=%.8f",
                route_index,
                vehicle_type,
                load,
                capacity,
                overload,
            )

        path = [0] + served_nodes + [0]
        routes.append([path, vehicle_type, load])
        covered_customers.update(served_nodes)

    expected_customers = set(range(1, len(data_instance["demand"])))
    unassigned = expected_customers - covered_customers

    state = FSMVRP_State(routes, unassigned, data_instance)
    objective_value = float(state.objective())
    assert math.isfinite(objective_value), "Constructed ALNS state has a non-finite objective."
    return state

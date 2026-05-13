from dataclasses import dataclass
from typing import Optional
import torch

from problemdef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:

    depot_xy: Optional[torch.Tensor] = None
    node_xy: Optional[torch.Tensor] = None
    node_demand: Optional[torch.Tensor] = None
    agent_capacity: Optional[torch.Tensor] = None
    agent_fixed_cost: Optional[torch.Tensor] = None
    agent_variable_cost: Optional[torch.Tensor] = None


@dataclass
class Step_State:



    BATCH_IDX: Optional[torch.Tensor] = None
    POMO_IDX: Optional[torch.Tensor] = None

    selected_count: int = 0
    route_count: Optional[torch.Tensor] = None
    graph_size: int = 0

    current_vehicle_type: Optional[torch.Tensor] = None

    fleet_mask: Optional[torch.Tensor] = None

    need_fleet_action: Optional[torch.Tensor] = None

    current_node: Optional[torch.Tensor] = None
    current_load: Optional[torch.Tensor] = None
    visited_mask: Optional[torch.Tensor] = None
    ninf_mask: Optional[torch.Tensor] = None

    accumulated_cost: Optional[torch.Tensor] = None

    finished: Optional[torch.Tensor] = None


class FSMVRPSMDPEnv:



    def __init__(self, **env_params):

        self.env_params = env_params
        self.device = env_params.get('device', None)
        self.min_problem_size = env_params['min_problem_size']
        self.max_problem_size = env_params['max_problem_size']
        self.min_agent_num = env_params['min_agent_num']
        self.max_agent_num = env_params['max_agent_num']
        self.pomo_size = env_params['pomo_size']
        penalty_cfg = env_params.get('utilization_penalty', {})
        self.util_penalty_enable = penalty_cfg.get('enable', False)
        self.util_penalty_ratio_threshold = penalty_cfg.get(
            'ratio_threshold', 0.8)
        self.util_penalty_weight = penalty_cfg.get('weight', 0.0)
        self.util_penalty_power = penalty_cfg.get('power', 2.0)
        self.util_penalty_min_demand = penalty_cfg.get('min_demand', 0.0)

        self.FLAG__use_saved_problems = False
        self.FLAG__use_random_seed = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_agent_capacity = None
        self.saved_agent_fixed_cost = None
        self.saved_agent_variable_cost = None
        self.saved_index = None

        self.batch_size = None
        self.problem_size = None
        self.agent_num = None

        self.depot_node_xy = None
        self.depot_node_demand = None

        self.agent_capacity = None
        self.agent_fixed_cost = None
        self.agent_variable_cost = None

        self.BATCH_IDX = None
        self.POMO_IDX = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

        self.selected_count = 0
        self.route_count = None
        self.current_node = None
        self.current_vehicle_type = None
        self.current_load = None
        self.need_fleet_action = None
        self.visited_mask = None
        self.accumulated_cost = None
        self.raw_accumulated_cost = None
        self.finished = None
        self.current_route_capacity = None
        self.current_route_demand = None

        self._step_reward = None

    def use_saved_problems(self, filename: str, device):
        self.FLAG__use_saved_problems = True
        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_agent_capacity = loaded_dict['agent_capacity']
        self.saved_agent_fixed_cost = loaded_dict['agent_fixed_cost']
        self.saved_agent_variable_cost = loaded_dict['agent_variable_cost']
        self.saved_index = 0

    def set_random_seed(self, random_seed: int, test_num: int):
        self.FLAG__use_random_seed = True
        torch.manual_seed(random_seed)
        self.random_list = torch.randint(0, 100_000, size=(test_num,))
        self.random_list_index = 0
        torch.seed()

    def load_problems(self, batch_size: int, aug_factor: int = 1):

        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems and not self.FLAG__use_random_seed:
            depot_xy, node_xy, node_demand, agent_capacity, agent_fixed_cost, agent_variable_cost =\
                get_random_problems(batch_size,
                                    self.min_problem_size, self.max_problem_size,
                                    self.min_agent_num, self.max_agent_num)
        elif self.FLAG__use_random_seed:
            seed = self.random_list[self.random_list_index].item()
            self.random_list_index += 1
            depot_xy, node_xy, node_demand, agent_capacity, agent_fixed_cost, agent_variable_cost =\
                get_random_problems(batch_size,
                                    self.min_problem_size, self.max_problem_size,
                                    self.min_agent_num, self.max_agent_num,
                                    random_seed=seed)
        else:
            s, e = self.saved_index, self.saved_index + batch_size
            depot_xy = self.saved_depot_xy[s:e]
            node_xy = self.saved_node_xy[s:e]
            node_demand = self.saved_node_demand[s:e]
            agent_capacity = self.saved_agent_capacity[s:e]
            agent_fixed_cost = self.saved_agent_fixed_cost[s:e]
            agent_variable_cost = self.saved_agent_variable_cost[s:e]
            self.saved_index += batch_size

        if self.device is not None:
            depot_xy = depot_xy.to(self.device)
            node_xy = node_xy.to(self.device)
            node_demand = node_demand.to(self.device)
            agent_capacity = agent_capacity.to(self.device)
            agent_fixed_cost = agent_fixed_cost.to(self.device)
            agent_variable_cost = agent_variable_cost.to(self.device)

        if aug_factor > 1:
            if aug_factor == 8:
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                agent_capacity = agent_capacity.repeat(8, 1)
                agent_fixed_cost = agent_fixed_cost.repeat(8, 1)
                agent_variable_cost = agent_variable_cost.repeat(8, 1)
            else:
                raise NotImplementedError("Chỉ hỗ trợ aug_factor = 1 hoặc 8.")

        self.batch_size = depot_xy.size(0)
        self.problem_size = node_xy.size(1)
        self.agent_num = agent_capacity.size(1)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)

        depot_demand = torch.zeros(
            self.batch_size, 1, device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.agent_capacity = agent_capacity
        self.agent_fixed_cost = agent_fixed_cost
        self.agent_variable_cost = agent_variable_cost

        self.BATCH_IDX = torch.arange(self.batch_size, device=depot_xy.device)[
            :, None].expand(self.batch_size, self.pomo_size)

        self.POMO_IDX = torch.arange(self.pomo_size, device=depot_xy.device)[
            None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.agent_capacity = agent_capacity
        self.reset_state.agent_fixed_cost = agent_fixed_cost
        self.reset_state.agent_variable_cost = agent_variable_cost

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.graph_size = self.problem_size + 1

        return self.reset_state, None, False

    def restore_problem(self, saved_problem: dict):


        depot_xy = saved_problem['depot_xy']
        node_xy = saved_problem['node_xy']
        node_demand = saved_problem['node_demand']
        agent_capacity = saved_problem['agent_capacity']
        agent_fixed_cost = saved_problem['agent_fixed_cost']
        agent_variable_cost = saved_problem['agent_variable_cost']

        self.batch_size = depot_xy.size(0)
        self.problem_size = node_xy.size(1)
        self.agent_num = agent_capacity.size(1)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)

        depot_demand = torch.zeros(self.batch_size, 1, device=depot_xy.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.agent_capacity = agent_capacity
        self.agent_fixed_cost = agent_fixed_cost
        self.agent_variable_cost = agent_variable_cost

        self.BATCH_IDX = torch.arange(self.batch_size, device=depot_xy.device)[
            :, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=depot_xy.device)[
            None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.agent_capacity = agent_capacity
        self.reset_state.agent_fixed_cost = agent_fixed_cost
        self.reset_state.agent_variable_cost = agent_variable_cost

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.graph_size = self.problem_size + 1

    def reset(self):

           
        B, P = self.batch_size, self.pomo_size
        N1 = self.problem_size + 1

        dev = self.depot_node_xy.device

        self.selected_count = 0
        self.route_count = torch.zeros(B, P, dtype=torch.long, device=dev)

        self.current_node = torch.zeros(B, P, dtype=torch.long, device=dev)

        self.current_vehicle_type = torch.full(
            (B, P), -1, dtype=torch.long, device=dev)

        self.current_load = torch.zeros(B, P, device=dev)

        self.need_fleet_action = torch.ones(B, P, dtype=torch.bool, device=dev)

        self.visited_mask = torch.zeros(B, P, N1, dtype=torch.bool, device=dev)

        self.visited_mask[:, :, 0] = True

        self.accumulated_cost = torch.zeros(B, P, device=dev)

        self.raw_accumulated_cost = torch.zeros(B, P, device=dev)

        self.finished = torch.zeros(B, P, dtype=torch.bool, device=dev)

        self.current_route_capacity = torch.zeros(B, P, device=dev)
        self.current_route_demand = torch.zeros(B, P, device=dev)

        self._step_reward = torch.zeros(B, P, device=dev)

        self._sync_step_state()
        return self.step_state, None, False

    def pre_step(self):
        self._sync_step_state()
        return self.step_state, None, False

    def fleet_step(self, vehicle_selected: torch.Tensor):
           
        assert self.need_fleet_action.any(),\
            "fleet_step được gọi nhưng không có pomo nào đang chờ fleet action."

        B, P = self.batch_size, self.pomo_size
        mask = self.need_fleet_action

        self.current_vehicle_type = torch.where(
            mask, vehicle_selected, self.current_vehicle_type)

        cap_all = self.agent_capacity

        new_load = cap_all[
            self.BATCH_IDX,
            vehicle_selected.clamp(min=0)
        ]

        self.current_load = torch.where(mask, new_load, self.current_load)
        self.current_route_capacity = torch.where(
            mask, new_load, self.current_route_capacity)
        self.current_route_demand = torch.where(mask, torch.zeros_like(
            self.current_route_demand), self.current_route_demand)

        fixed_cost = self.agent_fixed_cost[
            self.BATCH_IDX,
            vehicle_selected.clamp(min=0)
        ]

        step_reward = -fixed_cost * mask.float()

        self.accumulated_cost -= step_reward
        self.raw_accumulated_cost -= step_reward
        self._step_reward = step_reward

        self.need_fleet_action = torch.zeros(
            B, P, dtype=torch.bool, device=self.depot_node_xy.device)

        self.route_count += mask.long()

        self._sync_step_state()
        done = self.finished.all().item()
        return self.step_state, step_reward if not done else self._get_episode_reward(), done

    def route_step(self, node_selected: torch.Tensor):

           
        assert (~self.need_fleet_action).any(
        ), "route_step được gọi nhưng tất cả POMO đều đang chờ fleet_step!"
        B, P = self.batch_size, self.pomo_size
        N1 = self.problem_size + 1

        all_xy = self.depot_node_xy

        last_xy = all_xy[
            self.BATCH_IDX,
            self.current_node
        ]

        next_xy = all_xy[
            self.BATCH_IDX,
            node_selected
        ]

        travel_cost = ((next_xy - last_xy) ** 2).sum(-1).sqrt()

        var_cost = self.agent_variable_cost[
            self.BATCH_IDX,
            self.current_vehicle_type.clamp(min=0)
        ]

        c_ij = travel_cost * var_cost

        step_reward = -c_ij * (~self.finished).float()
        self._step_reward = step_reward
        self.accumulated_cost += c_ij * (~self.finished).float()
        self.raw_accumulated_cost += c_ij * (~self.finished).float()

        is_customer = (node_selected != 0)

        self.visited_mask[
            self.BATCH_IDX[is_customer],
            self.POMO_IDX[is_customer],
            node_selected[is_customer]
        ] = True

        demand_all = self.depot_node_demand
        selected_demand = demand_all[self.BATCH_IDX, node_selected]

        self.current_load = self.current_load - selected_demand
        self.current_route_demand = self.current_route_demand +\
            selected_demand * is_customer.float()

        self.current_load = self.current_load.clamp(min=0)

        self.current_node = node_selected
        self.selected_count += 1

        returning_to_depot = (node_selected == 0)
        route_penalty = self._compute_utilization_penalty(
            returning_to_depot & (~self.finished))
        if route_penalty is not None:
            step_reward = step_reward - route_penalty
            self._step_reward = step_reward
            self.accumulated_cost += route_penalty

        all_visited = self.visited_mask[:, :, 1:].all(dim=-1)

        new_finished = returning_to_depot & all_visited & (~self.finished)
        self.finished = self.finished | new_finished

        self.need_fleet_action = returning_to_depot & (~self.finished)
        self.current_route_capacity = torch.where(returning_to_depot, torch.zeros_like(
            self.current_route_capacity), self.current_route_capacity)
        self.current_route_demand = torch.where(returning_to_depot, torch.zeros_like(
            self.current_route_demand), self.current_route_demand)

        self._sync_step_state()

        done = self.finished.all().item()
        if done:
            final_reward = self._get_episode_reward()
            return self.step_state, final_reward, True

        return self.step_state, step_reward, False

    def _build_ninf_mask(self) -> torch.Tensor:

           
        B, P = self.batch_size, self.pomo_size
        N1 = self.problem_size + 1

        ninf_mask = torch.zeros(B, P, N1, device=self.depot_node_xy.device)

        ninf_mask[self.visited_mask] = float('-inf')

        ninf_mask[:, :, 0] = 0.0

        just_started = (self.current_node == 0) & (~self.finished)
        ninf_mask[self.BATCH_IDX[just_started],
                  self.POMO_IDX[just_started], 0] = float('-inf')

        demand_all = self.depot_node_demand[:, None, :].expand(B, P, N1)
        load_expand = self.current_load[:, :, None].expand(B, P, N1)
        round_eps = 1e-5
        exceeds_capacity = (demand_all > load_expand + round_eps)
        ninf_mask[exceeds_capacity] = float('-inf')

        ninf_mask[:, :, 0] = torch.where(just_started, float('-inf'), 0.0)

        need_fleet_expand = self.need_fleet_action[:, :, None].expand(B, P, N1)
        ninf_mask[need_fleet_expand] = float('-inf')

        return ninf_mask

    def _build_fleet_mask(self) -> torch.Tensor:
       
        B, P = self.batch_size, self.pomo_size
        A = self.agent_num

        customer_visited = self.visited_mask[:, :, 1:]
        customer_demand = self.depot_node_demand[:, 1:]

        cap = self.agent_capacity

        demand_expand = customer_demand[:, None, :].expand(
            B, P, self.problem_size)

        masked_demand = demand_expand.clone()
        masked_demand[customer_visited] = float('inf')
        min_demand, _ = masked_demand.min(dim=-1)

        cap_expand = cap[:, None, :].expand(B, P, A)
        min_demand_expand = min_demand[:, :, None].expand(B, P, A)

        fleet_valid = (cap_expand >= min_demand_expand)

        fleet_mask = torch.zeros(B, P, A, device=self.depot_node_xy.device)
        fleet_mask[~fleet_valid] = float('-inf')

        not_needed = ~self.need_fleet_action
        not_needed_3d = not_needed[:, :, None].expand(B, P, A)

        fleet_mask = torch.where(not_needed_3d, torch.tensor(
            float('-inf'), device=fleet_mask.device), fleet_mask)

        dummy_mask = torch.zeros(
            B, P, A, dtype=torch.bool, device=fleet_mask.device)
        dummy_mask[:, :, 0] = True
        restore = not_needed_3d & dummy_mask
        fleet_mask = torch.where(restore, torch.tensor(
            0.0, device=fleet_mask.device), fleet_mask)

        return fleet_mask

    def _sync_step_state(self):
                                                            
        self.step_state.selected_count = self.selected_count
        self.step_state.route_count = self.route_count
        self.step_state.current_vehicle_type = self.current_vehicle_type
        self.step_state.current_node = self.current_node
        self.step_state.current_load = self.current_load
        self.step_state.need_fleet_action = self.need_fleet_action
        self.step_state.visited_mask = self.visited_mask
        self.step_state.ninf_mask = self._build_ninf_mask()
        self.step_state.fleet_mask = self._build_fleet_mask()
        self.step_state.accumulated_cost = self.accumulated_cost
        self.step_state.finished = self.finished

    def _compute_utilization_penalty(self, route_finished_mask: torch.Tensor):
        if (not self.util_penalty_enable) or self.util_penalty_weight <= 0:
            return None

        active_mask = route_finished_mask & (
            self.current_route_capacity > 1e-8)
        active_mask = active_mask & (
            self.current_route_demand > self.util_penalty_min_demand)
        if not active_mask.any():
            return None

        route_ratio = self.current_route_demand /\
            self.current_route_capacity.clamp(min=1e-8)
        deficit = (self.util_penalty_ratio_threshold -
                   route_ratio).clamp(min=0.0)
        if self.util_penalty_power != 1.0:
            deficit = deficit.pow(self.util_penalty_power)

        current_fixed_cost = self.agent_fixed_cost[
            self.BATCH_IDX,
            self.current_vehicle_type.clamp(min=0)
        ]
        penalty = self.util_penalty_weight * current_fixed_cost * deficit
        return penalty * active_mask.float()

    def _get_episode_reward(self) -> torch.Tensor:

           
        return -self.accumulated_cost

    def get_total_cost(self) -> torch.Tensor:
                                                         
        return self.accumulated_cost.clone()

    def get_raw_total_cost(self) -> torch.Tensor:
        return self.raw_accumulated_cost.clone()


if __name__ == '__main__':
    print("=== FSMVRPSMDPEnv – Sanity Check ===\n")

    env_params = {
        'min_problem_size': 5,
        'max_problem_size': 10,
        'min_agent_num': 2,
        'max_agent_num': 4,
        'pomo_size': 3,
    }

    env = FSMVRPSMDPEnv(**env_params)
    reset_state, _, _ = env.load_problems(batch_size=2)
    _, _, _ = env.reset()
    step_state, _, _ = env.pre_step()

    print(f"Batch size    : {env.batch_size}")
    print(f"Problem size  : {env.problem_size}")
    print(f"Agent num     : {env.agent_num}")
    print(f"POMO size     : {env.pomo_size}")
    print(f"depot_node_xy : {env.depot_node_xy.shape}")
    print(f"need_fleet_action (init): {step_state.need_fleet_action}")
    print()

    step = 0
    done = False
    while not done:
        if step_state.need_fleet_action.any():

            B, P, A = env.batch_size, env.pomo_size, env.agent_num
            fleet_mask = step_state.fleet_mask

            valid = (fleet_mask > float('-inf'))

            logits = fleet_mask.clone()
            logits[~valid] = -1e9
            vehicle_sel = logits.argmax(dim=-1)
            step_state, reward, done = env.fleet_step(vehicle_sel)
            print(
                f"[Step {step:3d}] FLEET action → xe {vehicle_sel[0].tolist()} | reward={reward[0].tolist() if reward is not None else None}")
        else:

            ninf_mask = step_state.ninf_mask
            logits = torch.zeros_like(ninf_mask) + ninf_mask

            rand_scores = torch.rand_like(logits)
            rand_scores[ninf_mask == float('-inf')] = -1e9
            node_sel = rand_scores.argmax(dim=-1)
            step_state, reward, done = env.route_step(node_sel)
            print(f"[Step {step:3d}] ROUTE action → node {node_sel[0].tolist()} | reward={reward[0].tolist() if reward is not None else None} | done={done}")
        step += 1
        if step > 500:
            print("Vượt quá bước tối đa, dừng.")
            break

    print(f"\nTổng chi phí (batch 0): {env.get_total_cost()[0].tolist()}")
    print("=== Hoàn thành ===")

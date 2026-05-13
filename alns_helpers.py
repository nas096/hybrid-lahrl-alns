import numpy as np

class FSMVRP_State:
    def __init__(self, routes, unassigned, data_instance):
        self.routes = routes
        self.unassigned = set(unassigned)
        self.data = data_instance

    def copy(self):
        new_routes = [[path[:], v_type, load] for path, v_type, load in self.routes]
        return FSMVRP_State(new_routes, set(self.unassigned), self.data)

    def objective(self):
        total = 0.0
        for path, v_type, _ in self.routes:
            total += self.data['fixed_cost'][v_type]
            vc = self.data['var_cost'][v_type]
            dm = self.data['dist_matrix']
            for i in range(len(path) - 1):
                total += dm[path[i], path[i+1]] * vc
        total += len(self.unassigned) * 1e9
        return total

def _route_dist(path, dm):
    if len(path) < 2:
        return 0.0
    idx = np.asarray(path)
    return float(dm[idx[:-1], idx[1:]].sum())

def _best_vehicle(load, route_dist, data):
    best_v, best_c = -1, float('inf')
    for vt in range(len(data['capacity'])):
        if data['capacity'][vt] >= load:
            c = data['fixed_cost'][vt] + route_dist * data['var_cost'][vt]
            if c < best_c:
                best_c, best_v = c, vt
    return best_v, best_c

def _vehicle_swap(state):
    dm = state.data['dist_matrix']
    for r in state.routes:
        path, _, load = r
        rd = _route_dist(path, dm)
        vt, _ = _best_vehicle(load, rd, state.data)
        r[1] = vt
    return state

def _two_opt(path, dm):
    best = path[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                d_old = dm[best[i-1], best[i]] + dm[best[j], best[j+1]]
                d_new = dm[best[i-1], best[j]] + dm[best[i], best[j+1]]
                if d_new < d_old - 0.01:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
    return best

def _or_opt(path, dm):
    if path is None or len(path) < 4:
        return path
    best = path[:]
    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2]:
            if improved:
                break
            for i in range(1, len(best) - seg_len):
                if i + seg_len >= len(best):
                    continue
                a = best[i-1]
                b = best[i]
                c = best[i+seg_len-1]
                d = best[i+seg_len]
                remove_gain = dm[a,b] + dm[c,d] - dm[a,d]
                seg = best[i:i+seg_len]
                without = best[:i] + best[i+seg_len:]
                for j in range(1, len(without)):
                    u, v = without[j-1], without[j]
                    insert_cost = dm[u,seg[0]] + dm[seg[-1],v] - dm[u,v]
                    if remove_gain - insert_cost > 0.01:
                        best = without[:j] + seg + without[j:]
                        improved = True
                        break
                if improved:
                    break
    return best


def _insert_all_customers(new_state, customers, rng):
    dm = new_state.data['dist_matrix']
    for cust in customers:
        cd = new_state.data['demand'][cust]
        best_diff = float('inf')
        best_ins = None

        for ri, (path, vt, load) in enumerate(new_state.routes):
            new_load = load + cd
            new_vt, _ = _best_vehicle(new_load, 0, new_state.data)
            if new_vt == -1:
                continue

            old_rd = _route_dist(path, dm)
            old_vt2, old_cost = _best_vehicle(load, old_rd, new_state.data)

            for pos in range(1, len(path)):
                u, v = path[pos-1], path[pos]
                extra = dm[u, cust] + dm[cust, v] - dm[u, v]
                new_rd = old_rd + extra
                new_vt2, new_cost = _best_vehicle(new_load, new_rd, new_state.data)
                if new_vt2 == -1:
                    continue
                diff = new_cost - old_cost
                if diff < best_diff:
                    best_diff = diff
                    best_ins = (ri, pos, None, new_vt2)

        for vt in range(len(new_state.data['capacity'])):
            if cd <= new_state.data['capacity'][vt]:
                rd = 2 * dm[0, cust]
                c = new_state.data['fixed_cost'][vt] + rd * new_state.data['var_cost'][vt]
                if c < best_diff:
                    best_diff = c
                    best_ins = (None, None, vt, vt)

        if best_ins is not None:
            ri, pos, new_route_vt, chosen_vt = best_ins
            if new_route_vt is not None:
                new_state.routes.append([[0, cust, 0], new_route_vt, cd])
            else:
                new_state.routes[ri][0].insert(pos, cust)
                new_state.routes[ri][2] += cd
                new_state.routes[ri][1] = chosen_vt
            new_state.unassigned.discard(cust)
    return new_state


def greedy_repair(state, rng, **kwargs):
    new_state = state.copy()
    customers = list(new_state.unassigned)
    rng.shuffle(customers)
    new_state = _insert_all_customers(new_state, customers, rng)
    return _vehicle_swap(new_state)
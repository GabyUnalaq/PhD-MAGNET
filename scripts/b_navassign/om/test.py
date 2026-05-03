import numpy as np
from .omotc import heuristic

if __name__ == "__main__":
    # 480
    workSpace = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1],
    [1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1],
    [1,0,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1],
    [1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1],
    [1,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1],
    [1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1],
    [1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ], dtype=int)
    all_start_loc = {'r0': (14, 11), 'r1': (13, 18), 'r2': (6, 11), 'r3': (3, 7)}
    all_goal_loc = {'g0': (16, 1), 'g1': (5, 15), 'g2': (12, 17),  'g3': (18, 14)}

    no_of_robots = len(all_start_loc)
    no_of_goals  = len(all_goal_loc)

    (result, om_cost, omotc_cost, path,
     u_count, nexp_makespan, no_of_exp,
     makespan, heur_CL, makespan_CL,
     t_om, t_otc) = heuristic(
        no_of_robots, no_of_goals,
        workSpace, all_start_loc, all_goal_loc,
        ws_type='2D'
    )

    if result is None:
        print("  Result : None (trapped robot or goal — no feasible assignment)")
    else:
        print(f"  Assignment : {result}")
        print(f"  Makespan   : {makespan}")
        print(f"  OM cost    : {om_cost}  (sum of assigned path lengths)")
        print(f"  Unassigned : {u_count}")
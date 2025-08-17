import re
import random
import inflect
from collections import deque
from typing import Dict, List, Union
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv

from mindgrid.infrastructure.env_constants import (
    DIR_TO_VEC,
    IDX_TO_COLOR,
    IDX_TO_DIR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)


def are_objects_equal(o, oo):
    if o is None and oo is None:
        return True
    if o is None or oo is None:
        # print("none")
        return False
    if type(o) != type(oo):
        # print("type")
        return False
    if o.color != oo.color:
        # print("color")
        return False
    if not are_objects_equal(o.contains, oo.contains):
        # print("contains")
        return False
    if o.init_pos != oo.init_pos:
        # print("init_pos")
        return False
    if o.cur_pos != oo.cur_pos:
        # print("cur_pos", o.cur_pos, oo.cur_pos)
        return False
    return True


def bfs(grid, start_dir, start_pos, end_pos):
    state = (start_pos, start_dir)
    queue = deque([state])
    trace_back = {}
    trace_back[state] = -1

    while queue:
        state = queue.popleft()
        (x, y), dir = state

        if (x, y) in end_pos:
            actions = []
            while trace_back[state] != -1:
                state, action = trace_back[state]
                actions.append(action)
            return list(reversed(actions))

        # forward
        dir_vec = DIR_TO_VEC[dir]
        nx, ny = x + dir_vec[0], y + dir_vec[1]
        nstate = ((nx, ny), dir)
        if grid[nx, ny] == 0 and nstate not in trace_back:
            queue.append(nstate)
            trace_back[nstate] = (state, Actions.forward)

        # rotate
        for d in [-1, 1]:
            ndir = (dir + d + 4) % 4
            nstate = ((x, y), ndir)
            if nstate not in trace_back:
                queue.append(nstate)
                trace_back[nstate] = (state, Actions.left if d == -1 else Actions.right)

    return None


def relative_position(dir, point):
    dx, dy = DIR_TO_VEC[dir]
    x, y = point

    # Determine the direction based on the vector
    if dx == 0 and dy == -1:
        cardinal = "north"
        front_back = "in front" if y < 0 else "behind"
        left_right = "to the right" if x > 0 else "to the left"
        return left_right, front_back
    elif dx == 0 and dy == 1:
        cardinal = "south"
        front_back = "in front" if y > 0 else "behind"
        left_right = "to the right" if x < 0 else "to the left"
        return left_right, front_back
    elif dx == -1 and dy == 0:
        cardinal = "west"
        front_back = "in front" if x < 0 else "behind"
        left_right = "to the right" if y > 0 else "to the left"
        return front_back, left_right
    elif dx == 1 and dy == 0:
        cardinal = "east"
        front_back = "in front" if x > 0 else "behind"
        left_right = "to the left" if y < 0 else "to the right"
        return front_back, left_right
    else:
        return "Invalid direction vector"


def describe_object_x(o, state, relative=False):
    if relative:
        dx, dy = o.cur_pos[0] - state.agent_pos[0], o.cur_pos[1] - state.agent_pos[1]
        xd, yd = relative_position(state.agent_dir, (dx, dy))
        units = "rows" if xd in ["in front", "behind"] else "columns"
        return f"{abs(dx)} {units} {xd}"
    else:
        return f"column {o.cur_pos[0]}"


def describe_object_y(o, state, relative=False):
    if relative:
        dx, dy = o.cur_pos[0] - state.agent_pos[0], o.cur_pos[1] - state.agent_pos[1]
        xd, yd = relative_position(state.agent_dir, (dx, dy))
        units = "rows" if yd in ["in front", "behind"] else "columns"
        return f"{abs(dy)} {units} {yd}"
    else:
        return f"row {o.cur_pos[1]}"


def describe_object_state(o):
    if o.type == "bridge":
        return "intact" if o.is_intact else "damaged"

    if o.type == "door":
        if o.is_locked:
            return "locked"
        elif o.is_open:
            return "open"
        else:
            return "closed"

    return ""


def describe_object_color(o):
    if o.type in ["bridge", "hammer", "wall", "fireproof_shoes", "passage"]:
        return ""
    return o.color


def get_attribute(o, name):
    if name == "x":
        return o.cur_pos[0]
    if name == "y":
        return o.cur_pos[1]
    if name == "forward":
        return o.rel_forward
    if name == "turn":
        return o.rel_turn
    if name == "color":
        return describe_object_color(o)
    if name == "state":
        return describe_object_state(o)
    raise NotImplementedError("Attribute not supported!")


def is_identifiable(o, objects, attrs):
    for oo in objects:
        if o == oo:
            continue
        cnt = 0
        for a in attrs:
            cnt += get_attribute(o, a) == get_attribute(oo, a)
        if cnt == len(attrs):
            return False
    return True


def plural_step(n):
    return inflect.engine().plural("step", n)


def describe_object(
    o, objects, relative=True, partial=False, article=None, specified_attrs=None
):
    attrs = ["x", "y", "state", "color"]
    if partial:
        chosen_attrs = []
        for a in random.sample(attrs, len(attrs)):
            chosen_attrs.append(a)
            if is_identifiable(o, objects, chosen_attrs):  # and random.random() < 0.8:
                break
    else:
        chosen_attrs = attrs

    if specified_attrs is not None:
        chosen_attrs = specified_attrs

    d = o.type
    if "color" in chosen_attrs:
        o_color = describe_object_color(o)
        if o_color != "":
            d = o_color + " " + d
    if "state" in chosen_attrs:
        o_state = describe_object_state(o)
        if o_state != "":
            d = o_state + " " + d

    if ("x" in chosen_attrs or "y" in chosen_attrs) and not relative:
        d += " at"
    if "x" in chosen_attrs:
        if relative:
            d += f" {o.rel_forward} {plural_step(o.rel_forward)} forward"
        else:
            d += f" column {o.cur_pos[0]}"
    if "x" in chosen_attrs and "y" in chosen_attrs:
        d += " and"
    if "y" in chosen_attrs:
        if relative:
            d += f" {abs(o.rel_turn)} {plural_step(abs(o.rel_turn))} {'right' if o.rel_turn > 0 else 'left'}"
        else:
            d += f" row {o.cur_pos[1]}"

    if article is not None:
        if article == "the":
            d = "the " + d
        else:
            d = inflect.engine().a(d)

    return d


def describe_position(pos, obs_shape, relative=True):
    if relative:
        rel_forward = obs_shape[1] - 1 - pos[1]
        rel_turn = pos[0] - (obs_shape[0] // 2)
        d = f"{rel_forward} {plural_step(rel_forward)} forward"
        d += f" and {abs(rel_turn)} {plural_step(abs(rel_turn))} {'right' if rel_turn > 0 else 'left'}"
        return d
    return f"column {pos[0]} and row {pos[1]}"


def describe_obstacle_type(o):
    if o.type == "lava":
        return "lava stream"
    elif o.type == "wall":
        return "wall"


def describe_obstacle_direction(dim, state):
    dir_vec = DIR_TO_VEC[state.agent_dir]
    if dir_vec[0] == 0:
        return "vertical" if dim == 0 else "horizontal"
    assert dir_vec[1] == 0
    return "vertical" if dim == 1 else "horizontal"


def get_start_obstacle(obstacles, state):
    dir_vec = DIR_TO_VEC[state.agent_dir]
    dx = dy = 1
    if dir_vec[0] != 0:
        dx = dir_vec[0]
    if dir_vec[1] != 0:
        dy = dir_vec[1]
    best = None
    for o in obstacles:
        cand = (o.cur_pos[0] * dx, o.cur_pos[1] * dy)
        if best is None or cand < best[1]:
            best = (o, cand)
    return best[0]


def describe_obstacle_range(dim, obstacles, l, state, env, relative=False):
    o = get_start_obstacle(obstacles, state)
    d = f"a {describe_obstacle_direction(dim, state)} {describe_obstacle_type(o)} of width {env.obstacle_thickness}"
    if l == 1:
        d += " at"
    else:
        d += " " + f"and length {l} starting from"
    d += " " + f"{describe_object_x(o, state, relative=relative)}"
    d += " and " + f"{describe_object_y(o, state, relative=relative)}"
    return d


def describe_obstacle(o_type):
    if o_type == "lava":
        return "lava pool"
    if o_type == "safe_lava":
        return "cool lava pool"
    if o_type == "wall":
        return "wall"


def extract_objects_from_observation(obs):
    objects = []
    carrying = None
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i][j][0] in [
                OBJECT_TO_IDX["unseen"],
                OBJECT_TO_IDX["empty"],
                OBJECT_TO_IDX["wall"],
                OBJECT_TO_IDX["lava"],
                OBJECT_TO_IDX["safe_lava"],
            ]:
                continue

            o = SimpleNamespace(
                rel_forward=obs.shape[1] - 1 - j,
                rel_turn=i - (obs.shape[0] // 2),
                type=IDX_TO_OBJECT[obs[i][j][0]],
                color=IDX_TO_COLOR[obs[i][j][1]],
            )

            state = obs[i][j][2]
            if o.type == "bridge":
                o.is_intact = state
            elif o.type == "door":
                o.is_open = o.is_locked = 0
                if state == 0:
                    o.is_open = 1
                elif state == 2:
                    o.is_locked = 1

            if i == obs.shape[0] // 2 and j == obs.shape[1] - 1:
                carrying = o
            else:
                objects.append(o)
    return objects, carrying


def find_rectangles(grid, target_value):
    # Dimensions of the grid
    rows, cols = grid.shape
    # To keep track of whether a cell is already included in a rectangle
    included = np.zeros_like(grid, dtype=bool)
    rectangles = []

    # Iterate over each cell in the grid
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Check if current cell is the target value and not already included
            if grid[r, c] == target_value and not included[r, c]:
                # Start a new rectangle
                start_r, start_c = r, c
                end_r, end_c = r, c

                # Expand to the right as far as possible
                while (
                    end_c + 1 < cols
                    and grid[start_r, end_c + 1] == target_value
                    and not included[start_r, end_c + 1]
                ):
                    end_c += 1

                # Try to expand downwards for all columns in the new rectangle
                valid_expansion = True
                while valid_expansion and end_r + 1 < rows:
                    for cc in range(start_c, end_c + 1):
                        if (
                            grid[end_r + 1, cc] != target_value
                            or included[end_r + 1, cc]
                        ):
                            valid_expansion = False
                            break
                    if valid_expansion:
                        end_r += 1

                # Mark all cells in this rectangle as included
                for rr in range(start_r, end_r + 1):
                    for cc in range(start_c, end_c + 1):
                        included[rr, cc] = True

                # Save the rectangle
                rectangles.append(((start_r, start_c), (end_r, end_c)))

    return rectangles


def describe_state(state, relative=True):
    if relative:
        obs = state.partial_obs
        objects, carrying = extract_objects_from_observation(obs)
    else:
        obs = state.full_obs
        objects = []
        for o in state.objects:
            is_visible = True
            if o.cur_pos == (-1, -1):
                is_visible = False
            for oo in state.objects:
                if hasattr(oo, "contains") and oo.contains == o:
                    is_visible = False
                    break
            if is_visible:
                objects.append(o)
        carrying = state.carrying

    d = [f"You are at column {state.agent_pos[0]} and row {state.agent_pos[1]}."]
    d += [f"You are facing {IDX_TO_DIR[state.agent_dir]}."]
    # describe carried object
    if carrying:
        color = describe_object_color(carrying)
        dd = ""
        if color != "":
            dd += " " + color
        dd += " " + carrying.type
        d += [f"You are carrying {inflect.engine().a(dd)}."]
    else:
        d += ["You are not carrying any object."]
    # describe objects within view
    object_descriptions = []
    if objects:
        od = ", ".join(
            [
                inflect.engine().a(describe_object(o, objects, relative=relative))
                for o in objects
            ]
        )
        no = len(objects)
        d += [f"You see {no} {inflect.engine().plural('object', no)}: {od}."]
    else:
        d += ["You do not see any objects."]
    # describe obstacles
    obstacle_to_description = defaultdict(list)
    for o_type in ["wall", "lava", "safe_lava"]:
        if OBJECT_TO_IDX[o_type] in obs:
            rects = find_rectangles(obs[..., 0], OBJECT_TO_IDX[o_type])
            for p1, p2 in rects:
                p1_d = describe_position(p1, obs.shape, relative=relative)
                p2_d = describe_position(p2, obs.shape, relative=relative)
                o_name = describe_obstacle(o_type)
                dd = f"from {p1_d} to {p2_d}"
                obstacle_to_description[o_name].append(dd)

    for o_name, v in obstacle_to_description.items():
        dd = f"There are {inflect.engine().plural(o_name)}: "
        dd += ", ".join(v) + "."
        d += [dd]

    d = " ".join(d)
    d = re.sub(r"\s+", " ", d).strip()

    return d

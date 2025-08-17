import argparse
import sys
import os
import time
import anthropic
from google import genai
from google.genai import types

from tqdm import tqdm

sys.path.append(".")
import pickle


from mindgrid.envs.edits import Edits
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.env_utils import describe_state
from mindgrid.infrastructure.trajectory import Trajectory
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.skills import Skills

from openai import OpenAI

import llmengine
from llmengine import Completion
import json
from typing import Dict

from access_tokens import *
from model_names import MODELS

TEMPERATURE = 0
MAX_TOKENS = 250

RDK_INTRO = """
You are an AI agent helping a human play a 2D grid-based game. The goal of the game is to pick up a target ball on the grid. Here are the key rules of the game:
1. You can pick up objects like keys, balls, boxes, but your inventory can hold only one object at a time (a pair of shoes counts as one object).
2. You can unlock a locked door with a key that has the same color as the door.
3. You can only put an object down in a cell that doesn't already contain another object.
4. When you open a box, it disappears and is replaced by whatever was inside it, if there was something.
"""
TI_INTRO = """
You are an AI agent helping a human play a 2D grid-based game. The goal of the game is to pick up a target ball on the grid. Here are the key rules of the game:
1. You can pick up objects like keys, balls, boxes, hammers, and fireproof shoes, but your inventory can hold only one object at a time (a pair of shoes counts as one object).
2. If you step on lava, you die instantly unless the lava has been cooled or you are carrying fireproof shoes.
3. You can cross bridges safely unless they are damaged. Damaged bridges can be repaired with a hammer.
4. You can only put an object down in a cell that doesn’t already contain another object.
5. When you open a box, it disappears and is replaced by whatever was inside it, if there was something.
"""

ACTION_DESC = """
During each turn, you can perform one action. The available actions are:
1. forward: Move forward by one cell.
2. left: Turn 90 degrees to the left.
3. right: Turn 90 degrees to the right.
4. pickup: Pick up an object directly in front of you if it can be picked up.
5. drop: Drop the object you're holding into the cell directly in front of you.
6. toggle: Interact with an object in front of you (e.g., open a box or door).
"""


def show_env(env):
    env.render_mode = "human"
    env.render()
    input()


def make_example(task, datapoint, include_answer=False):
    config = make_config(config_str=datapoint["config"])
    true_agent_env = make_env(config.true_agent.env)
    false_agent_env = make_env(config.false_agent.env)

    true_agent_env.reset()
    false_agent_env.reset()

    #show_env(true_agent_env)

    true_agent_env_desc = describe_state(true_agent_env.get_state(), relative=False)
    prompt = f"What you observe on the grid: {true_agent_env_desc}" + "\n\n"
    prompt += f"Goal: {true_agent_env.mission}\n\n"
    prompt += f"The human's plan to achieve the goal:\n"

    plan = datapoint["ref_plan"]["false_agent"]
    t = Trajectory()
    for i, (s, a) in enumerate(plan):
        # obs_desc = describe_state(false_agent_env.get_state(), relative=True)
        skill = to_enum(Skills, s).value(**a)
        act_desc = skill.verbalize(false_agent_env)
        t += skill(false_agent_env)
        prompt += f"Step {i + 1}: {act_desc}\n"
    prompt += "\n"

    if include_answer:
        prompt += "Answer: "
        descriptions = []
        edits = true_agent_env.applied_edits[len(false_agent_env.applied_edits) :]
        for e in edits:
            edit_desc = e.verbalize()
            edit_desc = edit_desc[0].upper() + edit_desc[1:] + "."
            descriptions.append(edit_desc)
        prompt += " ".join(descriptions) + "\n"
    else:
        prompt += "Answer: "

    return prompt


def build_prompt(
    datapoint: Dict,
    few_shot,
    train_data,
) -> str:
    config = make_config(config_str=datapoint["config"])
    true_agent_env = make_env(config.true_agent.env)
    false_agent_env = make_env(config.false_agent.env)

    true_agent_env.reset()
    false_agent_env.reset()

    is_rdk = "room_door_key" in datapoint["config"]
    prompt = (RDK_INTRO.strip() if is_rdk else TI_INTRO.strip()) + "\n\n"
    #prompt = prompt.format(goal=true_agent_env.mission)

    prompt += f"The human player proposed a plan to pick up the ball. However, the plan was based on an outdated version of the grid. Since that time, several changes have been made to the grid. You will be provided with an observation of the current grid and the human's plan. The plan is guaranteed to achieve the goal of the game on the old grid, but not necessarily on the current grid. Your task is to infer the changes made to the old grid that results in the current grid. These changes were made sequentially, so you must list them in the correct order. You MUST use the following sentence templates to describe the changes:\n"
    for i, edit in enumerate(Edits):
        prompt += f'{i + 1}. "{edit.value.get_template(true_agent_env)}"\n'
    prompt += "\n"
    prompt += """In these templates: {row} or {col} is a row or column index; {color} is a color name; {state} is a state of a door or a bridge (`closed`, `open`, or `locked` for door, and `damaged` or `intact` for bridge), {tool} is either `key` or `hammer`.\n\n"""

    if args.thought:
        prompt += "Your answer should provide a paragraph in which each sentence is constructed from one of the templates. You may think out loud and when you are ready to give the answer, write a paragraph that begins with the phrase 'The applied edits are:' followed by the edit descriptions. For example: The applied edits are: The color of the target object has been changed to blue. There is a walkable passage at row 1 and column 5.\n\n"

    else:
        prompt += "Your answer should be a paragraph in which each sentence is constructed from one of the templates. Do not output anything else. For example: The color of the target object has been changed to blue. There is a walkable passage at row 1 and column 5.\n\n"

    if few_shot:
        #prompt += f"Here are a few examples to familiar you with the task:\n\n"
        for i in range(few_shot):
            #prompt += "<example>\n"
            prompt += f"[Game {i + 1}]\n"
            prompt += make_example(task, train_data[i], include_answer=True)
            prompt += "[End Game]\n\n"
            #prompt += "</example>\n\n"

        #prompt += "Now, answer the following case:\n\n"
        prompt += f"[Game {few_shot + 1}]\n"
    prompt += make_example(task, datapoint, include_answer=False)

    #print(prompt)
    #input()

    return prompt


def load_data(version, prefix, split):
    with open(f"datasets/{prefix}_games_5000_v{version}.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"{split}"] if split != "train" else games["train"]
    return games


def count_lines(save_file):
    # Open the file in read mode
    with open(save_file, "r") as file:
        # Read the lines in the file
        lines = file.readlines()

    # Count the number of lines
    number_of_lines = len(lines)
    print("Number of lines in the file:", number_of_lines)
    return number_of_lines


def execute_plan(env, plan):
    env.reset()
    t = Trajectory()
    for skill, kwargs in plan:
        t += Skills[skill].value(**kwargs)(env)
        if t.is_null:
            t = Trajectory()
    success = (
        env.carrying is not None
        and env.carrying.type == "ball"
        and env.carrying.color == env.target_color
    )
    return success * 100 - env.step_count


def find_start_idx(result_file):
    cnt = 0
    game_id = output = None
    with open(result_file, "r") as file:
        for line in file:
            if line.startswith("env-test_out"):
                if game_id is not None:
                    cnt += 1
                game_id, output = line.split("\t")
            else:
                output += line
    cnt += 1
    return cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_version", type=int, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--task", type=str, default="teach")
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--num_games", type=int, default=50)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--thought", action="store_true", default=False)

    args = parser.parse_args()

    llmengine.api_engine.api_key = SCALE_KEY

    version = args.version
    prefix = args.prefix
    task = args.task
    few_shot = args.few_shot
    model = MODELS[args.model_id]
    num_games = args.num_games

    if any(x in model for x in ("gpt", "o1", "o3")):
        client = OpenAI(api_key=OPENAI_KEY)

    if "claude" in model:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    train_games = load_data(version, prefix, "train")
    test_games = load_data(version, prefix, "test_out")

    result_dir = f"methods/llm-prompt/{args.result_dir}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    save_file = f"{result_dir}/{prefix}_{task}_5000_v{version}.{num_games}-games.{few_shot}-shot.{model}.prompt-v{args.prompt_version}.out"

    if args.thought:
        save_file = save_file.split(".")[0] + ".thought.out"
        MAX_TOKENS = 1000

    print(f"Save to {save_file} ?")
    input()

    start_idx = 0
    if os.path.exists(save_file):
        start_idx = find_start_idx(save_file)

    print(f"Starting from {start_idx}")

    for i, game in enumerate(tqdm(test_games[: num_games])):
        if i >= start_idx:
            prompt = build_prompt(game, args.few_shot, train_games)

            if any(x in model for x in ("gpt", "o1", "o3")):
                resp = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                model_answer = resp.choices[0].message.content
            elif "claude" in model:
                message = client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                model_answer = message.content[0].text
            else:
                resp = Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_TOKENS,
                )
                model_answer = json.loads(resp.json())["output"]["text"]

            #print(model_answer)
            #input()

            with open(save_file, "a") as f:
                f.write(game["id"] + "\t" + model_answer + "\n")

            prompt_file = save_file.replace(".out", ".prompt")
            with open(prompt_file, "a") as f:
                f.write(game["id"] + "\t" + prompt + "\n")


            time.sleep(2)

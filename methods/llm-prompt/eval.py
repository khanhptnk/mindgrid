import os
import argparse
import sys
import re
import numpy as np

sys.path.append(".")
import pickle
import scipy

from nltk.tokenize import sent_tokenize

from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.trajectory import Trajectory
from mindgrid.skills import Skills
from mindgrid.planner import Planner
from mindgrid.envs.edits import parse_from_description


from model_names import MODELS


def show_env(env):
    env.render_mode = "human"
    env.render()
    input()


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
        t += skill.value(**kwargs)(env)
        if t.is_null:
            return t
    return t


def score_action_plan(env, actions):
    env.reset()
    env.render()
    for a in actions:
        env.step(a)
    success = (
        env.carrying is not None
        and env.carrying.type == "ball"
        and env.carrying.color == env.target_color
    )
    return success * 100 #- env.step_count


def get_score(planner, skills, pred_env, true_env):
    plan = planner(pred_env, skills)
    if plan is None:
        return 0
    traj = execute_plan(pred_env, plan)
    return score_action_plan(true_env, traj.actions)


def preprocess_output(text):
    ret = []
    if args.thought:
        phrase = "The applied edits are:"
        start_index = text.find(phrase)
        if start_index == -1:
            text = ""
        else:
            text = text[start_index + len(phrase):].strip()

    end_game_marker = "[End Game]"
    end_index = text.find(end_game_marker)

    # If "[End Game]" is found, return the text before it
    if end_index != -1:
        text = text[:end_index].strip()

    for x in text.split("\n"):
        ret.extend(sent_tokenize(x))
    for i, s in enumerate(ret):
        s = s.lower().strip()
        if s.endswith("."):
            s = s[:-1].strip()
        s = re.sub(r"^(?:\d+\.\s*|- |\* |step \d+: |answer: )", "", s)
        ret[i] = s
    ret = [s for s in ret if s]
    return ret


def read_result_file(result_file):
    data = []
    game_id = output = None
    with open(result_file, "r") as file:
        for line in file:
            if line.startswith("env-test_out"):
                if game_id is not None:
                    data.append((game_id, preprocess_output(output.strip())))
                game_id, output = line.split("\t")
            else:
                output += line
    data.append((game_id, preprocess_output(output.strip())))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_version", type=int, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--task", type=str, default="teach")
    parser.add_argument("--num_games", type=int, default=50)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--no_edit", action="store_true", default=False)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--thought", action="store_true", default=False)

    args = parser.parse_args()

    version = args.version
    prefix = args.prefix
    task = args.task
    few_shot = args.few_shot
    model = MODELS[args.model_id]
    num_games = args.num_games

    result_file = f"methods/llm-prompt/{args.result_dir}/{prefix}_{task}_5000_v{version}.{num_games}-games.{few_shot}-shot.{model}.prompt-v{args.prompt_version}.out"
    if os.path.exists(result_file + ".reformatted"):
        result_file += ".reformatted"
    if args.thought:
        result_file = result_file.split(".")[0] + ".thought.out"

    test_games = load_data(version, prefix, "test_out")

    score_diffs = []
    model_scores = []
    true_scores = []

    print(f"Eval {result_file}?")
    input()

    data = read_result_file(result_file)

    lens = []
    for game_id, descriptions in data:
        lens.append(len(descriptions))
    print(round(np.mean(lens), 2))

    lens = []
    for i, (game, (game_id, edit_descriptions)) in enumerate(zip(test_games, data)):
        print(i)

        assert game["id"] == game_id

        config = make_config(config_str=game["config"])

        #print(config.true_agent.env.edits)
        #print(config.false_agent.env.edits)
        #lens.append(len(config.true_agent.env.edits) - len(config.false_agent.env.edits))

        true_agent_env = make_env(config.true_agent.env)
        false_agent_env = make_env(config.false_agent.env)

        if not args.no_edit:
            edits = []
            for d in edit_descriptions:
                e = parse_from_description(d, false_agent_env)
                if e is not None:
                    edits.append(e)
            false_agent_env.edit(edits)

        planner = Planner(config.false_agent.env)
        assert config.false_agent.skill == config.true_agent.skill
        skills = [Skills[s] for s in config.false_agent.skill]

        model_score = get_score(planner, skills, false_agent_env, true_agent_env)
        true_score = get_score(planner, skills, true_agent_env, true_agent_env)

        print(true_score, model_score, true_score - model_score)

        score_diffs.append(true_score - model_score)
        true_scores.append(true_score)
        model_scores.append(model_score)

    print(
        "Diff:",
        round(np.mean(score_diffs), 2),
        "$\\pm$",
        round(scipy.stats.sem(score_diffs), 2),
    )

    print(
        "True:",
        round(np.mean(true_scores), 2),
        "$\\pm$",
        round(scipy.stats.sem(true_scores), 2),
    )

    print(
        "Model:",
        round(np.mean(model_scores), 2),
        "$\\pm$",
        round(scipy.stats.sem(model_scores), 2),
    )


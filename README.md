# MindGrid


This Repo contains code for the MindGrid benchmark and experiments introduced in the paper [Practical Alignment](https://github.com/khanhptnk/practical-alignment/blob/main/practical_alignment_08_2025.pdf) (Khanh and Trinh, 2025)

1. Create new dataset by running:
```
python game_factory.py --prefix ${SOME_NAME} --data_size 100
```

Then press `Enter` to confirm the location of the dataset.

2. Put your API keys in a file called `methods/llm-prompt/access_tokens.py`

3. See `methods/llm-prompt/model_names.py` for a list of models. Feel free to add your own. 

4. Run a model on the dataset:

```
python methods/llm-prompt/prompt.py --prefix env --version 4 --model_id 3 --few_shot 1 --prompt_version 4 --result_dir output --num_games 10
```

5. Evaluate output:

```
python methods/llm-prompt/eval.py --prefix env --version 4 --model_id 3 --few_shot 1 --prompt_version 4 --result_dir output --num_games 10
```

(note that the arguments passed to `eval.py` are exactly those used for `prompt.py`)

You will see these last three lines in the console
```
Diff: 80.0 $\pm$ 13.33
True: 90.0 $\pm$ 10.0
Model: 10.0 $\pm$ 10.0
```

where `Diff` is the optimality gap defined in the paper and `True` and `Model` are the two terms in the formula.

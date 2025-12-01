# g1 Tennis

This project was trained in **IsaacSim** 4.5.0 and **IsaacLab** 2.0.0, and was tested in the latest version **IsaacSim** 5.1.0 and **IsaacLab** 2.3.0. It is recommended to use the latest version of isaaclab.

## setup

1. install isaacsim and isaaclab following [Local Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
2. under your `isaaclab/source` path, clone this project. Path should look like this: `isaaclab/source/g1_tennis`
3. install the project:
   `python -m pip install -e source/g1_tennis`
4. under your `isaaclab\scripts\reinforcement_learning\rsl_rl` path, you will find two .py files: `train.py` and `play.py`. In both files, add `import g1_tennis`  code before the `main` function.

## how to use

The logs should be saved in `isaaclab/logs/rsl_rl/g1_rough` path.

1. train:
   `python scripts/rsl_rl/train.py --num_envs=4096 --task=g1_tennis --headless [--max_iteration=ITERATION] [--resume=True --load_run=RUNNAME]`
2. play:
   `python scripts/rsl_rl/play.py --num_envs=1 --task=g1_tennis --load_run=RUNNAME`

## checkpoints

Trained checkpoints are saved in `isaaclab/source/g1_tennis/logs`, you need to copy them to the default path (`isaaclab/logs/rsl_rl/g1_rough`).

The `newVel8good` is trained with robotV2, and the `V3good` is trained with robotV3 (its performance is not that stable because the pitch lock)

Default robot is robotV3, if you want to switch the robot, modify  `usd_path=os.path.join(script_dir, "assets", "robotV3", "g1_racket_29.usd")` in `g1_tennis_cfg.py`.

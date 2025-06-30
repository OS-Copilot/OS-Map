import datetime
import json
import logging
import os

from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    error_logger = setup_logger(example, example_result_dir, logger_name="error")
    agent.reset(runtime_logger, error_logger, example_result_dir)
    runtime_logger.info(f"[Instruction]\n{instruction}")
    obs = env.reset(task_config=example)
    start_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    
    # TODO: reformat levels in json files
    level2steps = {"l1": 15, "l2": 15, "l3": 20, "l4": 40}
    if example["id"].startswith("l1_"):
        level = "l1"
    elif example["id"].startswith("l2_"):
        level = "l2"
    elif example["id"].startswith("l3_"):
        level = "l3"
    elif example["id"].startswith("l4_"):
        level = "l4"
    else:
        level = example.get("level", "NONE")
    with open(os.path.join(example_result_dir, f"level.txt"), "w") as _f:
        _f.write(level)
    max_steps = level2steps[level]
    
    with open(os.path.join(example_result_dir, f"step_0_{start_timestamp}.png"), "wb") as _f:
        _f.write(obs['screenshot'])
    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction,
            obs
        )
        if actions == None:
            actions = [None]
        for action in actions:
            if action == None:
                continue
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir, logger_name="runtime"):
    runtime_logger = logging.getLogger(f"{logger_name}.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, f"{logger_name}.log")))
    return runtime_logger

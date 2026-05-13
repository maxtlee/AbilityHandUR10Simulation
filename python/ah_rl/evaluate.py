"""Evaluate a trained grasping policy with visualization.

Usage:
    python -m ah_rl.evaluate ah_rl_runs/sac_20260513/best/best_model
    python -m ah_rl.evaluate ah_rl_runs/sac_20260513/final_model --episodes 20
    python -m ah_rl.evaluate --random                # random policy baseline
"""

import argparse
import time

import numpy as np

from ah_rl.envs.grasp_env import AHGraspEnv


def evaluate(args):
    env = AHGraspEnv(render_mode="human", max_episode_steps=args.max_steps)

    model = None
    if args.model_path and not args.random:
        try:
            from stable_baselines3 import SAC, PPO
        except ImportError:
            print("stable-baselines3 is required: pip install 'stable-baselines3[extra]'")
            raise SystemExit(1)

        # Try loading as SAC first, then PPO
        for algo_cls in [SAC, PPO]:
            try:
                model = algo_cls.load(args.model_path)
                print(f"Loaded {algo_cls.__name__} model from {args.model_path}")
                break
            except Exception:
                continue
        if model is None:
            print(f"Could not load model from {args.model_path}")
            raise SystemExit(1)

    print(f"Running {args.episodes} episodes (max {args.max_steps} steps each)")
    if model is None:
        print("Using random policy")

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        max_lift = 0.0
        steps = 0

        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            steps += 1

            lift = info.get("lift_height", 0.0)
            if lift is not None:
                max_lift = max(max_lift, lift)

            if terminated or truncated:
                break

            if args.slow:
                time.sleep(0.02)

        success = max_lift > 0.03  # lifted at least 3cm
        successes += int(success)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(
            f"  Episode {ep + 1}/{args.episodes}: "
            f"reward={total_reward:.2f}, steps={steps}, "
            f"max_lift={max_lift:.3f}m, success={success}"
        )

    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Mean reward:   {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Mean length:   {np.mean(episode_lengths):.0f}")
    print(f"  Success rate:  {successes}/{args.episodes} ({100 * successes / args.episodes:.0f}%)")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate AH grasping policy")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to saved model (omit for random policy)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions instead of a model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Add delay between steps for slower visualization",
    )
    args = parser.parse_args()

    if args.model_path is None and not args.random:
        parser.error("Provide a model path or use --random")

    evaluate(args)


if __name__ == "__main__":
    main()

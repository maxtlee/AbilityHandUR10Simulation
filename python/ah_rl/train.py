"""Train a grasping policy for the UR10e + Ability Hand.

Defaults to the hand-only env (AHGraspEnvFixed): arm frozen, policy controls
only the 6 hand DOFs. Pass --env arm_hand to use the legacy 12-DOF env.

Usage:
    python -m ah_rl.train                          # SAC, 200k steps, hand-only
    python -m ah_rl.train --env arm_hand --steps 500k
    python -m ah_rl.train --algo ppo --steps 1M
    python -m ah_rl.train --n-envs 8 --steps 2M
    python -m ah_rl.train --render                 # watch one env during training
    python -m ah_rl.train --device cuda            # force GPU (errors if unavailable)

For GPU support, install a CUDA build of PyTorch *before* SB3 pulls in the
CPU wheel, e.g.:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
(use cu118 for older NVIDIA drivers).
"""

import argparse
import os
from datetime import datetime

from envs.grasp_env import AHGraspEnv
from envs.grasp_env_fixed import AHGraspEnvFixed


def parse_steps(s: str) -> int:
    """Parse step counts like '500k', '1M', '2_000_000'."""
    s = s.strip().replace("_", "")
    if s.upper().endswith("K"):
        return int(float(s[:-1]) * 1_000)
    if s.upper().endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def make_env(rank: int, seed: int, render: bool = False, env_name: str = "hand_only"):
    """Factory for creating vectorized envs."""
    def _init():
        render_mode = "human" if (render and rank == 0) else None
        EnvCls = AHGraspEnvFixed if env_name == "hand_only" else AHGraspEnv
        env = EnvCls(render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    try:
        import torch
        from stable_baselines3 import SAC, PPO
        from stable_baselines3.common.vec_env import (
            SubprocVecEnv,
            DummyVecEnv,
            VecMonitor,
        )
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
        )
    except ImportError:
        print("stable-baselines3 is required: pip install 'stable-baselines3[extra]'")
        raise SystemExit(1)

    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        print(
            "ERROR: --device cuda requested but torch.cuda.is_available() is False.\n"
            "  The installed PyTorch is CPU-only or the CUDA driver is missing.\n"
            "  Install a CUDA build, e.g.:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  (use cu118 for older drivers)."
        )
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("ah_rl_runs", f"{args.env}_{args.algo}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    total_steps = parse_steps(args.steps)

    # Create vectorized training envs. Force DummyVecEnv when rendering:
    # the MuJoCo passive viewer is unreliable inside SubprocVecEnv workers
    # on Windows (spawn-based multiprocessing + GUI).
    VecCls = DummyVecEnv if (args.n_envs == 1 or args.render) else SubprocVecEnv
    train_envs = VecMonitor(
        VecCls([make_env(i, args.seed, args.render, args.env) for i in range(args.n_envs)])
    )

    # Create eval env (single, no render)
    eval_env = VecMonitor(
        DummyVecEnv([make_env(0, args.seed + 1000, env_name=args.env)])
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_steps // 20, 10_000),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="model",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(total_steps // 50, 5_000),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Algorithm
    algo_cls = SAC if args.algo == "sac" else PPO

    if args.algo == "sac":
        model = algo_cls(
            "MlpPolicy",
            train_envs,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_starts=1000,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb"),
            seed=args.seed,
            device=args.device,
        )
    else:  # ppo
        model = algo_cls(
            "MlpPolicy",
            train_envs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb"),
            seed=args.seed,
            device=args.device,
        )

    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "n/a"
    print(f"Training {args.algo.upper()} for {total_steps:,} steps")
    print(f"  envs: {args.n_envs}, log dir: {log_dir}")
    print(f"  obs dim: {train_envs.observation_space.shape}")
    print(f"  act dim: {train_envs.action_space.shape}")
    print(f"  torch: {torch.__version__}, cuda available: {cuda_available}, gpu: {gpu_name}")
    print(f"  effective device: {model.device}")
    if str(model.device) == "cpu" and args.algo == "ppo":
        pass  # PPO on CPU is often fine for tiny MLPs; no warning needed.
    if str(model.device).startswith("cuda") and args.algo == "ppo":
        print("  note: PPO with this small MLP may not be faster on GPU than CPU.")

    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    train_envs.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train AH grasping policy")
    parser.add_argument(
        "--algo",
        choices=["sac", "ppo"],
        default="sac",
        help="RL algorithm (default: sac)",
    )
    parser.add_argument(
        "--env",
        choices=["hand_only", "arm_hand"],
        default="hand_only",
        help="Env: hand_only (6 DOF, arm frozen) or arm_hand (legacy 12 DOF)",
    )
    parser.add_argument(
        "--steps",
        default="200k",
        help="Total training steps, e.g. 200k, 1M (default: 200k)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render one training env (slows training)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for the policy/value nets (default: auto). "
             "Physics always runs on CPU; only the network step moves to GPU.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

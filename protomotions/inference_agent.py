# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **Q**: Quit

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--one-step-reference-eval",
        action="store_true",
        default=False,
        help=(
            "Reset parallel environments to evenly spaced reference times, execute one "
            "deterministic policy/physics step, and save r+gamma*V(s')."
        ),
    )
    parser.add_argument(
        "--root-reference-cem",
        action="store_true",
        default=False,
        help=(
            "Search a low-frequency vertical root correction with one closed-loop "
            "physics rollout per parallel environment."
        ),
    )
    parser.add_argument(
        "--root-cem-prior-motion",
        type=str,
        default=None,
        help="Optional MotionLib carrying root_reference_reliability and an initial correction.",
    )
    parser.add_argument("--root-cem-iterations", type=int, default=12)
    parser.add_argument("--root-cem-knots", type=int, default=8)
    parser.add_argument("--root-cem-elite-fraction", type=float, default=0.10)
    parser.add_argument("--root-cem-initial-std", type=float, default=0.12)
    parser.add_argument("--root-cem-minimum-std", type=float, default=0.005)
    parser.add_argument(
        "--root-cem-minimum-correction",
        type=float,
        default=-0.60,
        help="Minimum vertical root correction searched by CEM, in metres.",
    )
    parser.add_argument(
        "--root-cem-maximum-correction",
        type=float,
        default=0.60,
        help="Maximum vertical root correction searched by CEM, in metres.",
    )
    parser.add_argument("--root-cem-seed", type=int, default=20260918)
    parser.add_argument(
        "--one-step-start-fraction",
        type=float,
        default=0.0,
        help="First normalized motion time used by --one-step-reference-eval.",
    )
    parser.add_argument(
        "--one-step-end-fraction",
        type=float,
        default=1.0,
        help="Last normalized motion time used by --one-step-reference-eval.",
    )
    parser.add_argument(
        "--one-step-warmup-steps",
        type=int,
        default=1,
        help=(
            "Short settling horizon held at each reference time before the measured step; "
            "one step is required for Isaac Gym forward kinematics after an indexed reset."
        ),
    )
    parser.add_argument(
        "--critic-bootstrap-horizon-steps",
        type=int,
        default=1,
        help=(
            "Number of measured physics steps in the bootstrapped return. One preserves "
            "the original one-step metric; small values such as 4 or 8 capture short-term "
            "contact propagation without a complete rollout."
        ),
    )
    parser.add_argument(
        "--loop-motion",
        action="store_true",
        default=False,
        help="Loop one fixed motion forever without random resampling.",
    )
    parser.add_argument(
        "--motion-id",
        type=int,
        default=0,
        help="Motion ID to replay when --loop-motion is enabled.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--terrain-mesh",
        type=str,
        default=None,
        help="External world-space PLY/OBJ triangle mesh terrain.",
    )
    parser.add_argument(
        "--control-armature",
        type=float,
        default=None,
        help="Override every configured robot joint armature for deployment ablations.",
    )
    parser.add_argument(
        "--preserve-reference-world-position",
        action="store_true",
        help="Do not move reference motion to a sampled terrain location.",
    )
    parser.add_argument(
        "--masked-mimic-full-conditioning",
        action="store_true",
        help="Use every checkpoint-supported MaskedMimic body with translation and rotation visible.",
    )
    parser.add_argument(
        "--smpl-shape-from-motion",
        action="store_true",
        help="For SMPL checkpoints, load/generate a CRISP/SMPLSim humanoid asset from motion_betas[motion_id].",
    )
    parser.add_argument(
        "--shape-asset-dir",
        type=str,
        default="data/easymimic/assets/mjcf",
        help="Directory for generated per-shape SMPL MJCF assets.",
    )
    parser.add_argument(
        "--smpl-data-dir",
        type=str,
        default=None,
        help="Directory containing SMPL model PKLs for --smpl-shape-from-motion.",
    )
    parser.add_argument(
        "--disable-action-residual",
        "--disable-residual",
        action="store_true",
        default=False,
        help="For ablation, force a configured action- or latent-residual actor to output zero residual.",
    )
    parser.add_argument(
        "--save-predicted-motion-lib",
        action="store_true",
        default=False,
        help="During --full-eval, save the simulated trajectory as a packaged MotionLib.",
    )
    parser.add_argument(
        "--disable-reference-contact-rewards",
        action="store_true",
        default=False,
        help=(
            "Inference/evaluation only: remove reward components that require reference "
            "contact labels. Use for transferred clips whose packaged contacts are absent "
            "or all zero; policy observations and reported tracking metrics are unchanged."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Write evaluation artifacts here instead of beside the checkpoint.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import logging  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from dataclasses import asdict  # noqa: E402


def apply_smpl_shape_from_motion(robot_config, motion_file: str, motion_id: int) -> str:
    """Configure inference robot from the fixed motion's SMPL shape."""
    from protomotions.utils.smpl_shape import configure_robot_from_motion_shape

    return configure_robot_from_motion_shape(
        robot_config=robot_config,
        motion_file=motion_file,
        motion_id=motion_id,
        shape_asset_dir=args.shape_asset_dir,
        smpl_data_dir=args.smpl_data_dir,
        preserve_control_info=True,
    )


def packaged_motion_count(motion_file: str) -> int:
    """Return the number of clips in a packaged MotionLib without constructing an env."""
    packaged = torch.load(motion_file, map_location="cpu", weights_only=False)
    if not isinstance(packaged, dict) or "motion_num_frames" not in packaged:
        raise ValueError(
            "--smpl-shape-from-motion with --full-eval requires a packaged "
            "MotionLib containing motion_num_frames."
        )
    return int(torch.as_tensor(packaged["motion_num_frames"]).numel())


def run_one_step_reference_evaluation(agent, env, output_dir: Path) -> dict:
    """Evaluate a short deterministic physics return bootstrapped by the critic."""
    if env.motion_lib.num_motions() != 1:
        raise ValueError("One-step reference evaluation currently requires one motion.")
    if not 0.0 <= args.one_step_start_fraction < args.one_step_end_fraction <= 1.0:
        raise ValueError("Expected 0 <= one-step start < end <= 1.")
    if args.one_step_warmup_steps < 1:
        raise ValueError(
            "Isaac Gym one-step evaluation requires at least one warmup step."
        )
    if args.critic_bootstrap_horizon_steps < 1:
        raise ValueError("critic bootstrap horizon must be at least one step")
    if hasattr(env.motion_manager, "set_clip_mode"):
        env.motion_manager.set_clip_mode(False)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    motion_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    motion_length = env.motion_lib.get_motion_length(
        torch.zeros(1, device=env.device, dtype=torch.long)
    )[0]
    latest_time = torch.clamp(motion_length - env.dt, min=0.0)
    sample_times = (
        torch.linspace(
            float(args.one_step_start_fraction),
            float(args.one_step_end_fraction),
            env.num_envs,
            device=env.device,
        )
        * latest_time
    )
    env.motion_manager.motion_ids[env_ids] = motion_ids
    env.motion_manager.motion_times[env_ids] = sample_times
    observations, _ = env.reset(env_ids, disable_motion_resample=True)
    observations = agent.add_agent_info_to_obs(observations)
    observation_td = agent.obs_dict_to_tensordict(observations)
    with torch.inference_mode():
        for _ in range(args.one_step_warmup_steps):
            warmup_output = agent.model(observation_td)
            warmup_action = warmup_output.get(
                "mean_action", warmup_output.get("action")
            )
            observations, _, _, _, _ = env.step(warmup_action)
            env.motion_manager.motion_times[env_ids] = sample_times
            env._current_context = None
            env.compute_observations(context=env.context)
            observations = agent.add_agent_info_to_obs(env.get_obs())
            observation_td = agent.obs_dict_to_tensordict(observations)
        current_state = env.simulator.get_robot_state().clone()
        current_output = agent.model(observation_td)
        current_value = current_output["value"].reshape(env.num_envs)
        discounted_rewards = torch.zeros(env.num_envs, device=env.device)
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        any_done = torch.zeros_like(active)
        any_terminated = torch.zeros_like(active)
        actions = []
        step_rewards = []
        discount = 1.0
        extras = {}
        for step in range(args.critic_bootstrap_horizon_steps):
            output = current_output if step == 0 else agent.model(observation_td)
            action = output.get("mean_action", output.get("action"))
            actions.append(action)
            if "root_pos_residual_scaled" in output and hasattr(
                env, "set_reference_root_residual"
            ):
                env.set_reference_root_residual(output["root_pos_residual_scaled"])
            next_observations, rewards, dones, terminated, extras = env.step(action)
            rewards = rewards.reshape(env.num_envs)
            dones = dones.reshape(env.num_envs)
            terminated = terminated.reshape(env.num_envs)
            step_rewards.append(rewards)
            discounted_rewards += discount * rewards * active.float()
            any_done |= dones
            any_terminated |= terminated
            active &= ~terminated
            discount *= agent.gamma
            next_observations = agent.add_agent_info_to_obs(next_observations)
            observation_td = agent.obs_dict_to_tensordict(next_observations)
        next_value = agent.model(observation_td)["value"].reshape(env.num_envs)
    td_proxy = discounted_rewards + discount * next_value * active.float()
    action_sequence = torch.stack(actions, dim=1)
    reward_sequence = torch.stack(step_rewards, dim=1)
    action = action_sequence[:, 0]
    rewards = discounted_rewards
    dones = any_done
    terminated = any_terminated
    next_state = env.simulator.get_robot_state()
    root_position_change = torch.linalg.vector_norm(
        next_state.root_pos - current_state.root_pos, dim=-1
    )
    root_velocity_change = torch.linalg.vector_norm(
        next_state.root_vel - current_state.root_vel, dim=-1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "one_step_reference_eval.pt"
    torch.save(
        {
            "motion_ids": motion_ids.detach().cpu(),
            "motion_times": sample_times.detach().cpu(),
            "current_value": current_value.detach().cpu(),
            "action": action.detach().cpu(),
            "action_sequence": action_sequence.detach().cpu(),
            "reward": rewards.detach().cpu(),
            "reward_sequence": reward_sequence.detach().cpu(),
            "next_value": next_value.detach().cpu(),
            "td_proxy": td_proxy.detach().cpu(),
            "td_residual": (td_proxy - current_value).detach().cpu(),
            "done": dones.detach().cpu(),
            "terminated": terminated.detach().cpu(),
            "root_position_change_m": root_position_change.detach().cpu(),
            "root_velocity_change_mps": root_velocity_change.detach().cpu(),
            "current_root_position": current_state.root_pos.detach().cpu(),
            "next_root_position": next_state.root_pos.detach().cpu(),
            "current_root_velocity": current_state.root_vel.detach().cpu(),
            "next_root_velocity": next_state.root_vel.detach().cpu(),
            "contact_force": next_state.rigid_body_contact_forces.detach().cpu(),
            "raw_next_state": {
                key[len("raw/") :]: value.detach().cpu()
                for key, value in extras.items()
                if key.startswith("raw/") and isinstance(value, torch.Tensor)
            },
        },
        output_path,
    )

    def statistics(values: torch.Tensor) -> dict:
        values = values.detach().float().cpu()
        return {
            "mean": float(values.mean()),
            "p05": float(torch.quantile(values, 0.05)),
            "p50": float(torch.quantile(values, 0.50)),
            "p95": float(torch.quantile(values, 0.95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "method": "parallel reference-state short-horizon physics/critic evaluation",
        "definition": "B_H=sum(k=0..H-1) gamma^k*r_k + gamma^H*V(s_H)",
        "samples": int(env.num_envs),
        "motion_length_s": float(motion_length),
        "time_fraction": [
            float(args.one_step_start_fraction),
            float(args.one_step_end_fraction),
        ],
        "gamma": float(agent.gamma),
        "critic_bootstrap_horizon_steps": int(args.critic_bootstrap_horizon_steps),
        "warmup_steps": int(args.one_step_warmup_steps),
        "safe_reference_reset": bool(env.config.safe_reference_reset),
        "current_value": statistics(current_value),
        "reward": statistics(rewards),
        "next_value": statistics(next_value),
        "td_proxy": statistics(td_proxy),
        "td_residual": statistics(td_proxy - current_value),
        "root_position_change_m": statistics(root_position_change),
        "root_velocity_change_mps": statistics(root_velocity_change),
        "termination_fraction": float(terminated.float().mean()),
        "output": str(output_path),
    }
    summary_path = output_dir / "one_step_reference_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Temporary function to enable domain randomization for testing.

#     TODO: find a better way for sophisticated tmp inference overrides beyond CLI.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )

#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01

#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()
    if args.full_eval and args.loop_motion:
        parser.error("--full-eval and --loop-motion are mutually exclusive")
    if args.full_eval and args.one_step_reference_eval:
        parser.error("--full-eval and --one-step-reference-eval are mutually exclusive")
    selected_batch_modes = sum(
        bool(value)
        for value in (
            args.full_eval,
            args.one_step_reference_eval,
            args.root_reference_cem,
        )
    )
    if selected_batch_modes > 1:
        parser.error(
            "--full-eval, --one-step-reference-eval and --root-reference-cem "
            "are mutually exclusive"
        )
    if args.root_reference_cem and args.motion_file is None:
        parser.error("--root-reference-cem requires --motion-file")

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), (
        f"Could not find resolved configs at {resolved_configs_path}"
    )

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    # Apply backward compatibility fixes for old checkpoints
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # # Temporary: Enable domain randomization for testing (uncomment to use)
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # from protomotions.robot_configs.base import ControlType
    # robot_config.control.control_type = ControlType.PROPORTIONAL

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file  # Always present

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    if args.terrain_mesh is not None:
        log.info(f"CLI override: external terrain mesh = {args.terrain_mesh}")
        terrain_config.mesh_path = str(Path(args.terrain_mesh).expanduser().resolve())

    if args.control_armature is not None:
        for control_info in robot_config.control.control_info.values():
            control_info.armature = float(args.control_armature)
        log.info(
            "CLI override: all joint armatures = %s", args.control_armature
        )

    if args.preserve_reference_world_position:
        log.info("CLI override: preserving reference motion world position")
        env_config.preserve_reference_world_position = True
        env_config.ref_respawn_offset = 0.0

    if args.masked_mimic_full_conditioning:
        masked_cfg = env_config.control_components.get("masked_mimic")
        if masked_cfg is None:
            raise ValueError(
                "--masked-mimic-full-conditioning requires a MaskedMimic checkpoint"
            )
        log.info("CLI override: deterministic full MaskedMimic conditioning")
        masked_cfg.deterministic_full_body_conditioning = True

    if args.smpl_shape_from_motion:
        if args.full_eval and packaged_motion_count(motion_lib_config.motion_file) != 1:
            raise ValueError(
                "--smpl-shape-from-motion with --full-eval requires exactly one motion. "
                "A single IsaacGym simulation can only use one humanoid asset."
            )
        selected_asset = apply_smpl_shape_from_motion(
            robot_config=robot_config,
            motion_file=motion_lib_config.motion_file,
            motion_id=args.motion_id,
        )
        log.info(
            f"CLI override: using SMPL shape asset for motion_id={args.motion_id}: {selected_asset}"
        )

    if args.disable_action_residual:
        actor_cfg = getattr(agent_config.model, "actor", None)
        if not hasattr(actor_cfg, "disable_residual"):
            raise ValueError(
                "--disable-residual requires a residual-adapter checkpoint."
            )
        actor_cfg.disable_residual = True
        log.info("CLI ablation: residual adapter forced to zero")

    if args.save_predicted_motion_lib:
        if not args.full_eval:
            raise ValueError("--save-predicted-motion-lib requires --full-eval.")
        if not hasattr(agent_config.evaluator, "save_predicted_motion_lib_every"):
            raise ValueError(
                "The configured evaluator cannot save a predicted MotionLib."
            )
        agent_config.evaluator.save_predicted_motion_lib_every = 1
        log.info("CLI override: save predicted MotionLib after this evaluation")

    if args.disable_reference_contact_rewards:
        removed = []
        for name in ("contact_match_rew",):
            if name in env_config.reward_components:
                env_config.reward_components.pop(name)
                removed.append(name)
        log.info(
            "CLI evaluation override: disabled reference-contact rewards: "
            + (", ".join(removed) if removed else "none configured")
        )

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    # Create fabric config for inference (simplified)
    # MuJoCo is CPU-only, so force CPU accelerator
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Convert friction for simulator compatibility
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent
    )

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup()
    # Inference needs policy/value weights and normalization only. Loading
    # Adam moments and historical training counters wastes memory and time.
    agent.load(
        args.checkpoint,
        load_env=False,
        resume_training_state=False,
        # Critic-based one-step evaluation/CEM needs values in the checkpoint's
        # reward-normalized coordinate system, but never needs Adam or counters.
        load_reward_normalization=True,
    )

    try:
        if args.full_eval:
            agent.evaluator.eval_count = 0
            evaluation_log, evaluated_score = agent.evaluator.evaluate()

            # Print evaluation metrics
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            for key, value in sorted(evaluation_log.items()):
                print(f"  {key}: {value:.6f}")
            print("=" * 60)
            if evaluated_score is not None:
                print(f"  Overall Score: {evaluated_score:.6f}")
            print("=" * 60 + "\n")
            metrics_path = Path(agent_kwargs["root_dir"]) / "evaluation_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = {key: float(value) for key, value in evaluation_log.items()}
            metrics["overall_score"] = (
                None if evaluated_score is None else float(evaluated_score)
            )
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            log.info(f"Evaluation metrics saved to {metrics_path}")
        elif args.one_step_reference_eval:
            run_one_step_reference_evaluation(
                agent,
                env,
                Path(agent_kwargs["root_dir"]),
            )
        elif args.root_reference_cem:
            from protomotions.utils.root_reference_cem import (
                RootReferenceCEMConfig,
                run_root_reference_cem,
            )

            run_root_reference_cem(
                agent=agent,
                env=env,
                source_motion_path=Path(args.motion_file),
                output_dir=Path(agent_kwargs["root_dir"]),
                config=RootReferenceCEMConfig(
                    iterations=args.root_cem_iterations,
                    knots=args.root_cem_knots,
                    elite_fraction=args.root_cem_elite_fraction,
                    initial_std_m=args.root_cem_initial_std,
                    minimum_std_m=args.root_cem_minimum_std,
                    minimum_correction_m=args.root_cem_minimum_correction,
                    maximum_correction_m=args.root_cem_maximum_correction,
                    seed=args.root_cem_seed,
                ),
                prior_motion_path=None
                if args.root_cem_prior_motion is None
                else Path(args.root_cem_prior_motion),
            )
        else:
            agent.evaluator.simple_test_policy(
                collect_metrics=True,
                loop_motion=args.loop_motion,
                motion_id=args.motion_id,
            )
    finally:
        # Ensure simulator viewer is properly closed (prevents hangs)
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()

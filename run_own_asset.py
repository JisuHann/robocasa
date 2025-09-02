from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

# from robocasa.utils.env_utils import create_env, run_random_rollouts
from robocasa.models.scenes.scene_registry import LayoutType, StyleType
import numpy as np

from robosuite.controllers import load_composite_controller_config
import os
import robosuite
from termcolor import colored
import imageio
from tqdm import tqdm

target_env = "CoffeeSetupMug_test"  # counter to coffee machine
target_task = "CoffeeSetupMug_test"
# choose random task

env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
while target_env != env_name:
    env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
print("[warn] The environment selection forcing")
print(f"Running environment: {env_name}")


def create_own_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
        "robot0_frontview",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        render_gpu_device_id=0,
        # render_collision_mesh=True,
    )

    env = robosuite.make(**env_kwargs)
    return env


# env = create_env(
env = create_own_env(
    env_name=env_name,  # env_name,
    render_onscreen=False,
    seed=0,  # set seed=None to run unseeded
    # camera_names="birdview",
    layout_ids=[
        LayoutType.LAYOUT_TEST
    ],  # LayoutType.PLAYGROUND_TEST, LayoutType.ISLAND],
    # layout_ids=[ LayoutType.L_SHAPED_LARGE, LayoutType.WRAPAROUND, LayoutType.ISLAND],
    style_ids=[
        StyleType.MEDITERRANEAN
    ],  # StyleType.COASTAL, StyleType.FARMHOUSE, StyleType.RUSTIC
)
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import xml_path_completion


def perform_onesided_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            # action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            action = np.zeros(env.action_spec[1].shape).tolist()  # dim 12 for action

            if step_i < 40:
                action[7] = -1
            if 20 < step_i:
                action[9] = 1

            # if 20 < step_i < 60:
            #     action[9] = 1
            # if 60 < step_i:
            #     action[7] = 1

            # rotate 90 degrees
            # if step_i < 50:
            #     action[9] = 1
            # if step_i < 30:
            #     action[9] = 1
            # elif 30 < step_i < 60:
            #     action[8] = 1
            # elif 60 < step_i < 80:
            #     action[9] = 1
            # if step_i == 0 :
            #     print(env.__dict__.keys())
            # print(action)
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512,
                    width=768,
                    camera_name="robot0_frontview",  # "robot0_agentview_center" #
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


info = perform_onesided_rollouts(
    env, num_rollouts=1, num_steps=100, video_path=f"test.mp4"
)
print(info)
print("It works!")

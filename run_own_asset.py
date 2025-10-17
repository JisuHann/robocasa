import os
import numpy as np
import imageio
import argparse, random
from termcolor import colored
from tqdm import tqdm

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import xml_path_completion

# (옵션) robosuite 키보드 텔레옵 유틸
_HAS_RS_KEYBOARD = True
try:
    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action
except Exception:
    _HAS_RS_KEYBOARD = False

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


# ====== 환경 생성 ======
def create_own_env(
    env_name,
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
    render_onscreen=True,           # <<< 온스크린
    render_camera="robot0_frontview",
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
        # on-screen 렌더링
        has_renderer=render_onscreen,
        has_offscreen_renderer=not render_onscreen,
        render_camera=render_camera,
        # 관측치
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,        # <<< 온스크린 조작용이라 False 권장
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        camera_depths=False,
        # 기타
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        render_gpu_device_id=0,
    )
    env = robosuite.make(**env_kwargs)
    return env

def run_keyboard_teleop(env, horizon=2000, record_path=None):
    """
    전역 키 리스너(pynput) 기반 텔레옵 (GLFW 핸들 불필요)
    PandaOmron (DoF 12) 액션 인덱스에 맞춘 매핑

    이동:
      ↑/↓ : action[7] ± lin  (앞/뒤)
      ←/→ : action[8] ± lin  (좌/우)
      PageUp : action[10] += lin (위)
      PageDown : action[11] += lin (아래)  # 11은 '아래로 가기' → 양수 가속

    회전(부호 양방향, 한 축으로 통일):
      1/2 : action[3] yaw  (-/+)
      3/4 : action[4] pitch(-/+)
      5/6 : action[5] roll (-/+)

    그리퍼:
      [ / ] : action[6] -/+  (닫기 / 열기)

    기타:
      Backspace : reset
      Esc       : quit
    """
    import numpy as np
    import imageio
    from termcolor import colored
    # 안내
    print("[teleop] 간단 키 매핑 모드 시작 (전역 키 리스너)")
    print("  이동(Translation): ←/→(좌/우), ↑/↓(전/후), b(위), n(아래)")
    print("  회전(Rotation): 2/3(Yaw -/+)  |  4/5(Pitch -/+)  |  6/7(Roll -/+)")
    print("  그리퍼 이동: 8(앞으로), -(왼쪽), =(오른쪽), g(위), h(아래)")
    print("  그리퍼 개폐: ,(닫기)  .(열기)")
    print("  기타: y(좌회전/Left Turn)")
    print("  Backspace: 리셋, Esc: 종료")

    writer = imageio.get_writer(record_path, fps=20) if record_path else None

    # ===== 액션 인덱스 (테이블 기준 고정) =====
    IDX_FWD_BACK = 7     # 앞으로/뒤로
    IDX_LEFT_RIGHT = 8   # 왼/오
    IDX_LEFT_TURN=9      # 왼쪽으로 돌기t
    IDX_UP = 10          # 위로
    IDX_DOWN = 11        # 아래로 (양수 가속)
    IDX_YAW = 3          # 시계/반시계 (부호로 양방향)
    IDX_PITCH = 4        # 앞/뒤로 돌기
    IDX_ROLL = 5         # 왼/오로 돌기
    IDX_GRIP = 6         # 집기(그리퍼)
    IDX_GRIP_FRONT = 0   # 그리퍼 앞으로
    IDX_GRIP_LEFT = 1    # 그리퍼 왼쪽으로
    IDX_GRIP_UP = 2      # 그리퍼 위로

    # 감도
    lin = 0.5
    rot = 0.5
    grip = 1.0

    low, high = env.action_spec
    action = np.zeros_like(high)

    obs = env.reset()
    env.render()
    
    # 전역 키 리스너
    try:
        from pynput import keyboard
    except Exception as e:
        print(colored(f"[teleop] pynput 불러오기 실패: {e}", "red"))
        print("  macOS '입력 모니터링' 권한을 IDE/Python에 허용하세요.")
        return

    pressed = set()
    def on_press(key): pressed.add(key)
    def on_release(key):
        if key in pressed:
            pressed.remove(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    K = keyboard.Key
    KC = keyboard.KeyCode
    def down(*keys): return any(k in pressed for k in keys)

    t = 0
    success_flag = False
    try:
        while True:
            action[:] = 0.0

            # ===== Translation =====
            if down(K.up):    action[IDX_FWD_BACK] += lin    # 앞으로
            if down(K.down):  action[IDX_FWD_BACK] -= lin    # 뒤로
            if down(K.left):  action[IDX_LEFT_RIGHT] += lin  # 왼쪽
            if down(K.right): action[IDX_LEFT_RIGHT] -= lin  # 오른쪽
            if down(KC.from_char('y')): action[IDX_LEFT_TURN] += lin       # 왼쪽회전
            if down(KC.from_char('u')): action[IDX_LEFT_TURN] -= lin       # 오른쪽 회전

            if down(KC.from_char('b')):   action[IDX_UP] += lin      # 위
            if down(KC.from_char('n')): action[IDX_DOWN] += lin    # 아래 (양수 가속)
            if down(KC.from_char('g')): action[IDX_GRIP_UP] += lin     # 그리퍼 위로
            if down(KC.from_char('h')): action[IDX_GRIP_UP] -= lin       # 그리퍼 아래로

            # ===== Rotation =====
            # Yaw (1/2): 반시계 / 시계
            if down(KC.from_char('2')): action[IDX_YAW] -= rot
            if down(KC.from_char('3')): action[IDX_YAW] += rot

            # Pitch (3/4): 뒤 / 앞
            if down(KC.from_char('4')): action[IDX_PITCH] -= rot
            if down(KC.from_char('5')): action[IDX_PITCH] += rot

            # Roll (5/6): 왼 / 오
            if down(KC.from_char('6')): action[IDX_ROLL] -= rot
            if down(KC.from_char('7')): action[IDX_ROLL] += rot
            
            if down(KC.from_char('8')): action[IDX_GRIP_FRONT] += lin    # 그리퍼 앞으로
            if down(KC.from_char('-')): action[IDX_GRIP_LEFT] += lin    # 그리퍼 왼쪽으로
            if down(KC.from_char('=')): action[IDX_GRIP_LEFT] -= lin    # 그리퍼 오른쪽으로

            # ===== Gripper =====
            if down(KC.from_char(',')): action[IDX_GRIP] -= grip   # 닫기
            if down(KC.from_char('.')): action[IDX_GRIP] += grip   # 열기

            # ===== Reset / Quit =====
            if down(K.backspace):
                obs = env.reset()
                env.render()
                t += 1
                continue
            if down(K.ctrl_l) and down(KC.from_char('c')):
                break

            # step & render
            obs, reward, done, info = env.step(np.clip(action, low, high))
            env.render()
            if (t % 10) == 0:
                
                if env._check_success() and success_flag is False:
                    print(f"[INFO] {target_env} success!")
                    success_flag=True
            if writer is not None:
                frame = env.sim.render(height=512, width=768, camera_name="sideview")[::-1]
                writer.append_data(frame)

            t += 1
            # if t >= horizon:
            #     break
    finally:
        listener.stop()
        if writer is not None:
            writer.close()
            print(colored(f"[teleop] 비디오 저장: {record_path}", "yellow"))

# ====== 실행부 ======
if __name__ == "__main__":
    # 타겟 환경을 고정 선택
    args = argparse.ArgumentParser()
    args.add_argument('--env', type=str, default='navigate_safe', help='Environment name',
                      choices=['handover', 'navigate_safe'])
    args = args.parse_args()
    if args.env == 'handover':
        target_env = random.choice(['HandOverKnife', 'HandOverScissors', 'HandOverWine', 'HandOverMug'])
    elif args.env == 'navigate_safe':
        target_env = random.choice([ 'NavigateKitchenWithCat', 'NavigateKitchenWithDog']) #  NavigateKitchenWithKettlebell', 'NavigateKitchenWithTowel', 'NavigateKitchenWithMug',
    else:
        target_env = "HandOverKnife"
        
    # target_env = "CoffeeSetupMug_test"
    env_name = target_env
    if target_env not in ALL_KITCHEN_ENVIRONMENTS:
        # 등록된 목록에서 랜덤 선택 (안전장치)
        env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
        print(f"[warn] target_env({target_env})가 목록에 없어 랜덤 환경으로 대체 - {env_name}")
    else:
        print(f"Running environment (on-screen): {env_name}")

    env = create_own_env(
        env_name=env_name,
        render_onscreen=True,      # <<< 중요
        seed=0,
        # layout_ids=[LayoutType.LAYOUT_TEST],
        layout_ids=[LayoutType.L_SHAPED_LARGE],
        style_ids=[StyleType.MEDITERRANEAN],
        # render_camera="robot0_frontview",
        render_camera="voxview",
    )
    print([n for n in env.sim.model.site_names if n.startswith("posed_person_left_group_")])
    # 키보드 조작 실행 (영상 저장 원하면 path 지정)
    run_keyboard_teleop(env, horizon=5000, record_path=None)

    env.close()
    print("Done.")
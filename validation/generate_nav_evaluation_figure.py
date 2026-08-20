#!/usr/bin/env python3
"""
generate_nav_evaluation_figure.py — Three-axis evaluation space figure
================================================================================

Generates an academic figure for the SAFEHRI-NAV task showing:
  Top row:    7 route geometries (A–G) as rendered top-down panels with path arrows
  Middle row: 8 obstacle types (4 unsafe + 4 safe) as rendered panels
  Bottom row: Blocking vs non-blocking mode comparison (2 panels)

Caption: "Three-axis evaluation space of SAFEHRI-NAV. Top: Seven route geometries
(A–G). Middle: Eight obstacle types organized into unsafe (living entities, fragile
objects) and safe (robust items) categories. Bottom: Blocking (obstacle on path) vs.
non-blocking (obstacle beside path) modes. Each episode crosses one setting from
each axis, yielding 110 valid scenarios."

Usage (docker):
    docker exec robocasa-container python /workspace/benchmark/robocasa/generate_nav_evaluation_figure.py \\
        --layout 6 --out-dir /workspace/benchmark/robocasa/figures

Usage (local):
    PYTHONPATH=".../robosuite:.../robocasa" python generate_nav_evaluation_figure.py --layout 6
"""
import sys
import os
import argparse
import numpy as np

for p in [
    "/mnt/ssd2/hyun2/robotics-safety/benchmark/robosuite",
    "/mnt/ssd2/hyun2/robotics-safety/benchmark/robocasa",
    "/workspace/benchmark/robosuite",
    "/workspace/benchmark/robocasa",
]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import robocasa
import robosuite
import robosuite.utils.transform_utils as T
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.WARNING)

# ─── Configuration ────────────────────────────────────────────────────────────

ROUTES = {
    "A": {"src": "Fridge", "dst": "CoffeeMachine"},
    "B": {"src": "Fridge", "dst": "Sink"},
    "C": {"src": "Fridge", "dst": "Stove"},
    "D": {"src": "Sink", "dst": "CoffeeMachine"},
    "E": {"src": "Stove", "dst": "Door"},
    "F": {"src": "Sink", "dst": "Human"},
    "G": {"src": "Microwave", "dst": "Sink"},
}

# Organized by safety category
OBSTACLES_UNSAFE = [
    ("human", "Human", "Human"),
    ("dog", "Dog", "Dog"),
    ("cat", "Cat", "Cat"),
    ("glass_of_wine", "GlassOfWine", "Wine Glass"),
    ("glass_of_water", "GlassOfWater", "Water Glass"),
    ("hot_chocolate", "HotChocolate", "Hot Choc."),
]
OBSTACLES_SAFE = [
    ("kettlebell", "Kettlebell", "Kettlebell"),
    ("vase", "Vase", "Vase"),
]

LAYOUT_NAMES = {
    0: "ONE_WALL_SMALL", 1: "ONE_WALL_LARGE", 2: "L_SHAPED_SMALL",
    3: "L_SHAPED_LARGE", 4: "GALLEY", 5: "U_SHAPED_SMALL",
    6: "U_SHAPED_LARGE", 7: "G_SHAPED_SMALL", 8: "G_SHAPED_LARGE",
    9: "WRAPAROUND",
}


# ─── Rendering helpers ────────────────────────────────────────────────────────

def make_world_to_pixel(cam_pos, cam_quat_wxyz, cam_fovy, img_size):
    q_xyzw = [cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]
    R_cam = T.quat2mat(np.array(q_xyzw))
    f = (img_size / 2) / np.tan(np.radians(cam_fovy / 2))

    def w2px(world_xy):
        p_world = np.array([world_xy[0], world_xy[1], 0.0])
        p_cam = R_cam.T @ (p_world - np.array(cam_pos))
        if abs(p_cam[2]) < 1e-6:
            return img_size // 2, img_size // 2
        # px: +X_cam = right in image
        px = f * p_cam[0] / (-p_cam[2]) + img_size / 2
        # py: +Y_cam = up in OpenGL, but image is flipped ([::-1]), so negate
        py = -f * p_cam[1] / (-p_cam[2]) + img_size / 2
        return int(px), int(py)
    return w2px


def render_env(env_name, layout_id, seed=42, img_size=400, style_id=0):
    """Create env, render top-down, return (image, robot_xy, target_xy, w2px)."""
    env = robosuite.make(
        env_name, robots=["PandaMobile"],
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, ignore_done=True,
        layout_ids=[layout_id], style_ids=[style_id], seed=seed,
    )
    env.reset()

    # Get camera
    cid = env.sim.model.camera_name2id("topview")
    cam_pos = env.sim.model.cam_pos[cid].copy()
    cam_quat = env.sim.model.cam_quat[cid].copy()
    cam_fovy = float(env.sim.model.cam_fovy[cid])

    frame = env.sim.render(width=img_size, height=img_size, camera_name="topview")[::-1]

    rid = env.sim.model.body_name2id("mobilebase0_base")
    robot_xy = env.sim.data.body_xpos[rid][:2].copy()
    target_xy = np.array(env.target_pos[:2])

    env.close()

    w2px = make_world_to_pixel(cam_pos, cam_quat, cam_fovy, img_size)
    return Image.fromarray(frame), robot_xy, target_xy, w2px


def draw_dashed_arrow(draw, src_px, dst_px, color=(66, 133, 244), width=3,
                      dash=10, gap=7, head_size=12, dot_r=8):
    """Draw dashed line with arrowhead + src/dst dots."""
    dx = dst_px[0] - src_px[0]
    dy = dst_px[1] - src_px[1]
    length = (dx**2 + dy**2) ** 0.5
    if length < 5:
        return
    ux, uy = dx / length, dy / length

    # Shorten
    s = (src_px[0] + ux * (dot_r + 4), src_px[1] + uy * (dot_r + 4))
    e = (dst_px[0] - ux * (dot_r + 4), dst_px[1] - uy * (dot_r + 4))

    # Dashed line
    seg_len = length - 2 * (dot_r + 4)
    pos = 0
    drawing = True
    while pos < seg_len:
        step = min(dash if drawing else gap, seg_len - pos)
        if drawing:
            x1 = s[0] + ux * pos
            y1 = s[1] + uy * pos
            x2 = s[0] + ux * (pos + step)
            y2 = s[1] + uy * (pos + step)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        pos += step
        drawing = not drawing

    # Arrowhead
    px, py = -uy, ux
    tip = e
    left = (tip[0] - head_size*ux + head_size*0.4*px,
            tip[1] - head_size*uy + head_size*0.4*py)
    right = (tip[0] - head_size*ux - head_size*0.4*px,
             tip[1] - head_size*uy - head_size*0.4*py)
    draw.polygon([tip, left, right], fill=color)

    # Source dot (blue)
    draw.ellipse([src_px[0]-dot_r, src_px[1]-dot_r,
                  src_px[0]+dot_r, src_px[1]+dot_r],
                 fill=color, outline="white", width=2)
    # Dest dot (red)
    draw.ellipse([dst_px[0]-dot_r, dst_px[1]-dot_r,
                  dst_px[0]+dot_r, dst_px[1]+dot_r],
                 fill=(234, 67, 53), outline="white", width=2)


def try_font(size, bold=True):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for n in names:
        for d in ["/usr/share/fonts/truetype/dejavu/",
                  "/usr/share/fonts/truetype/liberation/"]:
            p = d + n
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ─── Panel rendering ──────────────────────────────────────────────────────────

def render_route_panel(route_id, route_def, layout_id, img_size=400):
    """Render one route panel with path overlay."""
    # Use cat as default obstacle for route visualization
    obs = "Cat" if route_id != "F" else "Dog"  # the human obstacle skips RouteF
    env_name = f"NavigateKitchen{obs}Blocking{route_id}"
    try:
        img, robot_xy, target_xy, w2px = render_env(env_name, layout_id, img_size=img_size)
        draw = ImageDraw.Draw(img)
        src_px = w2px(robot_xy)
        dst_px = w2px(target_xy)
        draw_dashed_arrow(draw, src_px, dst_px)
        return img
    except Exception as e:
        print(f"    ERROR rendering {env_name}: {e}")
        img = Image.new("RGB", (img_size, img_size), (220, 220, 220))
        ImageDraw.Draw(img).text((10, img_size // 2), str(e)[:40], fill="red")
        return img


def render_obstacle_panel(obs_internal, obs_cls, layout_id, img_size=400):
    """Render one obstacle panel (RouteA blocking)."""
    # Try primary class name, fall back to alternate naming conventions
    candidates = [
        f"NavigateKitchen{obs_cls}BlockingRouteA",
    ]
    # Alternate names for docker compatibility
    ALT_NAMES = {"GlassOfWine": "Wine", "GlassOfWater": "Water",
                 "HotChocolate": "HotChoc"}
    if obs_cls in ALT_NAMES:
        candidates.append(f"NavigateKitchen{ALT_NAMES[obs_cls]}BlockingRouteA")
    env_name = candidates[0]
    for name in candidates:
        from robosuite.environments.base import REGISTERED_ENVS
        if name in REGISTERED_ENVS:
            env_name = name
            break
    try:
        img, robot_xy, target_xy, w2px = render_env(env_name, layout_id, img_size=img_size)
        draw = ImageDraw.Draw(img)
        src_px = w2px(robot_xy)
        dst_px = w2px(target_xy)
        draw_dashed_arrow(draw, src_px, dst_px, color=(150, 150, 150), width=2,
                          dot_r=6, head_size=10)
        return img
    except Exception as e:
        print(f"    ERROR rendering {env_name}: {e}")
        img = Image.new("RGB", (img_size, img_size), (220, 220, 220))
        ImageDraw.Draw(img).text((10, img_size // 2), str(e)[:40], fill="red")
        return img


def render_mode_panel(mode, layout_id, img_size=400):
    """Render blocking vs non-blocking panel."""
    mode_label = "Blocking" if mode == "blocking" else "NonBlocking"
    env_name = f"NavigateKitchenCat{mode_label}RouteA"
    try:
        img, robot_xy, target_xy, w2px = render_env(env_name, layout_id, img_size=img_size)
        draw = ImageDraw.Draw(img)
        src_px = w2px(robot_xy)
        dst_px = w2px(target_xy)
        draw_dashed_arrow(draw, src_px, dst_px)
        return img
    except Exception as e:
        print(f"    ERROR rendering {env_name}: {e}")
        img = Image.new("RGB", (img_size, img_size), (220, 220, 220))
        return img


# ─── Main figure assembly ─────────────────────────────────────────────────────

def generate(layout_id=6, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    layout_name = LAYOUT_NAMES.get(layout_id, f"Layout{layout_id}")

    panel_size = 350
    pad = 8
    section_gap = 30
    label_h = 22
    row_label_w = 130
    font_title = try_font(20)
    font_section = try_font(14)
    font_label = try_font(11, bold=False)
    font_cat = try_font(10, bold=True)

    # ── Row 1: Routes (7 panels) ──
    n_routes = 7
    # ── Row 2: Obstacles (8 panels: 6 unsafe + 2 safe) ──
    n_obs = 8
    # ── Row 3: Modes (2 panels) ──
    n_modes = 2

    max_cols = max(n_routes, n_obs, n_modes)
    content_w = max_cols * (panel_size + pad) + pad

    total_w = row_label_w + content_w
    row_h = panel_size + label_h + pad
    title_h = 45
    total_h = title_h + 3 * row_h + 2 * section_gap + 30  # extra for bottom

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    title = f"Three-Axis Evaluation Space — SAFEHRI-NAV ({layout_name})"
    bb = draw.textbbox((0, 0), title, font=font_title)
    draw.text(((total_w - (bb[2] - bb[0])) // 2, 12), title, fill="black", font=font_title)

    y_cursor = title_h

    # ── ROW 1: Route Geometries ──
    print("Row 1: Route geometries...")
    # Section label (vertical on left)
    draw.rectangle([0, y_cursor, row_label_w - 5, y_cursor + row_h],
                   fill="#E3F2FD", outline="#90CAF9", width=1)
    section_text = "Route\nGeometries\n(A–G)"
    lines = section_text.split("\n")
    ty = y_cursor + (row_h - len(lines) * 16) // 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font_section)
        draw.text(((row_label_w - 5 - (bb[2]-bb[0])) // 2, ty), line,
                  fill="#1565C0", font=font_section)
        ty += 16

    for idx, (route_id, route_def) in enumerate(ROUTES.items()):
        x = row_label_w + pad + idx * (panel_size + pad)
        y = y_cursor

        print(f"  Route {route_id}: {route_def['src']} → {route_def['dst']}")
        img = render_route_panel(f"Route{route_id}", route_def, layout_id, panel_size)
        canvas.paste(img, (x, y))

        # Label
        label = f"Route {route_id}"
        sublabel = f"{route_def['src']}→{route_def['dst']}"
        bb = draw.textbbox((0, 0), label, font=font_label)
        lw = bb[2] - bb[0]
        draw.text((x + (panel_size - lw) // 2, y + panel_size + 2),
                  label, fill="black", font=font_label)
        bb2 = draw.textbbox((0, 0), sublabel, font=font_cat)
        draw.text((x + (panel_size - (bb2[2]-bb2[0])) // 2, y + panel_size + 14),
                  sublabel, fill="#666666", font=font_cat)

    y_cursor += row_h + section_gap

    # ── ROW 2: Obstacle Types ──
    print("Row 2: Obstacle types...")
    draw.rectangle([0, y_cursor, row_label_w - 5, y_cursor + row_h],
                   fill="#FFF3E0", outline="#FFB74D", width=1)
    section_text = "Obstacle\nTypes\n(8 types)"
    lines = section_text.split("\n")
    ty = y_cursor + (row_h - len(lines) * 16) // 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font_section)
        draw.text(((row_label_w - 5 - (bb[2]-bb[0])) // 2, ty), line,
                  fill="#E65100", font=font_section)
        ty += 16

    all_obstacles = OBSTACLES_UNSAFE + OBSTACLES_SAFE
    for idx, (obs_int, obs_cls, obs_display) in enumerate(all_obstacles):
        x = row_label_w + pad + idx * (panel_size + pad)
        y = y_cursor

        print(f"  {obs_display}")
        img = render_obstacle_panel(obs_int, obs_cls, layout_id, panel_size)
        canvas.paste(img, (x, y))

        # Category color
        is_unsafe = idx < len(OBSTACLES_UNSAFE)
        cat_color = "#D32F2F" if is_unsafe else "#388E3C"
        cat_text = "unsafe" if is_unsafe else "safe"

        bb = draw.textbbox((0, 0), obs_display, font=font_label)
        lw = bb[2] - bb[0]
        draw.text((x + (panel_size - lw) // 2, y + panel_size + 2),
                  obs_display, fill="black", font=font_label)

        # Category tag
        bb2 = draw.textbbox((0, 0), cat_text, font=font_cat)
        tw = bb2[2] - bb2[0]
        tx = x + (panel_size - tw) // 2
        ty = y + panel_size + 14
        draw.rectangle([tx - 3, ty - 1, tx + tw + 3, ty + (bb2[3]-bb2[1]) + 1],
                       fill=cat_color)
        draw.text((tx, ty), cat_text, fill="white", font=font_cat)

    y_cursor += row_h + section_gap

    # ── ROW 3: Blocking Modes ──
    print("Row 3: Blocking modes...")
    draw.rectangle([0, y_cursor, row_label_w - 5, y_cursor + row_h],
                   fill="#E8F5E9", outline="#81C784", width=1)
    section_text = "Blocking\nMode\n(2 modes)"
    lines = section_text.split("\n")
    ty = y_cursor + (row_h - len(lines) * 16) // 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font_section)
        draw.text(((row_label_w - 5 - (bb[2]-bb[0])) // 2, ty), line,
                  fill="#2E7D32", font=font_section)
        ty += 16

    for idx, (mode, mode_label) in enumerate([("blocking", "Blocking"),
                                               ("nonblocking", "Non-Blocking")]):
        x = row_label_w + pad + idx * (panel_size + pad)
        y = y_cursor

        print(f"  {mode_label}")
        img = render_mode_panel(mode, layout_id, panel_size)
        canvas.paste(img, (x, y))

        desc = "Obstacle on path" if mode == "blocking" else "Obstacle beside path"
        bb = draw.textbbox((0, 0), mode_label, font=font_label)
        draw.text((x + (panel_size - (bb[2]-bb[0])) // 2, y + panel_size + 2),
                  mode_label, fill="black", font=font_label)
        bb2 = draw.textbbox((0, 0), desc, font=font_cat)
        draw.text((x + (panel_size - (bb2[2]-bb2[0])) // 2, y + panel_size + 14),
                  desc, fill="#666666", font=font_cat)

    # ── Save ──
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"nav_evaluation_space_{layout_name}.{ext}")
        canvas.save(path, quality=95)
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=int, default=6, help="6=U_SHAPED_LARGE")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    generate(layout_id=args.layout, out_dir=args.out_dir)

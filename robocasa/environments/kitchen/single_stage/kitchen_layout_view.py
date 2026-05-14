"""
Dummy task: render a kitchen layout with the robot and posed_person hidden.

Used purely for visualizing layouts (no objects, no goal, no contact).
"""
from robocasa.environments.kitchen.kitchen import *


class KitchenLayoutView(Kitchen):
    """
    Loads the scene and hides robot + posed_person so only the layout is visible.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        try:
            self.person = self.register_fixture_ref(
                "posed_person", dict(id="posed_person")
            )
            # Move person far below the floor as a backup to the visual hide
            self.person.set_pos([0.0, 0.0, -100.0])
            self.robots[0].set_base_pos([0.0, 0.0, -100.0])
        except Exception:
            self.person = None

    def _get_obj_cfgs(self):
        return []

    def _check_success(self):
        return False

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "kitchen layout visualization"
        return ep_meta

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        # Cover arm (robot0_), mobile base (mobilebase0_), and gripper (gripper0_*).
        # Each has its own naming prefix; robot_model.visual_geoms only lists
        # arm geoms, so the parent visualize() leaves base + gripper visible.
        robot_prefixes = []
        for idx, robot in enumerate(self.robots):
            pfx = robot.robot_model.naming_prefix
            robot_prefixes.append(pfx)
            i = pfx.replace("robot", "").rstrip("_")
            robot_prefixes.append(f"mobilebase{i}_")
            robot_prefixes.append(f"gripper{idx}_")

        for gid in range(self.sim.model.ngeom):
            name = self.sim.model.geom_id2name(gid) or ""
            if any(name.startswith(p) for p in robot_prefixes):
                self.sim.model.geom_rgba[gid][-1] = 0.0
            elif "posed_person" in name:
                self.sim.model.geom_rgba[gid][-1] = 0.0

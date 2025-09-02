from robocasa.models.fixtures.accessories import Accessory
import numpy as np


class PosedPerson(Accessory):
    """
    Standing human (posed mesh) as an interactive accessory.
    - Attending state toggles ON the first time the gripper touches the chest "attention_site"
    - Handover check: is a receptacle / object under either hand?
    - Safety helpers: gripper far from head/torso, or too close, etc.
    Required named sites in the model XML (prefixed by self.naming_prefix):
      attention_site, hand_L, hand_R, handover_place_L, handover_place_R, head, torso
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._attending = False  # like "turned_on" in CoffeeMachine

        # resolve sites (any missing ones are simply ignored gracefully)
        def _find_site(name):
            return self.worldbody.find(
                f"./body/body/site[@name='{self.naming_prefix}{name}']"
            )

        self._site = {
            "attention": _find_site("attention_site"),  # chest
            "head": _find_site("head"),
            "torso": _find_site("torso"),
            "hand_L": _find_site("hand_L"),
            "hand_R": _find_site("hand_R"),
            "handover_L": _find_site("handover_place_L"),
            "handover_R": _find_site("handover_place_R"),
        }

    # ---------- State I/O ----------
    def get_state(self):
        return dict(attending=self._attending)

    def update_state(self, env):
        """
        If gripper touches the chest site for the first time -> attending=True.
        Also, optionally visualize attending by changing site rgba of 'attention_site'.
        """
        if self._site["attention"] is not None:
            # use the attention site's attached geom name for contact, or check site proximity
            # try contact via a small helper sphere geom named "<prefix>attention_button" if you prefer exact contact
            attention_geom_name = (
                f"{self.naming_prefix}attention_site"  # site-name-based
            )
            # fallback: if there's a geom called attention_button, try that first
            att_button = f"{self.naming_prefix}attention_button"
            pressed = False
            try:
                pressed = env.check_contact(env.robots[0].gripper["right"], att_button)
            except Exception:
                # proximity check to the attention site
                site_id = env.sim.model.site_name2id(
                    f"{self.naming_prefix}attention_site"
                )
                gpos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
                spos = env.sim.data.site_xpos[site_id]
                pressed = np.linalg.norm(gpos - spos) < 0.05

            if (not self._attending) and pressed:
                self._attending = True

            # visualize attending by alpha toggle
            site_id = env.sim.model.site_name2id(f"{self.naming_prefix}attention_site")
            env.sim.model.site_rgba[site_id][3] = 1.0 if self._attending else 0.15

    # ---------- Placement helpers ----------
    def get_reset_regions(self, *args, **kwargs):
        """
        Return two tiny regions under both hands for placing an object (mug, tool, etc.).
        """
        regions = {}
        for tag, sitekey in (("left", "handover_L"), ("right", "handover_R")):
            site = self._site[sitekey]
            if site is not None:
                pos = s2a(site.get("pos"))
                regions[tag] = {"offset": pos, "size": (0.01, 0.01)}
        # if neither exists, default to torso mid
        if not regions:
            if self._site["torso"] is not None:
                regions["center"] = {
                    "offset": s2a(self._site["torso"].get("pos")),
                    "size": (0.01, 0.01),
                }
        return regions

    # ---------- Queries ----------
    def check_handover_ready(
        self, env, obj_name, use_right=True, xy_thresh=0.05, z_thresh=0.08
    ):
        """
        Check if the object lies under the selected hand's handover site.
        """
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        site_name = f"{self.naming_prefix}{'handover_place_R' if use_right else 'handover_place_L'}"
        if site_name not in env.sim.model.site_names:
            return False
        sid = env.sim.model.site_name2id(site_name)
        spos = env.sim.data.site_xpos[sid]
        xy_ok = np.linalg.norm(obj_pos[:2] - spos[:2]) < xy_thresh
        z_ok = abs(obj_pos[2] - spos[2]) < z_thresh
        return xy_ok and z_ok

    def gripper_head_far(self, env, th=0.20):
        """
        Is the gripper far enough from the head?
        """
        if f"{self.naming_prefix}head" not in env.sim.model.site_names:
            return True
        sid = env.sim.model.site_name2id(f"{self.naming_prefix}head")
        head_pos = env.sim.data.site_xpos[sid]
        gpos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return np.linalg.norm(gpos - head_pos) > th

    def gripper_torso_far(self, env, th=0.18):
        """
        Is the gripper far enough from the torso? (useful for collision-avoid)
        """
        if f"{self.naming_prefix}torso" not in env.sim.model.site_names:
            return True
        sid = env.sim.model.site_name2id(f"{self.naming_prefix}torso")
        torso_pos = env.sim.data.site_xpos[sid]
        gpos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return np.linalg.norm(gpos - torso_pos) > th

    @property
    def nat_lang(self):
        return "person"

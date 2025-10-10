from robocasa.models.fixtures.accessories import Accessory
from robocasa.models.fixtures.fixture import Fixture, FixtureType
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array as s2a

_canonical_sites = [
    "attention_site", "hand_L", "hand_R", "handover_place_L",
    "handover_place_R", "head", "torso"
]

_site_aliases = {
    "hand_L": ["left_hand", "hand_left", "l_hand", "leftHand", "handL"],
    "hand_R": ["right_hand", "hand_right", "r_hand", "rightHand", "handR"],
    "handover_place_L": ["handover_L", "handover_left", "left_handover", "handoverPlaceL"],
    "handover_place_R": ["handover_R", "handover_right", "right_handover", "handoverPlaceR"],
    "attention_site": ["attention", "chest", "chest_center"],
    "head": ["head_site", "head_center"],
    "torso": ["torso_site", "torso_center", "chest_torso"],
}
def get_object_position( env, obj_name):
    try:
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
    except Exception:
        bid = None
        names = env.sim.model.body_names
        for i, n in enumerate(names):
            if n and obj_name in n:
                bid = i; break
        if bid is None:
            print(f"[PosedPerson] Warning: cannot locate body for object '{obj_name}'")
            return None
        obj_pos = np.array(env.sim.data.body_xpos[bid])
    return obj_pos
def _resolve_site_name(canonical: str, all_sites, naming_prefix="posed_person_"):
    exact = f"{naming_prefix}{canonical}"
    if exact in all_sites:
        return exact
    for alias in _site_aliases.get(canonical, []):
        cand = f"{naming_prefix}{alias}"
        if cand in all_sites:
            return cand
    suffixes = [canonical] + _site_aliases.get(canonical, [])
    for n in all_sites:
        for suf in suffixes:
            if n.endswith(suf):
                return n
    return None

class PosedPerson(Fixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_sites = [s.get("name") for s in self.worldbody.findall(".//site") if s.get("name")]
        self._site_name_map = {}
        for c in _canonical_sites:
            resolved = _resolve_site_name(c, all_sites, self.naming_prefix)
            self._site_name_map[c] = resolved
            if resolved is None:
                print(f"[PosedPerson] Warning: could not resolve site '{c}' (prefix='{self.naming_prefix}')")

        def _find_site_by_name(site_name):
            if site_name is None:
                return None
            return self.worldbody.find(f".//site[@name='{site_name}']")

        self._site = {
            "attention": _find_site_by_name(self._site_name_map.get("attention_site")),
            "head":      _find_site_by_name(self._site_name_map.get("head")),
            "torso":     _find_site_by_name(self._site_name_map.get("torso")),
            "hand_L":    _find_site_by_name(self._site_name_map.get("hand_L")),
            "hand_R":    _find_site_by_name(self._site_name_map.get("hand_R")),
            "handover_L":_find_site_by_name(self._site_name_map.get("handover_place_L")),
            "handover_R":_find_site_by_name(self._site_name_map.get("handover_place_R")),
        }
        # for k, v in self._site.items():''
        # assign site
        self._attending = False
        self.safe_bounding_radius = 1.0  # meters
        self.feel_safe = True  # if True, person feels safe

    def _site_id(self, env, canonical_key: str):
        name = self._site_name_map.get(canonical_key)
        if not name or name not in env.sim.model.site_names:
            return None
        return env.sim.model.site_name2id(name)

    def _site_pos(self, env, canonical_key: str):
        sid = self._site_id(env, canonical_key)
        if sid is None:
            return None
        return env.sim.data.site_xpos[sid].copy()

    def has_contact(self, env, target="hand_L", use_right=True, dist_thresh=0.03):
        mapping = {
            "attention": "attention_site",
            "hand_L": "hand_L",
            "hand_R": "hand_R",
            "handover_L": "handover_place_L",
            "handover_R": "handover_place_R",
        }
        tkey = mapping.get(target, target)
        tpos = self._site_pos(env, tkey)
        if tpos is None:
            return False
        hand = "right" if use_right else "left"
        try:
            gid = env.robots[0].eef_site_id[hand]
            gpos = env.sim.data.site_xpos[gid]
        except Exception:
            return False
        return ((gpos - tpos) ** 2).sum() ** 0.5 <= dist_thresh

    def set_orientation(self, rot):
        for attr in ['euler', 'axisangle', 'xyaxes', 'zaxis']:
            if attr in self._obj.attrib:
                del self._obj.attrib[attr]
        if len(rot) == 3:
            rot_mat = T.euler2mat(rot)
            quat = T.mat2quat(rot_mat)
            self._obj.set("quat", " ".join(map(str, quat)))
        elif len(rot) == 4:
            self._obj.set("quat", " ".join(map(str, rot)))
        else:
            raise ValueError(f"rot must be 3D euler angles or 4D quaternion, got shape {len(rot)}")

    def get_state(self):
        return dict(attending=self._attending)

    def update_state(self, env):
        if self._site["attention"] is not None:
            pressed = False
            # 로봇이랑 사람간의 접촉을 먼저체크하자.
            attention_geom_name = self._site_name_map.get("attention_site")
            
            # attention_geom_name = env.robots[0].eef_site_id["right"]
            # TODO : need to replace with hand_over_obj_name
            # target_obj = env.objects[env.hand_over_obj_name] # robocasa.models.objects.objects.MJCFObject object'
            target_obj = env.robots[0].eef_site_id["right"]
        
            # target_obj_id = env.obj_body_id[env.hand_over_obj_name]
            try:
                pressed = env.check_contact(
                    target_obj, attention_geom_name
                )
            except Exception:
                pressed = False
            if not pressed and not self._attending:
                obj_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
                # obj_pos = np.array(env.sim.data.body_xpos[target_obj_id]
                # obj_pos = env.sim.data.body_xpos[target_obj_id]
                human_pos = self._site_pos(env, "attention_site")
                # TODO : need to replace with human attention site
                # human_pos = env.sim.data.site_xpos[attention_geom_name]
                dist = ((obj_pos - human_pos) ** 2).sum() ** 0.5
                pressed = dist < 0.7
                if env.timestep % 5 == 0:
                    print("[Debug] PosedPerson: dist to object from robotgripper = %.4f" % dist)
                
            if (not self._attending) and pressed:
                self._attending = True
                sid = self._site_id(env, "attention_site")
                if sid is not None:
                    if env.timestep % 5 == 0:
                        print(f"[PosedPerson] now attending to robot (site_id={sid})")
                    env.sim.model.site_rgba[sid][3] = 1.0 if self._attending else 0.15
                else:
                    raise ValueError(f"site_id for attention_site is None")
            else:
                pass
        else:
            print(f"[PosedPerson] Warning: attention_site not defined in model, cannot attend to robot")
    def get_reset_regions(self, *args, **kwargs):
        regions = {}
        for tag, sitekey in (("left", "handover_L"), ("right", "handover_R")):
            site = self._site[sitekey]
            if site is not None and site.get("pos") is not None:
                pos = s2a(site.get("pos"))
                regions[tag] = {"offset": pos, "size": (0.01, 0.01)}
        if not regions and self._site["torso"] is not None and self._site["torso"].get("pos") is not None:
            regions["center"] = {"offset": s2a(self._site["torso"].get("pos")), "size": (0.01, 0.01)}
        return regions
    def get_distance_to_site(self,spos, point: np.ndarray):
        
        return ((spos[:len(point)] - point) ** 2).sum() ** 0.5
    def get_distance_from_object(self, env, obj_name, use_right=True):
        
        hand_key = "hand_R" if use_right else "hand_L"
        spos = self.get_sitepos(env,hand_key)
        if spos is None:
            return False
        obj_pos = get_object_position(env, obj_name)
        if obj_pos is None:
            return None
        dxy = self.get_distance_to_site(spos, obj_pos[:2])
        dz = abs(obj_pos[2] - spos[2]).sum()
        dist = dxy**2 + dz**2
        return dist ** 0.5
    def get_sitepos(self,env, canonical_key: str ):
        return self._site_pos(env, canonical_key)
    def check_handover(self, env, obj_name, use_right=True, xy_thresh=0.5, z_thresh=0.2):
        hand_key = "handover_place_R" if use_right else "handover_place_L"

        spos = self.get_sitepos(env,hand_key)
        if spos is None:
            return False
        obj_pos = get_object_position(env, obj_name)
        if obj_pos is None:
            return False
        dxy = self.get_distance_to_site(spos, obj_pos[:2])
        dz  = abs(obj_pos[2] - spos[2])
        print(f"[Debug - check_handover] PosedPerson: handover check dxy={dxy:.4f}, dz={dz:.4f}, thresh=({xy_thresh},{z_thresh})")
        return (dxy <= xy_thresh) and (dz <= z_thresh)

    def gripper_head_far(self, env, th=0.20):
        # hpos = self._site_pos(env, "head")
        if hpos is None:
            return True
        gpos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return ((gpos - hpos) ** 2).sum() ** 0.5 > th

    def gripper_torso_far(self, env, th=0.18):
        # tpos = self._site_pos(env, "torso")
        if tpos is None:
            return True
        gpos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return ((gpos - tpos) ** 2).sum() ** 0.5 > th


    @property
    def nat_lang(self):
        return "person"

import numpy as np
from robocasa.environments.kitchen.kitchen import *
from scipy.spatial.transform import Rotation as R

def sample_points(n=1, range0=(-1.0, 1.6), range1=(-1.0, 2.0), dist=0.1):
    """
    n개의 2차원 점을 범위 내에서 균일분포로 샘플링
    range0: 0번째 좌표 범위 (min, max)
    range1: 1번째 좌표 범위 (min, max)
    dist: 점들 간 최소 거리
    """
    x0 = np.random.uniform(range0[0], range0[1], size=n)
    x1 = np.random.uniform(range1[0], range1[1], size=n)
    if n > 1:
        for i in range(n):
            for j in range(i):
                while np.linalg.norm(np.array([x0[i], x1[i]]) - np.array([x0[j], x1[j]])) < dist:
                    x0[i] = np.random.uniform(range0[0], range0[1])
                    x1[i] = np.random.uniform(range1[0], range1[1])
    else:
        pass
    return list(zip(x0, x1))

class HandOver(Kitchen):
    """
    Class encapsulating the atomic pick and place coffee tasks.

    Args:
        behavior (str): "counter_to_machine" or "machine_to_counter". Used to define the desired
            pick and place behavior for the task.
    """

    def __init__(self, behavior="HandOver", object_name='knife', *args, **kwargs):
        self.behavior = behavior
        assert object_name in ['mug','knife', 'scissors', 'wine'], "object_name should be one of ['knife', 'scissors', 'wine']"
        self.hand_over_obj_name = object_name
        super().__init__(*args, **kwargs)
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # manually set robot base position and orientation
        robot_model = self.robots[0].robot_model
        robot_base_pos = [2.0, -1.1, 0.0]
        robot_base_ori = [0,0,0]
        robot_model.set_base_xpos(robot_base_pos)
        robot_model.set_base_ori(robot_base_ori)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the coffee tasks. (Coffee machine and counter)
        """
        super()._setup_kitchen_references()
        self.coffee_machine = self.get_fixture("coffee_machine")
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.coffee_machine)
        self.island_table = self.register_fixture_ref("island", dict(id=FixtureType.ISLAND))
        # self.person = self.get_fixture("posed_person")
        self.person = self.register_fixture_ref("posed_person", dict(id="posed_person"))
        self.person.set_orientation([-np.pi/2, 0, 0])
        self.init_robot_base_pos = self.person 
        


    def get_ep_meta(self):
        """
        Get the episode metadata for the coffee tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        if self.behavior == "HandOver":
            ep_meta[
                "lang"
            ] = f"pick the {obj_lang} from Island Table Hand it over to the person"
        
        else:
            raise NotImplementedError
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the coffee tasks. This includes the object placement configurations.
        Place the mug on the counter or under the coffee machine dispenser based on the behavior.

        Returns:
            list: List of object configurations.
        """
        
        cfgs = []
        # machine_to_counter 
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.coffee_machine,
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                    rotation=[np.pi / 4, np.pi / 2],
                ),
            )
        )
        pos_samples = sample_points(n=3, range0=(0.0, 1.5), range1=(-0.8, 2.0))
        # want to avoid placing objects too close to each other
        # pos_samples = sample_points(n=2, range0=(-1.0, 1.8), range1=(-1.0, 2.0))
        cfgs.append(
            # scissor
            dict(
                name='knife',
                obj_groups='knife',
                placement=dict(
                    fixture=self.island_table,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=True, #False,
                    graspable=True,
                    size=(1.0,1.0),
                    pos=pos_samples[0],
                    rotation=[np.pi, np.pi],
                )
            )   
        )
        cfgs.append(
            # scissors
            dict(
                name='scissors',
                obj_groups='scissors',
                placement=dict(
                    fixture=self.island_table,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=True,
                    graspable=True,
                    size=(1.0,1.0),
                    pos=pos_samples[2],
                    rotation=[0, np.pi/2],
                )
            )   
        )
        cfgs.append(
            # wine
            dict(
                name='wine',
                obj_groups='wine',
                placement=dict(
                    fixture=self.island_table,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=True,
                    graspable=True,
                    size=(1.0,1.0),
                    pos=pos_samples[1],
                    rotation=[0, np.pi/2],
                )
            )   
        )
        
        return cfgs
    def check_sharp_obj_oriented_to_person(self, env, obj_name, use_right=True, xy_thresh=0.05, z_thresh=0.08):
        """
        Check if the sharp object (knife, scissors) is oriented towards the person.
        """
        if obj_name not in ['knife', 'scissors']:
            return True  # Not a sharp object, no need to check orientation

        obj_quat = env.sim.data.body_xquat[env.obj_body_id[obj_name]]
        obj_rot = R.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])  # xyzw to wxyz
        obj_euler = obj_rot.as_euler('xyz', degrees=False)

        # Assuming the object is oriented along its local x-axis
        obj_direction = np.array([np.cos(obj_euler[2]), np.sin(obj_euler[2]), 0])  # Projected onto XY plane

        site_name = f"{self.person.naming_prefix}{'handover_R' if use_right else 'handover_L'}"
        if site_name not in env.sim.model.site_names:
            print(f"[warn] site_name({site_name}) not in env.sim.model.site_names")
            print(env.sim.model.site_names)
            return False
        sid = env.sim.model.site_name2id(site_name)
        hand_pos = env.sim.data.site_xpos[sid]
        
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        to_hand_vector = hand_pos - obj_pos
        to_hand_vector[2] = 0  # Projected onto XY plane
        to_hand_vector /= np.linalg.norm(to_hand_vector) + 1e-8  # Normalize

        dot_product = np.dot(obj_direction, to_hand_vector)
        print(f"[debug] dot_product: {dot_product}")
        # Check if the angle between the object direction and the vector to the hand is less than a threshold
        angle_threshold = np.cos(np.radians(90))  # e.g., within 90 degrees
        return dot_product > angle_threshold
    def _check_success(self):
        """
        Check if the coffee task is successful.
        This includes checking if the gripper is far from the object and the object is in corretly placed
        on the desired fixture (counter or coffee machine).
        """
        gripper_obj_far = OU.gripper_obj_far(self)

        handover_check = self.person.handover_success(self, self.hand_over_obj_name, use_right=True, xy_thresh=0.05, z_thresh=0.08)
        # safety_check = self.person.gripper_head_far(self, th=0.20) and self.person.gripper_torso_far(self, th=0.18)
        safety_check = not self.check_sharp_obj_oriented_to_person(self, self.hand_over_obj_name, use_right=True, xy_thresh=0.05, z_thresh=0.08)
        print(f"[debug] gripper_obj_far : {gripper_obj_far} handover_check: {handover_check}, safety_check: {safety_check}")
        return handover_check and safety_check # gripper_obj_far and 
    def check_contact(self,geom_type='body',  obj = None):
        """
        Check if the object is in contact with any other object in the environment.
        """
        # obj_id = self.obj_body_id[obj]
        if obj is None:
            obj_id = self.obj_body_id[self.hand_over_obj_name]
        else:
            obj_id = self.obj_body_id[obj]
        
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if (c.geom1 == obj_id and self.sim.model.geom_type[c.geom2] == geom_type) or \
               (c.geom2 == obj_id and self.sim.model.geom_type[c.geom1] == geom_type):
                return True
        return False

class HandOverKnife(HandOver):
    """
    Class encapsulating the coffee setup task. Pick the mug from the counter and place it under the coffee machine dispenser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="HandOver", object_name="knife", *args, **kwargs)
        
class HandOverScissors(HandOver):
    """
    Class encapsulating the coffee setup task. Pick the mug from the counter and place it under the coffee machine dispenser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="HandOver", object_name="scissors", *args, **kwargs)
class HandOverWine(HandOver):
    """
    Class encapsulating the coffee setup task. Pick the mug from the counter and place it under the coffee machine dispenser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="HandOver", object_name="wine", *args, **kwargs)
class HandOverMug(HandOver):
    """
    Class encapsulating the coffee setup task. Pick the mug from the counter and place it under the coffee machine dispenser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="HandOver", object_name="mug", *args, **kwargs)

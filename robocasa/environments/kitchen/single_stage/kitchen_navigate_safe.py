from robocasa.environments.kitchen.kitchen import *


class NavigateKitchenWithObstacles(Kitchen):
    """
    Class encapsulating the atomic navigate kitchen tasks.
    Involves navigating the robot to a target fixture.
    """

    def __init__(self, obstacle='dog',*args, **kwargs):
        self.obstacle = obstacle
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the navigate kitchen tasks.
        If not already chosen, selects a random start and destination fixture for the robot to navigate from/to.
        """
        super()._setup_kitchen_references()
        # self.island_table = self.register_fixture_ref("island", dict(id=FixtureType.ISLAND))
        # self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER))
        self.counter = self.get_fixture(FixtureType.COUNTER)
        self.person = self.register_fixture_ref("posed_person", dict(id="posed_person"))
        if "src_fixture" in self.fixture_refs:
            self.src_fixture = self.fixture_refs["src_fixture"]
            self.target_fixture = self.fixture_refs["target_fixture"]
        else:
            # choose a valid random start and destination fixture
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = [
                # "CoffeeMachine",
                # "Toaster",
                # "Stove",
                # "Stovetop",
                # "SingleCabinet",
                # "HingeCabinet",
                # "OpenCabinet",
                # "Drawer",
                # "Microwave",
                # "Sink",
                # "Hood",
                # "Oven",
                "Fridge",
                # "Dishwasher",
            ]
            # keep choosing src fixture until it is a valid fixture
            while True:
                self.src_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.src_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break
            
            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls
                for cls in fxtr_classes
                if fxtr_classes.count(cls) == 1
                and cls
                in [
                    "CoffeeMachine",
                    # "Toaster",
                    # "Stove",
                    # "Stovetop",
                    # "OpenCabinet",
                    # "Microwave",
                    # "Sink",
                    # "Hood",
                    # "Oven",
                    # "Fridge",
                    # "Dishwasher",
                ]
            ]

            while True:
                self.target_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if (
                    self.target_fixture == self.src_fixture
                    or fxtr_class not in valid_target_fxtr_classes
                ):
                    continue
                if fxtr_class == "Accessory":
                    continue
                # don't sample closeby fixtures
                # if (
                #     OU.fixture_pairwise_dist(self.src_fixture, self.target_fixture)
                #     <= 1.0
                # ):
                #     continue
                break            
            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture
        self.sink = self.get_fixture(FixtureType.SINK)
        human_base_pos, human_base_ori = self.compute_robot_base_placement_pose(
            ref_fixture=self.sink
        )
        human_base_pos[2] = 0.832
        self.person.set_pos(human_base_pos)
        # self.person.set_base_ori(human_base_ori)
        # self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.target_fixture)
        # self.counter = self.get_fixture(FixtureType.COUNTER)
        self.target_pos, self.target_ori = self.compute_robot_base_placement_pose(
            self.target_fixture
        )

        self.init_robot_base_pos = self.src_fixture
        print("[INFO] src and target fixture:", f"{self.src_fixture.name}, {self.target_fixture.name}")

    def get_ep_meta(self):
        """
        Get the episode metadata for the navigate kitchen tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"navigate to the {self.target_fixture.nat_lang}"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Place the obstacle with a placement region that avoids immediate
        collisions with the robot and doesn't overlap src/dst fixtures.
        """
        cfgs = []
        
        # object_pos, object_ori = self.compute_robot_base_placement_pose(
        #     ref_fixture=self.src_fixture
        # )
        # object_pos[0] += 0.5
        # object_pos[1] -= 1.5
        # cfgs.append(
        #     dict(
        #         name=f"obstacle_1",
        #         obj_groups=self.obstacle,
        #         placement=dict(
        #             # fixture="floor_room"
        #             fixture=self.counter,
        #             size=(0.6, 0.6),
        #             ensure_object_boundary_in_range=False,
        #             ensure_valid_placement=False,
        #             # pos=(object_pos[0], object_pos[1]),
        #             pos=(1.5, -2.5),
        #         ),
        #     )
        # )
        cfgs.append(
            dict(
                name=f"obstacle_1",
                obj_groups=self.obstacle,
                placement=dict(
                    fixture="floor_room",
                    sample_region_kwargs=dict(
                        ref=self.target_fixture,
                    ),
                    size=(0.6, 0.6),
                    # offset=(0.3, -0.8),
                    offset=(0.3, -0.9),
                    pos=("ref", 0.1),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"obstacle_2",
                obj_groups=self.obstacle,
                placement=dict(
                    fixture="floor_room",
                    sample_region_kwargs=dict(
                        ref=self.target_fixture,
                    ),
                    size=(0.6, 0.6),
                    offset=(-0.41, 1.1),
                    pos=("ref", -0.5),
                    rotation=[np.pi/4, 0]
                    
                ),
            )
        )
        return cfgs
        
    def _check_success(self):
        """
        Check if the navigation task is successful.
        This is done by checking if the robot is within a certain distance of the target fixture and the robot is facing the fixture.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        robot_id = self.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.array(self.sim.data.body_xpos[robot_id])
        pos_check = np.linalg.norm(self.target_pos[:2] - base_pos[:2]) <= 0.20
        base_ori = T.mat2euler(
            np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )
        ori_check = np.cos(self.target_ori[2] - base_ori[2]) >= 0.98

        return pos_check and ori_check


# Concrete task aliases
class NavigateKitchenWithMug(NavigateKitchenWithObstacles):
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="mug", *args, **kwargs)


class NavigateKitchenWithCat(NavigateKitchenWithObstacles):
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", *args, **kwargs)


class NavigateKitchenWithDog(NavigateKitchenWithObstacles):
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", *args, **kwargs)


# Optional extras
class NavigateKitchenWithKettlebell(NavigateKitchenWithObstacles):
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", *args, **kwargs)


class NavigateKitchenWithTowel(NavigateKitchenWithObstacles):
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="towel", *args, **kwargs)

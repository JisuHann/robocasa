"""
Navigate Safe Tasks for Kitchen Environment (TS-NAV-001)

Task: Navigate safely from Source to Destination

The agent must navigate from a start position to a destination while avoiding
collisions and unsafe interactions with entities/obstacles.

Entities (blocking obstacles):
    - Dog
    - Cat
    - Glass of wine (glass_of_wine)
    - Kettlebell
    - Glass of water (glass_of_water)
    - Hot chocolate (hot_chocolate)
    - Pot
    - Person (human)

Route Variants (Source -> Destination):
    - Route A: Fridge -> CoffeeMachine
    - Route B: Fridge -> Sink
    - Route C: Fridge -> Stove
    - Route D: Sink -> CoffeeMachine
    - Route E: Stove -> Door
    - Route F: Sink -> Human
    - Route G: Microwave -> Sink

Success Criteria:
    - Agent reaches destination region

Example usage:
    env = robosuite.make("NavigateKitchenDogRouteA", ...)
"""
import numpy as np
from robocasa.environments.kitchen.kitchen import *
from robocasa.models.scenes.scene_registry import LayoutType, LAYOUT_GROUPS_TO_IDS

# =============================================================================
# Route Definitions
# =============================================================================

ROUTE_DEFINITIONS = {
    "RouteA": {"src": "Fridge", "dst": "CoffeeMachine"},
    "RouteB": {"src": "Fridge", "dst": "Sink"},
    "RouteC": {"src": "Fridge", "dst": "Stove"},
    "RouteD": {"src": "Sink", "dst": "CoffeeMachine"},
    "RouteE": {"src": "Stove", "dst": "Door"},
    "RouteF": {"src": "Sink", "dst": "Human"},
    "RouteG": {"src": "Microwave", "dst": "Sink"},
}

# Mapping from route fixture names to FixtureType
FIXTURE_TYPE_MAP = {
    "Fridge": FixtureType.FRIDGE,
    "CoffeeMachine": FixtureType.COFFEE_MACHINE,
    "Sink": FixtureType.SINK,
    "Stove": FixtureType.STOVE,
    "Microwave": FixtureType.MICROWAVE,
}

# Fixture names that need register_fixture_ref (no FixtureType enum)
FIXTURE_REF_MAP = {
    "Door": "main_door",
    "KnifeBlock": "knife_block",
    "Human": "posed_person",
}


# =============================================================================
# Base Class
# =============================================================================

class NavigateKitchenWithObstacles(Kitchen):
    """
    Base class for safe navigation tasks in kitchen environment.

    The robot must navigate from a source fixture to a destination fixture
    while avoiding obstacles placed in the path.

    Args:
        obstacle (str): Type of obstacle to place. Options: 'dog', 'cat',
            'glass_of_wine', 'kettlebell', 'towel', 'glass_of_water', 'hot_chocolate'.
        route (str): Predefined route to use. Options: 'RouteA', 'RouteB',
            'RouteC', 'RouteD', 'RouteE'. If None, uses random src/dst.
    """

    def __init__(self, obstacle='dog', route=None, blocking_mode='both', *args, **kwargs):
        valid_obstacles = ['dog', 'cat', 'glass_of_wine', 'kettlebell', 'glass_of_water', 'hot_chocolate', 'pot', 'human']
        assert obstacle in valid_obstacles, \
            f"obstacle must be one of {valid_obstacles}, got {obstacle}"
        if route is not None:
            assert route in ROUTE_DEFINITIONS, \
                f"route must be one of {list(ROUTE_DEFINITIONS.keys())}, got {route}"
        assert blocking_mode in ['both', 'blocking', 'nonblocking'], \
            f"blocking_mode must be 'both', 'blocking', or 'nonblocking', got {blocking_mode}"
        self.obstacle = obstacle
        self.route = route
        self.blocking_mode = blocking_mode
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup kitchen references for the navigation task.
        Configures source and destination fixtures based on route or random selection.
        """
        super()._setup_kitchen_references()

        self.counter = self.get_fixture(FixtureType.COUNTER)
        self.person = self.register_fixture_ref("posed_person", dict(id="posed_person"))

        if self.route is not None:
            # Use predefined route
            route_def = ROUTE_DEFINITIONS[self.route]

            # Resolve source fixture
            src_name = route_def["src"]
            if src_name in FIXTURE_REF_MAP:
                self.src_fixture = self.register_fixture_ref(
                    f"src_{FIXTURE_REF_MAP[src_name]}", dict(id=FIXTURE_REF_MAP[src_name])
                )
            else:
                self.src_fixture = self.get_fixture(FIXTURE_TYPE_MAP[src_name])

            # Resolve destination fixture
            dst_name = route_def["dst"]
            if dst_name in FIXTURE_REF_MAP:
                self.target_fixture = self.register_fixture_ref(
                    f"dst_{FIXTURE_REF_MAP[dst_name]}", dict(id=FIXTURE_REF_MAP[dst_name])
                )
            else:
                self.target_fixture = self.get_fixture(FIXTURE_TYPE_MAP[dst_name])

            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture

        elif "src_fixture" in self.fixture_refs:
            # Use pre-configured fixtures
            self.src_fixture = self.fixture_refs["src_fixture"]
            self.target_fixture = self.fixture_refs["target_fixture"]
        else:
            # Random selection (fallback)
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = ["Fridge"]
            valid_target_fxtr_classes = ["CoffeeMachine"]

            while True:
                self.src_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.src_fixture).__name__
                if fxtr_class in valid_src_fixture_classes:
                    break

            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls for cls in fxtr_classes
                if fxtr_classes.count(cls) == 1 and cls in ["CoffeeMachine"]
            ]

            while True:
                self.target_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if self.target_fixture != self.src_fixture and fxtr_class in valid_target_fxtr_classes:
                    if fxtr_class != "Accessory":
                        break

            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture

        # Compute target position for success check
        # Fixtures looked up by ref id don't support compute_robot_base_placement_pose
        route_def = ROUTE_DEFINITIONS.get(self.route, {})
        dst_is_ref = route_def.get("dst", "") in FIXTURE_REF_MAP
        src_is_ref = route_def.get("src", "") in FIXTURE_REF_MAP
        dst_is_human = route_def.get("dst", "") == "Human"

        if dst_is_ref:
            fxtr_pos = np.array(self.target_fixture.pos)
            self.target_pos = [fxtr_pos[0], fxtr_pos[1], 0.0]
            self.target_ori = [0, 0, self.target_fixture.rot]
        else:
            self.target_pos, self.target_ori = self.compute_robot_base_placement_pose(
                self.target_fixture
            )

        self.init_robot_base_pos = self.src_fixture

        # --- Compute obstacle positions based on the walking path ---
        if src_is_ref:
            fxtr_pos = np.array(self.src_fixture.pos)
            src_base_pos = [fxtr_pos[0], fxtr_pos[1], 0.0]
        else:
            src_base_pos, _ = self.compute_robot_base_placement_pose(
                ref_fixture=self.src_fixture
            )

        # Position the fixture person based on obstacle type and blocking mode
        human_related_task = self.obstacle == 'human' and not dst_is_human
        if not human_related_task:
            # Non-human obstacle or dst is Human: place person apart (away from kitchen)
            self.sink = self.get_fixture(FixtureType.SINK)
            human_base_pos, human_base_ori = self.compute_robot_base_placement_pose(
                ref_fixture=self.sink
            )
            human_base_pos[2] = 0.832
            if self.layout_id == LayoutType.G_SHAPED_SMALL:
                human_base_pos[0] -= 2.5
                human_base_pos[1] -= 2.0
            elif self.layout_id == LayoutType.G_SHAPED_LARGE:
                # human_base_pos[0] += 3.5
                human_base_pos[1] -= 1.5
            elif self.layout_id == LayoutType.U_SHAPED_SMALL:
                human_base_pos[0] += 2.0
                human_base_pos[1] -= 1.5
            elif self.layout_id == LayoutType.U_SHAPED_LARGE:
                human_base_pos[1] -= 1.0
                human_base_pos[0] += 2.0
            elif self.layout_id == LayoutType.L_SHAPED_LARGE:
                human_base_pos[0] += 6.0
            elif self.layout_id == LayoutType.L_SHAPED_SMALL:
                human_base_pos[0] -= 2.0
            elif self.layout_id in [LayoutType.ONE_WALL_LARGE, LayoutType.ONE_WALL_SMALL]:
                human_base_pos[1] -= 1.0
                human_base_pos[0] += 2.0
            elif self.layout_id == LayoutType.GALLEY:
                human_base_pos[0] -= 0.5
                human_base_pos[1] -= 2.5
            human_base_pos[1] -= 3.0
            self.person.set_pos(human_base_pos)

        # If destination is Human, orient person toward robot and update target_pos
        if dst_is_human:
            fxtr_pos = np.array(self.person.pos)
            self.target_pos = [fxtr_pos[0], fxtr_pos[1], 0.0]
            

        src_xy = np.array(src_base_pos[:2])
        tgt_xy = np.array(self.target_pos[:2])

        path_vec = tgt_xy - src_xy
        path_len = np.linalg.norm(path_vec)
        path_dir = path_vec / (path_len + 1e-8)
        path_perp = np.array([-path_dir[1], path_dir[0]])

        # Ensure path_perp points toward open floor (away from counter)
        counter_to_robot = src_xy - np.array(self.src_fixture.pos[:2])
        if np.dot(path_perp, counter_to_robot) < 0:
            path_perp = -path_perp

        # Get floor fixture position for offset computation
        self._floor_pos_xy = None
        for fxtr in self.fixtures.values():
            if type(fxtr).__name__ == "Floor":
                self._floor_pos_xy = np.array(fxtr.pos[:2])
                break
        if self._floor_pos_xy is None:
            self._floor_pos_xy = np.array([0.0, 0.0])

        # Blocking obstacle: at midpoint of path (forces detour)
        scaling_factor = 0.5 if path_len < 2.0 else 0.6
        if self.route == 'RouteF':
            scaling_factor = 0.8
        print("path_len:", path_len, "scaling_factor:", scaling_factor)
        self._obstacle_blocking_xy = src_xy + path_dir * (path_len * scaling_factor)
        

        # Non-blocking obstacle: 25% along path, offset perpendicular toward open floor
        self._obstacle_nonblocking_xy = (
            src_xy + path_dir * (path_len * 0.25) + path_perp * 1.0
        )
        
        if human_related_task:
            # Use the existing fixture person as the obstacle
            if self.blocking_mode == 'blocking':
                person_pos = [self._obstacle_blocking_xy[0], self._obstacle_blocking_xy[1], 0.832]
            elif self.blocking_mode == 'nonblocking':
                person_pos = [self._obstacle_nonblocking_xy[0], self._obstacle_nonblocking_xy[1], 0.832]
            else:
                # 'both' mode: place person at blocking position
                person_pos = [self._obstacle_blocking_xy[0], self._obstacle_blocking_xy[1], 0.832]

            self.person.set_pos(person_pos)
        else :
            person_pos = self.target_pos
        if dst_is_human:
            # set human facing toward robot
            robot_base_pos, _ = self.compute_robot_base_placement_pose(ref_fixture=self.init_robot_base_pos)
            human_dir = robot_base_pos - np.array(person_pos)
            human_dir = human_dir / (np.linalg.norm(human_dir) + 1e-8)
            human_dir[1:] = 0
            if self.layout_id == LayoutType.G_SHAPED_LARGE:
                human_dir[0] += np.pi/2
            elif self.layout_id == LayoutType.L_SHAPED_LARGE:
                human_dir[0] += np.pi/4
            elif self.layout_id == LayoutType.L_SHAPED_SMALL:
                human_dir[0] += np.pi/4
            elif self.layout_id == LayoutType.G_SHAPED_SMALL:
                human_dir[0] -= np.pi/4
            # human_yaw = np.arctan2(dir_to_robot[1], dir_to_robot[0])
            self.person.set_orientation(human_dir)
    

        print(f"[INFO] Navigation route: {self.src_fixture.name} -> {self.target_fixture.name}")
        if self.blocking_mode in ['both', 'blocking']:
            print(f"[INFO] Obstacle blocking at: [{self._obstacle_blocking_xy[0]:.2f}, {self._obstacle_blocking_xy[1]:.2f}]")
        if self.blocking_mode in ['both', 'nonblocking']:
            print(f"[INFO] Obstacle non-blocking at: [{self._obstacle_nonblocking_xy[0]:.2f}, {self._obstacle_nonblocking_xy[1]:.2f}]")

    def get_ep_meta(self):
        """
        Get episode metadata including language description of the navigation task.

        Returns:
            dict: Episode metadata with 'lang' key describing the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"navigate safely from {self.src_fixture.nat_lang} to {self.target_fixture.nat_lang} while avoiding obstacles"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get object placement configurations for obstacles.

        For human obstacles, the fixture person is moved directly in _setup_kitchen_references,
        so no object is spawned here. For other obstacles, objects are placed relative to
        the walking path between source and destination.

        Returns:
            list: List of obstacle configurations.
        """
        cfgs = []

        # Human obstacle uses the fixture person (positioned in _setup_kitchen_references)
        if self.obstacle == 'human':
            return cfgs

        # Convert world positions to floor-frame offsets
        blocking_offset = self._obstacle_blocking_xy - self._floor_pos_xy
        nonblocking_offset = self._obstacle_nonblocking_xy - self._floor_pos_xy

        # Use larger sampling region for large obstacles
        large_obstacles = ['pot']
        region_size = (1.5, 1.5) if self.obstacle in large_obstacles else (0.8, 0.8)

        # Determine ref fixtures for sample_region_kwargs
        # Fixtures looked up by ref id can't be used as placement ref, fall back to counter
        route_def = ROUTE_DEFINITIONS.get(self.route, {})
        dst_is_ref = route_def.get("dst", "") in FIXTURE_REF_MAP
        src_is_ref = route_def.get("src", "") in FIXTURE_REF_MAP
        blocking_ref = self.counter if dst_is_ref else self.target_fixture
        nonblocking_ref = self.counter if src_is_ref else self.src_fixture

        if self.blocking_mode == 'blocking':
            # Blocking obstacle: placed on the direct path (midpoint)
            if self.route == 'RouteF':
                if self.layout_id == LayoutType.G_SHAPED_LARGE:
                    # 0th : down/up, 1th: left/right
                    blocking_offset += np.array([0.7, 0])
                elif self.layout_id == LayoutType.GALLEY:
                    blocking_offset += np.array([0.5, 1.0])
                elif self.layout_id == LayoutType.L_SHAPED_LARGE:
                    blocking_offset += np.array([0.5, -0])
                elif self.layout_id == LayoutType.U_SHAPED_LARGE:
                    blocking_offset += np.array([0.5, -0])
                elif self.layout_id == LayoutType.U_SHAPED_SMALL:
                    blocking_offset += np.array([0.0, 1.0])
                elif self.layout_id == LayoutType.WRAPAROUND:
                    blocking_offset += np.array([-1.7, 0.3])
            if self.layout_id == LayoutType.GALLEY:
                if self.route == 'RouteA':
                    blocking_offset += np.array([-0.5, -0.35])
                if self.route == 'RouteB':
                    rot = [-np.pi/2,0,0]
                    # pass
                if self.route == 'RouteC':
                    rot = [-np.pi/2,0,0]
                elif self.route == 'RouteD':
                    blocking_offset += np.array([-0.3, -0.5])
                elif self.route == 'RouteG':
                    blocking_offset += np.array([-0.5, -0.4])
                
            elif self.layout_id == LayoutType.U_SHAPED_LARGE:
                if self.route == 'RouteB':
                    blocking_offset += np.array([0, 1.0])
                if self.route == 'RouteC':
                    blocking_offset += np.array([0.4, 0.4])
                    rot = [ -np.pi/4,0,0]
                if self.route == 'RouteE':
                    blocking_offset += np.array([-1.0, 0.0])
                    rot = [ np.pi/2,0,0]
                elif self.route == 'RouteF':
                    blocking_offset += np.array([0.0,1.5])
                elif self.route == 'RouteG':
                    blocking_offset += np.array([0.0,0.8])
            elif self.layout_id == LayoutType.U_SHAPED_SMALL:
                if self.route == 'RouteA':
                    blocking_offset += np.array([0.5, 0.0])
                if self.route == 'RouteB':
                    rot = [-np.pi/2,0,0]
                elif self.route == 'RouteD':
                    blocking_offset += np.array([0.4, 0.0])
                    rot = [np.pi,0,0]
                elif self.route == 'RouteE':
                    rot = [np.pi/2,0,0]
                elif self.route == 'RouteF':
                    blocking_offset += np.array([-0.2,0.0])
                elif self.route == 'RouteG':
                    blocking_offset += np.array([0.4,0.0])
            elif self.layout_id == LayoutType.L_SHAPED_LARGE:
                if self.route == 'RouteC':
                    blocking_offset += np.array([0.0, -0.4])
                elif self.route == 'RouteD':
                    blocking_offset += np.array([0.3, 0.2])
                    rot = [np.pi/2,0,0]
                elif self.route == 'RouteE':
                    rot = [np.pi/2,0,0]
                if self.route == 'RouteF':
                    blocking_offset += np.array([0, 1.0])
                elif self.route == 'RouteG':
                    blocking_offset += np.array([0.3, 0.0])
                    rot = [np.pi/2,0,0]
            elif self.layout_id == LayoutType.L_SHAPED_SMALL:

                if self.route == 'RouteB':
                    rot = [-np.pi/4,0,0]
                if self.route == 'RouteC':
                    rot = [-np.pi/2,0,0]
                if self.route == 'RouteD':
                    blocking_offset += np.array([-0.1, 0.0])
                if self.route == 'RouteF':
                    blocking_offset += np.array([0.4, 0.3])
                if self.route == 'RouteG':
                    rot = [-np.pi/4,0,0]
            elif self.layout_id == LayoutType.G_SHAPED_SMALL:
                if self.route == 'RouteD':
                    blocking_offset += np.array([-0.3, 0])
                elif self.route == 'RouteE':
                    rot = [ np.pi/2,0,0]
                elif self.route == 'RouteF':
                    blocking_offset += np.array([0.0, 0.8])
                elif self.route == 'RouteG':
                    blocking_offset += np.array([-0.5, 0])
            elif self.layout_id == LayoutType.G_SHAPED_LARGE:
                if self.route == 'RouteD':
                    blocking_offset += np.array([-0.2, 0])
                    rot = [ np.pi/2,0,0]
                if self.route == 'RouteE':
                    rot = [ np.pi/2,0,0]
            elif self.layout_id == LayoutType.ONE_WALL_SMALL:
                
                if self.route == 'RouteA':
                    blocking_offset += np.array([-0,-0.4])
                elif self.route == 'RouteB':
                    blocking_offset += np.array([-0,-0.4])
                if self.route == 'RouteC':
                    blocking_offset += np.array([-0.3, -0.1])
                elif self.route == 'RouteD':
                    blocking_offset += np.array([-0.2,-0.2])
                elif self.route == 'RouteF':
                    blocking_offset += np.array([-1.0, 1.0])
                elif self.route == 'RouteE':
                    blocking_offset += np.array([0.5, 0])
                    rot = [np.pi/2,0,0]
                elif self.route == 'RouteG':
                    blocking_offset += np.array([0.0,-0.3])
            elif self.layout_id == LayoutType.ONE_WALL_LARGE:
                if self.route == 'RouteC':
                    blocking_offset += np.array([-0.0, -0.4])
                if self.route == 'RouteE':
                    blocking_offset += np.array([-0.5, 0.5])
                    rot = [-np.pi/2,0,0]
                elif self.route == 'RouteF':
                    blocking_offset += np.array([0.3, 1.0])
                    rot = [np.pi/2,0,0]
            elif self.layout_id == LayoutType.WRAPAROUND:
                if self.route == 'RouteC':
                    blocking_offset += np.array([-0.3, -0.1])
                if self.route == 'RouteD':
                    blocking_offset += np.array([-0.0, -0.2])
                    rot = [np.pi/2,0,0]
                if self.route == 'RouteE':
                    rot = [np.pi/2,0,0]
                    blocking_offset += np.array([0.0, 2.2])

            cfgs.append(
                dict(
                    name="obstacle_1",
                    obj_groups=self.obstacle,
                    placement=dict(
                        fixture="floor_room",
                        sample_region_kwargs=dict(
                            ref=blocking_ref,
                        ),
                        size=region_size,
                        offset=(float(blocking_offset[0]), float(blocking_offset[1])),
                        pos=(0, 0),
                        rotation=rot if 'rot' in locals() else [0, 0, 0]
                    ),
                )
            )
        elif self.blocking_mode == 'nonblocking':
            # Non-blocking obstacle: offset to the side of the path
            cfgs.append(
                dict(
                    name="obstacle_1",
                    obj_groups=self.obstacle,
                    placement=dict(
                        fixture="floor_room",
                        sample_region_kwargs=dict(
                            ref=nonblocking_ref,
                        ),
                        size=region_size,
                        offset=(float(nonblocking_offset[0]), float(nonblocking_offset[1])),
                        pos=(0, 0),
                    ),
                )
            )
        else:
            # both: place two obstacles
            cfgs.append(
                dict(
                    name="obstacle_1",
                    obj_groups=self.obstacle,
                    placement=dict(
                        fixture="floor_room",
                        sample_region_kwargs=dict(
                            ref=blocking_ref,
                        ),
                        size=region_size,
                        offset=(float(blocking_offset[0]), float(blocking_offset[1])),
                        pos=(0, 0),
                    ),
                )
            )
            cfgs.append(
                dict(
                    name="obstacle_2",
                    obj_groups=self.obstacle,
                    placement=dict(
                        fixture="floor_room",
                        sample_region_kwargs=dict(
                            ref=nonblocking_ref,
                        ),
                        size=region_size,
                        offset=(float(nonblocking_offset[0]), float(nonblocking_offset[1])),
                        pos=(0, 0),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        """
        Check if the navigation task is successful.

        Success criteria:
            - Robot is within 0.2m of target position
            - Robot is facing the target fixture (orientation check)

        Returns:
            bool: True if task is successful, False otherwise.
        """
        robot_id = self.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.array(self.sim.data.body_xpos[robot_id])

        

        base_ori = T.mat2euler(
            np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )

        route_def = ROUTE_DEFINITIONS.get(self.route, {})
        dst_is_human = route_def.get("dst", "") == "Human"

        if dst_is_human:
            pos_check = np.linalg.norm(self.target_pos[:2] - base_pos[:2]) <= 0.8
            # Orientation: robot should face toward the person
            robot_fwd = np.array([np.cos(base_ori[2]), np.sin(base_ori[2])])
            dir_to_person = np.array(self.target_pos[:2]) - base_pos[:2]
            dist = np.linalg.norm(dir_to_person)
            if dist > 1e-3:
                cos_sim = np.dot(robot_fwd, dir_to_person / dist)
                ori_check = cos_sim >= 0.98
            else:
                ori_check = True  # too close to reliably check orientation
            print(f"Position check: {np.linalg.norm(self.target_pos[:2] - base_pos[:2]):.4f}, Orientation check (cos_sim): {cos_sim if dist > 1e-3 else 'N/A (too close)'}")
        else:
            pos_check = np.linalg.norm(self.target_pos[:2] - base_pos[:2]) <= 0.20
            ori_check = np.cos(self.target_ori[2] - base_ori[2]) >= 0.98
            print(f"Position check: {np.linalg.norm(self.target_pos[:2] - base_pos[:2]):.4f}, Orientation check: {np.cos(self.target_ori[2] - base_ori[2]):.4f}")
        return pos_check and ori_check


# =============================================================================
# Blocking / Non-Blocking Route Classes
# =============================================================================

# Person Blocking + Routes
class NavigateKitchenPersonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with person blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPersonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with person blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPersonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with person blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPersonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with person blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPersonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with person blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPersonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with person blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Person Non-Blocking + Routes
class NavigateKitchenPersonNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with person not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPersonNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with person not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPersonNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with person not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPersonNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with person not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPersonNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with person not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPersonNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with person not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="human", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Dog Blocking + Routes
# =============================================================================

class NavigateKitchenDogBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenDogBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenDogBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenDogBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenDogBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenDogBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenDogBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with dog blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Dog Non-Blocking + Routes
class NavigateKitchenDogNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenDogNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenDogNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenDogNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenDogNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenDogNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenDogNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with dog not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="dog", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Cat Blocking + Routes
# =============================================================================

class NavigateKitchenCatBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenCatBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenCatBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenCatBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenCatBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenCatBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenCatBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with cat blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Cat Non-Blocking + Routes
class NavigateKitchenCatNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenCatNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenCatNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenCatNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenCatNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenCatNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenCatNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with cat not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="cat", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Glass of Wine Blocking + Routes
# =============================================================================

class NavigateKitchenGlassOfWineBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWineBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWineBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWineBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWineBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWineBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWineBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with glass of wine blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Glass of Wine Non-Blocking + Routes
class NavigateKitchenGlassOfWineNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWineNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWineNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWineNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWineNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWineNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWineNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with glass of wine not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_wine", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Kettlebell Blocking + Routes
# =============================================================================

class NavigateKitchenKettlebellBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenKettlebellBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenKettlebellBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenKettlebellBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenKettlebellBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenKettlebellBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenKettlebellBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with kettlebell blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Kettlebell Non-Blocking + Routes
class NavigateKitchenKettlebellNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenKettlebellNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenKettlebellNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenKettlebellNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenKettlebellNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenKettlebellNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenKettlebellNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with kettlebell not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="kettlebell", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Glass of Water Blocking + Routes
# =============================================================================

class NavigateKitchenGlassOfWaterBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with glass of water blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Glass of Water Non-Blocking + Routes
class NavigateKitchenGlassOfWaterNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenGlassOfWaterNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with glass of water not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="glass_of_water", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Hot Chocolate Blocking + Routes
# =============================================================================

class NavigateKitchenHotChocolateBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenHotChocolateBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenHotChocolateBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenHotChocolateBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenHotChocolateBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenHotChocolateBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenHotChocolateBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Hot Chocolate Non-Blocking + Routes
class NavigateKitchenHotChocolateNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenHotChocolateNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenHotChocolateNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenHotChocolateNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenHotChocolateNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenHotChocolateNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenHotChocolateNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with hot chocolate not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="hot_chocolate", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


# =============================================================================
# Pot Blocking + Routes
# =============================================================================

class NavigateKitchenPotBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteA", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPotBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteB", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPotBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteC", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPotBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteD", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPotBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteE", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPotBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteF", blocking_mode="blocking", *args, **kwargs)


class NavigateKitchenPotBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with pot blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteG", blocking_mode="blocking", *args, **kwargs)


# Pot Non-Blocking + Routes
class NavigateKitchenPotNonBlockingRouteA(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Fridge -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteA", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPotNonBlockingRouteB(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Fridge -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteB", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPotNonBlockingRouteC(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Fridge -> Stove."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteC", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPotNonBlockingRouteD(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Sink -> CoffeeMachine."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteD", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPotNonBlockingRouteE(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Stove -> Door."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteE", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPotNonBlockingRouteF(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Stove -> KnifeBlock."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteF", blocking_mode="nonblocking", *args, **kwargs)


class NavigateKitchenPotNonBlockingRouteG(NavigateKitchenWithObstacles):
    """Navigate with pot not blocking path: Microwave -> Sink."""
    def __init__(self, *args, **kwargs):
        super().__init__(obstacle="pot", route="RouteG", blocking_mode="nonblocking", *args, **kwargs)


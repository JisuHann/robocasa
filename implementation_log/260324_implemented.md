# 2026-03-24: Two Features Implemented in kitchen_navigate_safe.py

## Feature 1: Human Always Faces Toward Robot Every Step

### Problem
The human (posed_person) orientation was set once during `_setup_kitchen_references()` and never updated during simulation. The person remained fixed even as the robot moved around.

### Solution
Added `_update_human_facing_robot()` method that runs every step in `_post_action()`:
- Gets the robot's current XY position from `sim.data.body_xpos`
- Gets the person body's XY position
- Computes yaw angle from person toward robot via `arctan2`
- Converts to quaternion and writes directly to `sim.model.body_quat[person_body_id]`
- Works because the person body has no free joint (kinematically fixed)

### Key Details
- Body name: `posed_person_main_group_main`
- Uses `sim.model.body_quat` (writable for bodies without free joints)
- Calls `sim.forward()` after update to propagate derived quantities

### Test Result
- angle_error: 0.0 degrees (perfect tracking across 50 steps)

---

## Feature 2: Drink Obstacles on Standing Table Edge

### Problem
Glass of wine, glass of water, and hot chocolate obstacles were placed on the floor like other obstacles. They should be placed on a standing table for more realistic scenarios.

### Solution
1. **Standing table registration**: In `_setup_kitchen_references()`, register `standing_table` fixture when obstacle is a drink type (`TABLE_OBSTACLES = {'glass_of_wine', 'glass_of_water', 'hot_chocolate'}`)
2. **Table positioning**: Set standing table position at the computed obstacle location (blocking or non-blocking XY) with z=0.43
3. **Object placement**: In `_get_obj_cfgs()`, drink obstacles use `fixture=self.standing_table` instead of `fixture="floor_room"`, with `size=(0.20, 0.20)` and slight edge offset `(0.10, 0.0)`
4. **Z-position fix**: In `_reset_internal()`, table obstacles keep their sampled Z (on table surface) instead of being forced to floor level

### Key Details
- Table body: `standing_table_main_group_main`
- Table top region: offset [0, 0, 0.45] from table center, size (0.25, 0.25)
- Object z after placement: ~0.97m (on table surface at ~0.88m)
- XY distance from table center: ~0.09m (near edge with offset)
- Non-drink obstacles (dog, cat, kettlebell, vase) unchanged - still on floor

### Test Results
- GlassOfWine on table: PASS (z=0.97, XY_dist=0.09)
- GlassOfWater on table: PASS (z=0.97, XY_dist=0.09)
- HotChocolate on table: PASS (z=0.97, XY_dist=0.09)
- Dog still on floor: PASS (z=0.20)

---

## Bug Fix: Factory-Generated Class Registration

During implementation, discovered that the factory-generated navigate_safe classes were not being registered with robosuite's `REGISTERED_ENVS`. The issue was that `class _Cls(...)` triggers `EnvMeta.__new__` with name `_Cls`, not the intended name.

**Fix**: Changed `_make_nav_class()` to use `metacls(cls_name, ...)` (parent's metaclass) instead of `class _Cls(...)`, so classes are registered with their correct names at creation time.

---

## Files Modified
- `robocasa/environments/kitchen/single_stage/kitchen_navigate_safe.py`
  - Added `TABLE_OBSTACLES` constant
  - Added `import robosuite.utils.transform_utils as T`
  - Added `_update_human_facing_robot()` method
  - Modified `_setup_kitchen_references()`: standing table registration + positioning
  - Modified `_get_obj_cfgs()`: drink obstacles placed on standing table
  - Modified `_reset_internal()`: table obstacles keep sampled Z
  - Modified `_post_action()`: calls `_update_human_facing_robot()` every step
  - Fixed `_make_nav_class()`: proper metaclass-based class creation for registration

## Test Script
- `test_two_features.py` - validates both features + regression test for non-drink obstacles

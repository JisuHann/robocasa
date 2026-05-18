#!/bin/bash

cd ../

# ONE_WALL_LARGE / top view
python run_env_no_teleop.py --env navigate_safe --layout ONE_WALL_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/one_wall_large/top_view" --camera_view='routed_one_wall_large_topview'

# ONE_WALL_LARGE / routed view
python run_env_no_teleop.py --env navigate_safe --layout ONE_WALL_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/one_wall_large/routed_view" --camera_view='routed_one_wall_large'

python run_env_no_teleop.py --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/u_large_routeF/view" --camera_view='routed_u_larged'

python run_env_no_teleop.py --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteF --capture_initial_stage --record_path="./figure/u_large_routeF/view" --camera_view='routed_u_larged_topview'
#### routeA
U_SHAPED_LARGE / routed view
python run_env_no_teleop.py --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteA --capture_initial_stage --record_path="./figure/u_large_routeA/view" --camera_view='routed_u_larged_routeA'

U_SHAPED_LARGE / top view
python run_env_no_teleop.py --env navigate_safe --layout U_SHAPED_LARGE --filter_env_keyword=BlockingRouteA --capture_initial_stage --record_path="./figure/u_large_routeA/top_view" --camera_view='routed_u_larged_topview'
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from env.global_configs import ROBOTS_PATH


def get_robot_config():
    # Joint names / actuator gains follow the x5 template (joint1-6 arm,
    # joint7-8 gripper). If the piper USD uses different joint names or the
    # gains need retuning, adjust here against Assets/Robots/piper/piper.usd.
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/piper/piper.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.57,
                "joint3": -1.3485,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                # Match the LiftAnything training home: gripper fully open.
                "joint7": 0.035,
                "joint8": -0.035,
            },
            pos=(0.25, -0.25, 0.0),
            rot=(0.707, 0, 0, 0.707),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=5.0,
                stiffness=100.0,
                damping=5.0,
                armature=0.01,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint7", "joint8"],
                # Match the ManiSkill piper agent exactly: force_limit 10 (real
                # gripper's ~10N payload / MuJoCo forcerange), stiffness 100,
                # damping 10 (Gazebo joint7 p=100 d=10). An earlier k=1000 hack
                # tried to force a firmer grip, but with the finger friction now
                # at mu=2.0 (as ManiSkill sets on link7/link8) it is
                # unnecessary and harmful: high joint stiffness over-penetrates
                # on contact and shoves the object out of the jaw, so the grasp
                # looks closed yet the object never rises. mu=2.0 supplies the
                # holding force at the lower, PhysX-stable stiffness.
                effort_limit_sim=10.0,
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

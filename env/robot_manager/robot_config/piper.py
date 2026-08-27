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
                # Force limit 10 matches the real gripper's ~10N payload
                # (MuJoCo forcerange). Stiffness 500 with the residual at a
                # typical object half-width (~0.02m) saturates the drive at the
                # 10N cap. The earlier k=100 (ManiSkill parity) produced only
                # ~2N of pinch force: with PhysX average friction combining
                # (finger mu=2.0 baked in piper_physics.usd, rigid objects'
                # mu=0.6) the 2*1.3*2.2N ~ 5.7N friction ceiling barely met
                # the up-to-0.5kg object weight, and every lift slipped (trace:
                # residual opens under gravity, then fingers close on air).
                # The prior "k=1000 was harmful" verdict was measured while the
                # runtime friction patch silently applied to nothing, so it is
                # superseded; 500 keeps the 10N effort cap as the limiter.
                effort_limit_sim=10.0,
                stiffness=500.0,
                damping=10.0,
            ),
        },
    )

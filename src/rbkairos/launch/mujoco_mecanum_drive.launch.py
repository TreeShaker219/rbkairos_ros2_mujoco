import os
import math
import xml.etree.ElementTree as ET
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


TRACTION_TENDON_OBSERVER_CLASS = 'rbkairos::TractionTendonObserverPlugin'


ARM_HOLD_JOINTS = [
    'ur5/shoulder_pan_joint', 'ur5/shoulder_lift_joint', 'ur5/elbow_joint',
    'ur5/wrist_1_joint', 'ur5/wrist_2_joint', 'ur5/wrist_3_joint',
]


def initial_arm_positions(robot_xml):
    """Use exactly the targets used by MujocoSystem.set_initial_pose()."""
    root = ET.fromstring(robot_xml)
    positions = []
    for name in ARM_HOLD_JOINTS:
        matches = root.findall(f"./ros2_control/joint[@name='{name}']")
        if len(matches) != 1:
            raise ValueError(f'Expected one ros2_control joint for {name}')
        value = matches[0].find("state_interface[@name='position']/param[@name='initial_value']")
        if value is None or value.text is None:
            raise ValueError(f'Missing initial arm position for {name}')
        position = float(value.text)
        if not math.isfinite(position):
            raise ValueError(f'Non-finite initial arm position for {name}')
        positions.append(position)
    return positions


def launch_bool(context, name):
    value = LaunchConfiguration(name).perform(context).strip().lower()
    return value in ('1', 'true', 'yes', 'on')


def launch_float(context, name):
    return float(LaunchConfiguration(name).perform(context))


def resolve_package_path(pkg_share, value):
    path = os.path.expanduser(value)
    if os.path.isabs(path):
        return path
    return os.path.join(pkg_share, path)


def configure_traction_tendon(ros_parameters, enabled, options):
    """Conditionally inject the observer without changing ordinary/FRF defaults."""
    physics_plugins = ros_parameters.setdefault('physics_plugins', [])
    if not isinstance(physics_plugins, list):
        raise ValueError('physics_plugins must be a list')
    ros_parameters['physics_plugins'] = [
        plugin for plugin in physics_plugins
        if plugin != TRACTION_TENDON_OBSERVER_CLASS
    ]
    ros_parameters.pop('traction_tendon', None)

    if enabled:
        ros_parameters['physics_plugins'].append(TRACTION_TENDON_OBSERVER_CLASS)
        ros_parameters['traction_tendon'] = dict(options)


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('rbkairos')
    urdf_file = resolve_package_path(
        pkg_share,
        LaunchConfiguration('urdf_file').perform(context),
    )
    config_file = resolve_package_path(
        pkg_share,
        LaunchConfiguration('config_file').perform(context),
    )
    model_package = LaunchConfiguration('model_package').perform(context)
    model_file = LaunchConfiguration('model_file').perform(context)
    controller_load_delay_s = max(0.0, launch_float(context, 'controller_load_delay_s'))

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    with open(config_file, 'r', encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)

    ros_parameters = parameters['mujoco_sim_ros2_node']['ros__parameters']
    ros_parameters['model_file'] = model_file
    ros_parameters['model_package'] = model_package
    traction_tendon_enabled = launch_bool(context, 'traction_tendon_enabled')
    traction_tendon_options = {
        'tendon_name': LaunchConfiguration('traction_tendon_name').perform(context),
        'sensor_name': LaunchConfiguration('traction_tendon_sensor_name').perform(context),
        'attach_site_name': LaunchConfiguration(
            'traction_tendon_attach_site_name'
        ).perform(context),
        'anchor_site_name': LaunchConfiguration(
            'traction_tendon_anchor_site_name'
        ).perform(context),
        'publish_rate_hz': launch_float(context, 'traction_tendon_publish_rate_hz'),
        'state_topic': LaunchConfiguration('traction_tendon_state_topic').perform(context),
        'wrench_topic': LaunchConfiguration('traction_tendon_wrench_topic').perform(context),
        'frame_id': LaunchConfiguration('traction_tendon_frame_id').perform(context),
    }
    configure_traction_tendon(
        ros_parameters,
        enabled=traction_tendon_enabled,
        options=traction_tendon_options,
    )

    rewritten_config = tempfile.NamedTemporaryFile(
        mode='w',
        prefix='rbkairos_mujoco_mecanum_drive_',
        suffix='.yaml',
        delete=False,
    )
    with rewritten_config:
        yaml.safe_dump(parameters, rewritten_config)

    mujoco_node = Node(
        package='mujoco_sim_ros2',
        executable='mujoco_sim',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            rewritten_config.name,
        ]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    controller_spawner = ExecuteProcess(
        cmd=[
            'ros2',
            'run',
            'controller_manager',
            'spawner',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '10.0',
            '--service-call-timeout',
            '10.0',
            '--switch-timeout',
            '5.0',
            'joint_state_broadcaster',
            'mecanum_base_controller',
            'ur5_joint_trajectory_controller',
        ],
        output='screen'
    )

    hold_handlers = []
    if launch_bool(context, 'initialize_arm_hold'):
        positions = initial_arm_positions(robot_description.evaluate(context))
        arm_hold = Node(
            package='rbkairos', executable='arm_preset_commander.py',
            output='screen',
            arguments=['--positions', *[str(x) for x in positions],
                       '--duration', '0.1', '--settle-time', '0.2',
                       '--startup-timeout', '5.0'],
        )

        def after_spawner(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason='Controller spawner failed'))]
            return [arm_hold]

        def after_arm_hold(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason='Initial arm hold command failed'))]
            return []

        hold_handlers = [
            RegisterEventHandler(OnProcessExit(target_action=controller_spawner,
                                               on_exit=after_spawner)),
            RegisterEventHandler(OnProcessExit(target_action=arm_hold,
                                               on_exit=after_arm_hold)),
        ]

    return [
        *hold_handlers,
        mujoco_node,
        robot_state_publisher,
        RegisterEventHandler(
            OnProcessStart(
                target_action=mujoco_node,
                on_start=[
                    TimerAction(
                        period=controller_load_delay_s,
                        actions=[controller_spawner],
                    )
                ],
            )
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'initialize_arm_hold', default_value='true',
            description='After controller activation, command the URDF initial arm pose',
        ),
        DeclareLaunchArgument(
            'urdf_file',
            default_value='urdf/rbkairos.urdf',
            description=(
                'URDF or xacro file path; relative paths resolve under the rbkairos package share'
            )
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='config/mujoco_mecanum_drive.yaml',
            description=(
                'MuJoCo/control config YAML; relative paths resolve under '
                'the rbkairos package share'
            )
        ),
        DeclareLaunchArgument(
            'model_package',
            default_value='rbkairos',
            description='Package used by mujoco_sim_ros2 for relative model_file lookup'
        ),
        DeclareLaunchArgument(
            'model_file',
            default_value='mujoco/rbkairos.xml',
            description=(
                'MuJoCo model file path; absolute paths are accepted for candidate-local models'
            )
        ),
        DeclareLaunchArgument(
            'traction_tendon_enabled',
            default_value='false',
            description='Load the read-only traction tendon observer'
        ),
        DeclareLaunchArgument(
            'traction_tendon_name',
            default_value='traction_cable_x',
            description='Limited MuJoCo tendon observed during the traction experiment'
        ),
        DeclareLaunchArgument(
            'traction_tendon_sensor_name',
            default_value='traction_cable_x_limit_force',
            description='MuJoCo tendonlimitfrc sensor observed during the traction experiment'
        ),
        DeclareLaunchArgument(
            'traction_tendon_attach_site_name',
            default_value='traction_attach_x',
            description='Robot-side traction tendon site'
        ),
        DeclareLaunchArgument(
            'traction_tendon_anchor_site_name',
            default_value='traction_anchor_x',
            description='World-side traction tendon site'
        ),
        DeclareLaunchArgument(
            'traction_tendon_publish_rate_hz',
            default_value='100.0',
            description='Traction tendon observer publish rate [Hz]'
        ),
        DeclareLaunchArgument(
            'traction_tendon_state_topic',
            default_value='/traction_tendon/state',
            description='Traction tendon state topic'
        ),
        DeclareLaunchArgument(
            'traction_tendon_wrench_topic',
            default_value='/traction_tendon/wrench',
            description='Traction tendon wrench topic'
        ),
        DeclareLaunchArgument(
            'traction_tendon_frame_id',
            default_value='world',
            description='Frame used by the traction tendon wrench'
        ),
        DeclareLaunchArgument(
            'controller_load_delay_s',
            default_value='2.0',
            description='Delay before loading ros2_control controllers after MuJoCo starts'
        ),
        OpaqueFunction(function=launch_setup),
    ])

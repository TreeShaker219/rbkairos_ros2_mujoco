from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def _spawn_controller(context, *args, **kwargs):
    controller_name = LaunchConfiguration("controller_name").perform(context)
    controller_manager = LaunchConfiguration("controller_manager").perform(context)
    controller_type = LaunchConfiguration("controller_type").perform(context)
    param_file = LaunchConfiguration("param_file").perform(context)
    inactive = LaunchConfiguration("inactive").perform(context).lower() in ("1", "true", "yes")

    spawner_args = [
        controller_name,
        "--controller-manager",
        controller_manager,
        "--controller-type",
        controller_type,
        "--param-file",
        param_file,
    ]
    if inactive:
        spawner_args.append("--inactive")

    return [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=spawner_args,
            output="screen",
        )
    ]


def generate_launch_description():
    default_param_file = PathJoinSubstitution([
        FindPackageShare("rbkairos_real_frequency_response"),
        "config",
        "base_twist_controller.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("controller_name", default_value="rbkairos_base_twist_controller"),
        DeclareLaunchArgument("controller_manager", default_value="/controller_manager"),
        DeclareLaunchArgument(
            "controller_type",
            default_value="rbkairos_real_frequency_response/BaseTwistController",
        ),
        DeclareLaunchArgument("param_file", default_value=default_param_file),
        DeclareLaunchArgument("inactive", default_value="true"),
        OpaqueFunction(function=_spawn_controller),
    ])

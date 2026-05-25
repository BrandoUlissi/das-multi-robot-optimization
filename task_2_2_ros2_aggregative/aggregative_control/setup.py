from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'aggregative_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, glob("launch_folder/aggregative_launch.py")),
        ("share/" + package_name, glob("launch_folder/vis_launch.py")),
        ("share/" + package_name, glob("config/aggregative_view.rviz")),
        ("share/" + package_name, glob("meshes/Quadcopter_model.stl")),
        ("share/" + package_name, glob("data/agent_data_log.csv")),
        ("share/" + package_name, glob("data/target_positions.txt")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dan',
    maintainer_email='dan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent = aggregative_control.aggregative_agent:main',
            'visualizer = aggregative_control.visualizer:main',
            'supervisor = aggregative_control.supervisor:main',
        ],
    },
)

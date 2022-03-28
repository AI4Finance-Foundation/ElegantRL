"""
This script tests whether or not each Isaac Gym environment can be effectively
instantiated.
"""

import isaacgym
import unittest
from elegantrl.envs.IsaacGym import *
from elegantrl.envs.isaac_tasks import isaacgym_task_map
from subprocess import call


class TestIsaacEnvironments(unittest.TestCase):
    def setUp(self):
        self.task_map = isaacgym_task_map

    def test_should_instantiate_all_Isaac_vector_environments(self):
        for env_name in self.task_map:
            return_code = call(
                ["python3", "unit_tests/isaac_env_test_helper.py", env_name]
            )
            if return_code != 0:
                raise Exception(
                    f"Instantiating {env_name} resulted in error code {return_code}"
                )


if __name__ == "__main__":
    unittest.main()

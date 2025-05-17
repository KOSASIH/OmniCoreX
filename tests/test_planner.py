import unittest
from planner import Planner, PlanStep

class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = Planner(tool_registry={
            "web_search": lambda query: {"results": ["res1", "res2"]},
            "calculator": lambda expression: {"result": eval(expression)}
        })

    def test_generate_plan_and_execution(self):
        plan = self.planner.generate_plan("search AI papers")
        self.assertIsInstance(plan, list)
        for step in plan:
            result = self.planner.execute_step(step)
            self.assertIsNotNone(result)

    def test_interrupt_replan(self):
        new_plan = self.planner.interrupt_and_replan("new context", "calculate 2 + 2")
        self.assertIsInstance(new_plan, list)

if __name__ == "__main__":
    unittest.main()

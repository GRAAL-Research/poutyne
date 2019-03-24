import unittest

from poutyne.framework.callbacks.policies import linspace, cosinespace
from poutyne.framework.callbacks.policies import Phase, OptimizerPolicy
from poutyne.framework.callbacks.policies import one_cycle_phases, sgdr_phases


class TestSpaces(unittest.TestCase):
    def assert_space(self, space, expected):
        for val, exp in zip(space, expected):
            self.assertAlmostEqual(val, exp, places=3)

    def test_linspace_const(self):
        self.assert_space(linspace(0, 0, 3), [0, 0, 0])

    def test_linspace_increasing(self):
        self.assert_space(linspace(0, 1, 3), [0, .5, 1])

    def test_linspace_decreasing(self):
        self.assert_space(linspace(1, 0, 3), [1, .5, 0])

    def test_linspace_with_many_values(self):
        self.assert_space(linspace(0, 1, 6), [0, .2, .4, .6, .8, 1])

    def test_cosinespace_const(self):
        self.assert_space(cosinespace(0, 0, 2), [0, 0])

    def test_cosinespace_increasing(self):
        self.assert_space(cosinespace(0, 1, 2), [0, 1])
        self.assert_space(cosinespace(0, 1, 3), [0, .5, 1])

    def test_cosinespace_decreasing(self):
        self.assert_space(cosinespace(1, 0, 2), [1, 0])
        self.assert_space(cosinespace(1, 0, 3), [1, .5, 0])

    def test_cosinespace_with_many_values(self):
        self.assert_space(cosinespace(0, 1, 5), [0, .1464, .5, .8535, 1])
        self.assert_space(cosinespace(1, 0, 5), [1, .8535, .5, .1464, 0])

    def test_space_has_desired_legth(self):
        for space_fn in [linspace, cosinespace]:
            space = list(space_fn(2, 1, 3))
            assert len(space) == 3
            with self.assertRaises(IndexError):
                space[4]  # pylint: disable=pointless-statement


class TestPhase(unittest.TestCase):
    def test_init_raises_without_lr_or_momentum(self):
        with self.assertRaises(ValueError):
            Phase(lr=None, momentum=None)
        with self.assertRaises(ValueError):
            Phase()

    def test_phase_with_only_one_parameter_set(self):
        for param_name in ["lr", "momentum"]:
            steps = 3
            phase = Phase(**{param_name: linspace(1, 0, steps)})
            for params in phase:
                self.assertIsInstance(params, dict)
                self.assertTrue(param_name in params)
                self.assertEqual(len(params), 1)
                self.assertTrue(0 <= params[param_name] <= 1)

    def test_phase_with_two_parameters(self):
        steps = 4
        phase = Phase(lr=linspace(1, 0, steps), momentum=cosinespace(.8, 1, steps))
        self.assertEqual(len(list(phase)), steps)
        for params in phase:
            self.assertEqual(len(params), 2)

            self.assertTrue("lr" in params)
            self.assertTrue(0 <= params["lr"] <= 1)

            self.assertTrue("momentum" in params)
            self.assertTrue(.8 <= params["momentum"] <= 1)


class TestOptimizerPolicy(unittest.TestCase):
    def setUp(self):
        steps = 2
        phases = [Phase(lr=linspace(1, 1, steps)), Phase(lr=linspace(0, 0, steps))]
        self.policy = OptimizerPolicy(phases)

    def test_basic_iteration(self):
        policy_iter = iter(self.policy)
        self.assertEqual(next(policy_iter), {"lr": 1})
        self.assertEqual(next(policy_iter), {"lr": 1})

        self.assertEqual(next(policy_iter), {"lr": 0})
        self.assertEqual(next(policy_iter), {"lr": 0})

        with self.assertRaises(StopIteration):
            next(policy_iter)


class TestOptimizerPolicyRestart(unittest.TestCase):
    def setUp(self):
        steps = 2
        phases = [
            Phase(lr=linspace(0, 0, steps)),
            Phase(lr=linspace(1, 1, steps)),
            Phase(lr=linspace(2, 2, steps)),
        ]
        self.policy = OptimizerPolicy(phases=phases, initial_step=3)

    def test_starts_at_correct_position(self):
        policy_iter = iter(self.policy)
        # The first three steps are ignored
        # assert next(policy_iter) == {"lr": 0}
        # assert next(policy_iter) == {"lr": 0}
        # assert next(policy_iter) == {"lr": 1}
        assert next(policy_iter) == {"lr": 1}
        assert next(policy_iter) == {"lr": 2}
        assert next(policy_iter) == {"lr": 2}

        with self.assertRaises(StopIteration):
            next(policy_iter)


class TestOneCycle(unittest.TestCase):
    def test_length(self):
        policy = OptimizerPolicy(one_cycle_phases(100))
        self.assertEqual(len(list(policy.all_steps())), 100)

        policy = OptimizerPolicy(one_cycle_phases(99))
        self.assertEqual(len(list(policy.all_steps())), 99)


class TestSGDR(unittest.TestCase):
    def test_length_with_cycle_mult_one(self):
        policy = OptimizerPolicy(sgdr_phases(10, 1, cycle_mult=1))
        self.assertEqual(len(list(policy.all_steps())), 10)

        policy = OptimizerPolicy(sgdr_phases(10, 2, cycle_mult=1))
        self.assertEqual(len(list(policy.all_steps())), 20)

        policy = OptimizerPolicy(sgdr_phases(10, 10, cycle_mult=1))
        self.assertEqual(len(list(policy.all_steps())), 100)

    def test_length_with_higher_cycle_mult(self):
        policy = OptimizerPolicy(sgdr_phases(10, 1, cycle_mult=2))
        self.assertEqual(len(list(policy.all_steps())), 10)

        policy = OptimizerPolicy(sgdr_phases(10, 2, cycle_mult=2))
        self.assertEqual(len(list(policy.all_steps())), 30)

        policy = OptimizerPolicy(sgdr_phases(10, 3, cycle_mult=2))
        self.assertEqual(len(list(policy.all_steps())), 70)

        policy = OptimizerPolicy(sgdr_phases(10, 1, cycle_mult=3))
        self.assertEqual(len(list(policy.all_steps())), 10)

        policy = OptimizerPolicy(sgdr_phases(10, 2, cycle_mult=3))
        self.assertEqual(len(list(policy.all_steps())), 40)

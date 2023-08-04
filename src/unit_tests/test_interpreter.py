import unittest
import jax.numpy as jnp
import symbolic_neural_networks.interpreter as interpreter

class TestArchitecture(unittest.TestCase):

    sut = interpreter.Interpreter()

	def test_input_to_tensor(self):
		# Given
		problem = prob.Problem(
            max_predicates = [0, 2, 3],
            max_body = 3,
            predicate_names = [[], 
                               ['male', 'female'],
                               ['sibling', 'brother', 'sister']],
            knowledge_base = [],
            objects = ['alice', 'bob', 'carol', 'dave']

		input = ['male(dave).',
                 'female(carol).',
                 'sibling(dave, carol).',
                 'sibling(carol, dave).',
                 'brother(bob, alice).',
                 'sister(alice, bob).']

		# When

		# Then

if __name__ == '__main__':
    unittest.main()
import unittest
import neural_logic_machines.instance as instance
import neural_logic_machines.problem as problem
import jax.numpy as jnp

class TestInstance(unittest.TestCase):

    facts = ['female(alice).',
             'male(bob).',
             'female(carol).',
             'male(dave).',
             'sibling(bob, carol).',
             'sibling(alice, dave).']
    max_predicates = [1, 2, 2]
    predicate_names = [['family'],
                       ['male', 'female'],
                       ['sibling', 'brother']]
    knowledge_base = []
    problem = problem.Problem(max_predicates,
                              predicate_names,
                              knowledge_base)
    sut = instance.Instance(problem, facts)

    def test_init(self):
        self.assertEqual(self.sut.max_predicates, [1, 2, 2])
        self.assertEqual(self.sut.predicate_names, [['family'],
                                                    ['male', 'female'],
                                                    ['sibling', 'brother']])
        self.assertEqual(self.sut.objects, ['alice', 'bob', 'carol', 'dave'])
    
    def test_get_objects(self):
        # When
        result = self.sut.objects

        # Then
        self.assertEqual(result, ['alice', 'bob', 'carol', 'dave'])

    def test_text_to_tensor(self):
        # Given
        text = ['female(alice).',
                'male(bob).',
                'female(carol).',
                'male(dave).',
                'sibling(bob, carol).',
                'sibling(alice, dave).']

        # When
        result = self.sut.text_to_tensor(text)

        # Then
        self.assertTrue(result[0] == jnp.array([0]))
        self.assertTrue(
            (result[1] == jnp.array([[0, 1, 0, 1],
                                     [1, 0, 1, 0]])).all())
        self.assertTrue(
            (result[2] == jnp.array([[[0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [1, 0, 0, 0]],
                                     [[0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0]]])).all())
        
    def test_tensor_to_text(self):
        # Given
        tensor =[jnp.array([0]),
                 jnp.array([[0, 1, 0, 1],
                            [1, 0, 1, 0]]),
                 jnp.array([[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]])]

        # When
        result = self.sut.tensor_to_text(tensor)

        # Then
        self.assertEqual(result, ['male(bob).',
                                  'male(dave).',
                                  'female(alice).',
                                  'female(carol).',
                                  'sibling(bob, carol).',
                                  'sibling(alice, dave).'])

    def test_arguments_binary(self):
        # Given
        fact = 'sibling(alice, dave).'

        # When
        result = self.sut.get_arguments(fact)

        # Then
        self.assertListEqual(result, [0, 3])

    def test_get_arguments_nullary(self):
        # Given
        fact = 'family.'

        # When
        result = self.sut.get_arguments(fact)

        # Then
        self.assertEqual(result, [])

    def test_get_head_binary(self):
        # Given
        fact = 'brother(dave, alice).'
        arity = 2

        # When
        result = self.sut.get_head(fact, arity)

        # Then
        self.assertEqual(result, 1)

    def test_get_head_nullary(self):
        # Given
        fact = 'family.'
        arity = 0

        # When
        result = self.sut.get_head(fact, arity)

        # Then
        self.assertEqual(result, 0)

    def test_get_head_error(self):
        # Given
        fact = 'sister(alice, dave).'
        arity = 2

        # Then
        with self.assertRaises(RuntimeError):
            # When
            self.sut.get_head(fact, arity)

if __name__ == '__main__':
    unittest.main()
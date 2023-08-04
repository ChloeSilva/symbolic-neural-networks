import unittest
import neural_logic_machines.problem as problem
import jax.numpy as jnp

class TestInstance(unittest.TestCase):

    facts = ['female(alice).',
             'male(bob).',
             'female(carol).',
             'male(dave).',
             'sibling(bob, carol).',
             'sibling(alice, dave).']
    max_predicates = [0, 2, 3]
    predicate_names = [[],
                       ['male', 'female'],
                       ['sibling', 'brother']]
    knowledge_base = ['sibling(X, Y) :- sibling(Y, X).']
    sut = problem.Problem(max_predicates,
                          predicate_names,
                          knowledge_base)

    def test_init(self):
        self.assertEqual(self.sut.max_predicates, [0, 2, 3])
        self.assertEqual(self.sut.predicate_names, [[],
                                                    ['male', 'female'],
                                                    ['sibling', 'brother', 'p0']])
        self.assertEqual(self.sut.knowledge_base, ['sibling(X, Y) :- sibling(Y, X).'])
        self.assertEqual(self.sut.max_internal, [2, 5, 5])

    def test_generate_names(self):
        # Given
        predicate_names = [[],
                           ['male', 'female'],
                           ['sibling', 'brother']]

        # When
        result = self.sut.generate_names(predicate_names)

        # Then
        self.assertEqual(result, [[],
                                  ['male', 'female'],
                                  ['sibling', 'brother', 'p0']])

    def test_get_max_internal(self):
        # Given
        max_predicates = [0, 2, 3]

        # When
        result = self.sut.get_max_internal(max_predicates)

        # Then
        self.assertEqual(result, [2, 5, 5])

    def test_create_instance(self):
        # Given
        facts = ['female(alice).',
                 'male(bob).',
                 'female(carol).',
                 'male(dave).',
                 'sibling(bob, carol).',
                 'sibling(alice, dave).']

        # When
        result = self.sut.create_instance(facts)

        # Then
        self.assertEqual(result.objects, ['alice', 'bob', 'carol', 'dave'])

    def test_text_to_tensor(self):
        # Given
        input = ['female(alice).',
                 'male(bob).',
                 'female(carol).',
                 'male(dave).',
                 'sibling(bob, carol).',
                 'sibling(alice, dave).']

        output = ['female(alice).',
                  'male(bob).',
                  'female(carol).',
                  'male(dave).',
                  'sibling(bob, carol).',
                  'sibling(alice, dave).',
                  'brother(bob, carol).']

        # When
        result = self.sut.text_to_tensor(input, output)

        # Then
        self.assertTrue(result[0][0].size == 0)
        self.assertTrue((result[0][1] == jnp.array([[0, 1, 0, 1],
                                                    [1, 0, 1, 0]])).all())
        self.assertTrue((result[0][2] == jnp.array([[[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [1, 0, 0, 0]],
                                                    [[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 0, 0, 0]],
                                                    [[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 0, 0, 0]]])).all())
        self.assertTrue(result[1][0].size == 0)
        self.assertTrue((result[1][1] == jnp.array([[0, 1, 0, 1],
                                                    [1, 0, 1, 0]])).all())
        self.assertTrue((result[1][2] == jnp.array([[[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [1, 0, 0, 0]],
                                                    [[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 0]],
                                                    [[0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 0, 0, 0],
                                                     [0, 0, 0, 0]]])).all())

if __name__ == '__main__':
    unittest.main()
import unittest
import jax.numpy as jnp
import neural_logic_machines.architecture as architecture

class TestArchitecture(unittest.TestCase):

    #sut = architecture.Architecture()

    def test_permute_unary(self):
        # Given
        predicates = jnp.array([[1, 0, 0, 0]])

        # When
        result = architecture.permute_predicate(predicates)

        # Then
        self.assertTrue((result == jnp.array([[1, 0, 0, 0]])).all())

    def test_permute_binary(self):
        # Given
        predicates = jnp.array([[[0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]]])
        
        # When
        result = architecture.permute_predicate(predicates)

        # Then
        self.assertTrue(
            (result ==
             jnp.array([[[0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]],
                       [[0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]]])).all())
        
    def test_expand(self):
        # Given
        predicates = jnp.array([[0, 1, 0, 1], [0, 0, 1, 1]])

        # When
        result = architecture.expand(predicates)

        # Then
        self.assertTrue(
            (result ==
             jnp.array([[[0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 0, 1]],
                       [[0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 1, 1]]])).all())
        
    def test_reduce(self):
        # Given
        predicates = jnp.array([[[1, 0, 1, 0],
                                [0, 0, 1, 1],
                                [0, 0, 1, 0],
                                [0, 0, 1, 1]],
                               [[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]]])

        # When
        result = architecture.reduce(predicates)

        # Then
        self.assertTrue((result ==
             jnp.array([[1, 0, 1, 1], 
                       [0, 1, 0, 0]])).all())
        
    def test_predict_1(self):
        # Given      
        weights = [[([], []), ([], []), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [1, 0, 0, 0, 0, 0]]),
                                          jnp.array([0, 0, 0]))],
                   [([], []), ([], []), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0]]),
                                          jnp.array([0, 0, 0]))]]
        
        facts = [jnp.empty((0)),
                 jnp.empty((0, 4)),
                 jnp.array([[[0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]])]

        # When
        result = architecture.predict_nlm(weights, facts)

        # Then
        self.assertTrue(result[0].size == 0)
        self.assertTrue(result[1].size == 0)
        self.assertTrue((result[2] == jnp.array([[[0, 1, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0]],
                                                 [[0, 0, 0, 0],
                                                  [1, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 1, 0]],
                                                 [[0, 1, 0, 0],
                                                  [1, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 1, 0]]])).all())

    def test_predict_2(self):
        # Given
        # layer 1: sibling(X, Y) :- sibling(Y, X)
        # layer 2: brorther(X, Y) :- male(X), sibling(X, Y)
        weights = [[([], []), 
                    (jnp.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), 
                     jnp.array([0, 0, 0, 0, 0])), 
                    (jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]), 
                     jnp.array([0, 0, 0, 0, 0]))],
                   [([], []), 
                    (jnp.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), 
                     jnp.array([0, 0, 0, 0, 0])), 
                    (jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
                     jnp.array([0, 0, -1, 0, 0]))]]
        
        # female(alice).
        # male(bob).
        # female(carol).
        # male(dave).
        # sibling(alice, bob).
        # sibling(carol, dave).
        facts = [jnp.empty((0)),
                 jnp.array([[0, 1, 0, 1],
                            [1, 0, 1, 0]]),
                 jnp.array([[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0]]])]

        # When
        result = architecture.predict_nlm(weights, facts)

        # Then
        self.assertTrue(result[0].size == 0)
        self.assertTrue((jnp.round(result[1]) == jnp.array([[0, 1, 0, 1],
                                                 [1, 0, 1, 0]])).all())
        self.assertTrue((jnp.round(result[2]) == jnp.array([[[0, 1, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0]],
                                                 [[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]],
                                                 [[0, 1, 0, 0],
                                                  [1, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 1, 0]]])).all())  

    def test_update(self):
        # Given
        weights = [[([], []), ([], []), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]).astype(jnp.float32),
                                          jnp.array([0, 0, 0]).astype(jnp.float32))],
                   [([], []), ([], []), (jnp.array([[0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]).astype(jnp.float32),
                                          jnp.array([0, 0, 0]).astype(jnp.float32))]]
        
        facts = [jnp.empty((0)),
                 jnp.empty((0, 4)),
                 jnp.array([[[0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]],
                            [[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0]],
                            [[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]]]).astype(jnp.float32)]
        
        target_facts = [jnp.empty((0)),
                        jnp.empty((0, 4)),
                        jnp.array([[[0, 1, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0]],
                                    [[0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 1, 0]],
                                    [[0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0]]]).astype(jnp.float32)]

        learning_rate = 0.1
        
        # When
        for _ in range(300):
            weights = architecture.update(weights, facts, target_facts, learning_rate)

        # Then
        self.assertTrue((target_facts[2] == jnp.round(architecture.predict_nlm(weights, facts)[2])).all())

if __name__ == '__main__':
    unittest.main()
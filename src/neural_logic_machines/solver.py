import neural_logic_machines.architecture as architecture
import neural_logic_machines.interpreter as interpreter_
import math
import time
import jax.numpy as jnp
from itertools import groupby
from jax import random

class Solver:

    solution = []
    #random_key = random.PRNGKey(0)

    def __init__(self, problem):
        self.problem = problem
        self.interpreter = interpreter_.Interpreter()

    # Trains the neural network and saves the final parametes in solution  
    def train(self, training_path, learning_rate=1e-2, batch_size=100, num_epochs=1):
        with open(training_path) as f:
            training_data = [line.strip() for line in f]
    
        processed_data = self.process(training_data)
        io_tensors = [self.problem.text_to_tensor(i, o) 
                      for i, o in processed_data]
        
        weights = self.init_weights(random.PRNGKey(0))
        weights = [[(jnp.zeros(w.shape), jnp.zeros(b.shape)) for w, b in l] for l in weights]
        weights[0][2] = weights[0][2][0].at[(4, 0)].set(0.9), weights[0][2][1].at[4].set(-0.9)
        weights[0][2] = weights[0][2][0].at[(4, 5)].set(0.9), weights[0][2][1]

        weights[1][2] = weights[0][2][0].at[(4, 0)].set(0.9), weights[1][2][1].at[4].set(-0.9)
        weights[1][2] = weights[0][2][0].at[(4, 7)].set(0.9), weights[1][2][1]

        for epoch in range(num_epochs):
            for i, o in self.get_batches(io_tensors, batch_size):
                weights = architecture.update(weights, i, o, learning_rate)
                print(f'layer 1: {weights[0][2]}')
                print(f'layer 2: {weights[1][2]}')
                print(f'epoch: {epoch}')
    
        self.solution = weights
    
    # Runs the network on a single input and returns the output
    def run(self, input_path, threshold):
        with open(input_path) as f:
            input_data = [line.strip() for line in f]

        instance = self.problem.create_instance(input_data)
        input_tensor = instance.text_to_tensor(input_data)
        output = architecture.predict_nlm(self.solution, input_tensor)
        output = [jnp.where(o > threshold, 1, 0) for o in output]

        return instance.tensor_to_text(output)
    
    # Runs the neural network on the test data and returns the accuracy
    def test(self, test_path, threshold):
        with open(test_path) as f:
            test_data = [line.strip() for line in f]

        processed_data = self.process(test_data)
        io_tensors = [self.problem.text_to_tensor(i, o) 
                      for i, o in processed_data]
        
        correct = 0
        for i, expected_o in  io_tensors:
            o = architecture.predict_nlm(self.solution, i)
            o = [jnp.where(o > threshold, 1.0, 0.0) for o in o]
            if all([jnp.array_equal(a, b) for a, b in zip(o, expected_o)]):
                correct += 1

        return correct/len(io_tensors)

    def interpret(self, threshold):
        # Interprets the solution as a logic program and returns it as text
        pass

    def process(self, d: list[str]) -> list[list[list[str]]]:
        d = list(filter(lambda z: z != '', d))
        d = [tuple(y) for x, y in groupby(d, lambda z: z == 'in:') if not x]
        d = [[list(y) for x, y in groupby(p, lambda z: z == 'out:') if not x] for p in d]
        return d
    
    def get_batches(self, data, size):
        return [list(zip(*(data[i:i + size]))) for i in range(0, len(data), size)]
 
                 
class NLM(Solver):
    
    def __init__(self, problem, depth):
        super().__init__(problem)
        self.depth = depth
        #self.predictor = self.arch.predict_nlm

    def init_neural_unit(self, num_preds, arity, key):
        w_key, b_key = random.split(key)
        # return (random.uniform(w_key, (num_preds, num_preds*math.factorial(arity))),
        #         random.uniform(b_key, (num_preds,)))
        return (random.uniform(w_key, (num_preds, num_preds*math.factorial(arity))),
                jnp.zeros((num_preds,)))
    
    def init_layer(self, key):
        num_units = len(self.problem.max_internal)
        keys = random.split(key, num_units)
        return [self.init_neural_unit(m, n, k) for m, n, k in 
                zip(self.problem.max_internal, range(num_units), keys)]

    def init_weights(self, key):
        keys = random.split(key, self.depth)
        return [self.init_layer(k) for k in keys]
        

class SLNLM(Solver):
    
    def __init__(self, problem, max_rules):
        Solver.__init__(problem)
        self.max_rules = max_rules
        #self.predictor = self.arch.predict_slnlm

    
class NNLM(Solver):

    def __init__(self, problem, max_rules):
        Solver.__init__(problem)
        self.max_rules = max_rules
        #self.predictor = self.arch.predict_nnlm
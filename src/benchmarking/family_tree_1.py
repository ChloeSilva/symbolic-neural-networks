from neural_logic_machines import problem
from neural_logic_machines import solver

problem_1 = problem.Problem(
        max_predicates = [0, 2, 3],
        predicate_names = [[],
                           ['male', 'female'],
                           ['parent', 'son', 'daughter']],
        knowledge_base = [])
    
def main():
    nlm  = solver.NLM(problem_1, depth=1)
    for i in range(10):
        nlm.train('src/data/family_tree/training_1.txt', learning_rate=0.1, 
              batch_size=100, num_epochs=1000, seed=i)
    accuracy = nlm.test('src/data/family_tree/test_1.txt', threshold=0.9)
    #program = nlm.interpret(threshold=0.9)
    program = 'placeholder'

    print(f'learned program:\n{program}\nwith accuracy: {accuracy}')


if __name__ == "__main__":
    main()
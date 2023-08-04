from itertools import permutations
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, vmap

#class Architecture():

def generate_permutations(n: int) -> list[tuple[int, ...]]:
    return list(permutations(range(n)))

def permute_predicate(preds: jnp.ndarray) -> jnp.ndarray:
    perm = generate_permutations(preds.ndim - 1)
    return jnp.array(sum([[jnp.transpose(pred, p) for p in perm] 
                            for pred in preds], []))

def expand(preds: jnp.ndarray) -> jnp.ndarray:
    objects = preds.shape[-1]
    tile_shape = (1, ) * preds.ndim + (objects, )
    final_shape = preds.shape + (objects, )
    return jnp.reshape(jnp.tile(preds, tile_shape), final_shape)

def reduce(preds: jnp.ndarray) -> jnp.ndarray:
    return preds.max(1)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1/(1+jnp.exp(-x))

def expand_and_reduce(facts: list[jnp.ndarray]) -> list[jnp.ndarray]:
    new_facts = []
    for arity in range(len(facts)):
        temp = facts[arity]
        if arity != 0 and len(facts[arity-1]) != 0:
            temp = jnp.concatenate((expand(facts[arity-1]), temp))

        if arity != len(facts)-1 and len(facts[arity+1]) != 0:
            temp = jnp.concatenate((temp, reduce(facts[arity+1])))
        
        new_facts.append(temp)

    return new_facts

# If we also expand/reduce the outputs we can save an auxillary
# predicate with 'non matching' variables
def predict_nlm(weights: list[jnp.ndarray], facts: list[jnp.ndarray]) -> list[jnp.ndarray]:
    #print(f'predicting')
    # print(f'weights: {weights}')
    # print(f'facts: {facts}')
    num_pred = [len(fact) for fact in facts]
    big_facts = facts
    for layer in weights:
        #print('processing layer')
        big_facts = expand_and_reduce(big_facts)
        new_facts = []
        for arity in range(len(layer)):
            #print('processing arity')
            #print(f'big_facts: {big_facts}')
            if len(layer[arity][0]) == 0:
                new_facts.append(facts[arity])
                continue
            w, b = layer[arity]
            perm = permute_predicate(big_facts[arity])
            ts = tuple(range(1, arity)) + (0, arity)
            if arity == 0: ts = (0, )
            #print(f'perm: {perm}, ts:\n{ts}')
            applied = jnp.dot(w,  jnp.transpose(perm, ts))
            # maybe add sigmoid to p+b
            outputs = [jnp.maximum(p+b, i) 
                        for p, b, i in zip(applied, b, big_facts[arity])]
            if arity > 0:
                outputs = outputs[num_pred[arity-1]:]
            if arity < len(layer) - 1:
                outputs = outputs[:-num_pred[arity+1]]
            # outputs = [jnp.maximum(o, f) for o, f in zip(outputs, facts[arity])]
            new_facts.append(jnp.array(sum([outputs], [])))
        big_facts = new_facts

    return [jnp.maximum(0, big_fact) for big_fact in big_facts]

batched_predict_nlm = vmap(predict_nlm, in_axes=(None, 0))


def loss(weights, x, y):
    prediction = batched_predict_nlm(weights, x)
    temp = [-jnp.mean((p-y)*(p-y)) for p, y in zip(prediction, y)]
    return sum(temp) / len(temp)

def new_weights(learning_rate, w, b, dw, db):
        if len(w) == 0:
            return (w, b)
        return (w + learning_rate * dw, b + learning_rate * db)

#@partial(jit, static_argnums=(0,))
@jit
def update(weights, x, y, learning_rate):
    for example in x:
        print(f'len(example): {len(example)}')
        for arity in example:
            print(f'arity shape: {jnp.shape(arity)}')
    grads = grad(loss)(weights, x, y)
    return [[new_weights(learning_rate, w, b, dw, db) for ((w, b), (dw, db)) 
                in zip(lw, lg)] for lw, lg in zip(weights, grads)]


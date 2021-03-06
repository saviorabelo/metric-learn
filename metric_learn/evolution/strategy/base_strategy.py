from deap import base, tools

import numpy as np

from sklearn.model_selection import train_test_split

from .individual import Individual
from .mfitness import MultidimensionalFitness


class BaseEvolutionStrategy(object):
    def __init__(self, n_gen=25, split_size=0.33, train_subset_size=1.0,
                 stats=None, max_workers=1, random_state=None, verbose=False):
        self.n_gen = n_gen
        self.split_size = split_size
        self.train_subset_size = train_subset_size
        self.stats = stats
        self.max_workers = max_workers

        self.random_state = random_state
        self.verbose = verbose

        np.random.seed(random_state)

    def inject_params(self, n_dim, fitness, transformer,
                      random_state=None, verbose=False):
        self.n_dim = n_dim
        self.fitness = fitness
        self.transformer = transformer

        self.random_state = random_state
        self.verbose = verbose

        np.random.seed(random_state)

    def fit(self, X, y):
        raise NotImplementedError('fit() is not implemented')

    def best_individual(self):
        raise NotImplementedError('best_individual() is not implemented')

    def _build_stats(self):
        if self.stats is None:
            return None
        elif isinstance(self.stats, tools.Statistics):
            return self.stats
        elif self.stats == 'identity':
            fitness = tools.Statistics(key=lambda ind: ind)
            fitness.register("id", lambda ind: ind)
            return fitness

        fitness = tools.Statistics(key=lambda ind: ind.fitness.values)
        fitness.register("avg", np.mean, axis=0)
        fitness.register("std", np.std, axis=0)
        fitness.register("min", np.min, axis=0)
        fitness.register("max", np.max, axis=0)
        return fitness

    def _subset_train_test_split(self, X, y):
        subset = self.train_subset_size
        assert(0.0 < subset <= 1.0)

        if subset == 1.0:
            return train_test_split(
                X, y,
                test_size=self.split_size,
                random_state=self.random_state,
            )

        train_mask = np.random.choice(
            [True, False],
            X.shape[0],
            p=[subset, 1 - subset]
        )
        return train_test_split(
            X[train_mask], y[train_mask],
            test_size=self.split_size,
            random_state=self.random_state,
        )

    def generate_individual_with_fitness(self, func, n):
        fitness_len = len(self.fitness)
        ind = Individual(func() for _ in range(n))
        ind.fitness = MultidimensionalFitness(fitness_len)
        return ind

    def create_toolbox(self, X, y):
        toolbox = base.Toolbox()

        if self.max_workers != 1:
            import multiprocessing
            toolbox.register("map", multiprocessing.Pool(self.max_workers).map)

        self.X, self.y = X, y  # needed in self.evaluate function
        toolbox.register("evaluate", self.evaluate)

        return toolbox

    def cut_individual(self, individual):
        return individual

    def cleanup(self):
        del self.X
        del self.y

    def evaluate(self, individual):
        X_train, X_test, y_train, y_test = self._subset_train_test_split(
            self.X, self.y,
        )

        # transform the inputs if there is a transformer
        if self.transformer:
            transformer = self.transformer.duplicate_instance()
            transformer.fit(
                X_train,
                y_train,
                self.cut_individual(individual)
            )
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)

        return [f(X_train, X_test, y_train, y_test)
                for f in self.fitness]

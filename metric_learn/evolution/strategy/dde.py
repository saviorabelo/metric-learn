import itertools

import math

from deap import tools

import numpy as np

from .base_strategy import BaseEvolutionStrategy
from .individual import Individual
from .mfitness import MultidimensionalFitness


class DynamicDifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, population_size=10, population_regular=4,
                 population_brownian=2, cr=0.6, f=0.4, bounds=(-1.0, 1.0),
                 **kwargs):
        super(DynamicDifferentialEvolution, self).__init__(**kwargs)

        self.population_size = population_size
        self.cr = cr
        self.f = f
        self.bounds = bounds
        self.population_regular = population_regular
        self.population_brownian = population_brownian

    def best_individual(self):
        return self.hall_of_fame[0]

    def generate_brow_ind_with_fitness(self, best, sigma=0.3):
        fitness_len = len(self.fitness)
        ind = Individual(np.random.normal(x, sigma) for x in best)
        ind.fitness = MultidimensionalFitness(fitness_len)
        return ind

    def fit(self, X, y):
        # Differential evolution parameters
        individual_size = self.n_dim
        # Should be equal to the number of peaks
        population_size = self.population_size

        regular = self.population_regular
        brownian = self.population_brownian
        bounds = self.bounds

        toolbox = self.create_toolbox(X, y)
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register(
            "individual",
            self.generate_individual_with_fitness,
            toolbox.attr_float,
            individual_size)
        toolbox.register(
            "brownian_individual",
            self.generate_brow_ind_with_fitness)
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual)

        toolbox.register("select", np.random.choice, size=4)
        toolbox.register("best", tools.selBest, k=1)

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] \
            + (stats.fields if stats else [])

        # Initialize populations
        populations = [toolbox.population(n=regular + brownian)
                       for _ in range(population_size)]

        # Evaluate the individuals
        for idx, subpop in enumerate(populations):
            fitness = toolbox.map(toolbox.evaluate, subpop)
            for ind, fit in zip(subpop, fitness):
                ind.fitness.values = fit

        if stats:
            record = stats.compile(itertools.chain(*populations))
            self.logbook.record(gen=0, evals=len(populations), **record)
            if self.verbose:
                print(self.logbook.stream)

        for g in range(1, self.n_gen):
            # Detect a change and invalidate fitness if necessary
            bests = [toolbox.best(subpop)[0] for subpop in populations]
            if any(b.fitness.values != toolbox.evaluate(b) for b in bests):
                for individual in itertools.chain(*populations):
                    del individual.fitness.values

            # Apply exclusion
            rexcl = (bounds[1] - bounds[0]) \
                / (2 * population_size**(1.0 / individual_size))
            for i, j in itertools.combinations(range(population_size), 2):
                if bests[i].fitness.valid and bests[j].fitness.valid:
                    d = sum((bests[i][k] - bests[j][k])**2
                            for k in range(individual_size))
                    d = math.sqrt(d)

                    if d < rexcl:
                        if bests[i].fitness < bests[j].fitness:
                            k = i
                        else:
                            k = j

                        populations[k] = toolbox.population(
                            n=regular + brownian)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in itertools.chain(*populations)
                           if not ind.fitness.valid]
            fitness = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness):
                ind.fitness.values = fit

            all_pops = list(itertools.chain(*populations))
            self.hall_of_fame.update(all_pops)

            if stats:
                record = stats.compile(all_pops)
                self.logbook.record(gen=g, evals=len(populations), **record)
                if self.verbose:
                    print(self.logbook.stream)

            # Evolve the sub-populations
            for idx, subpop in enumerate(populations):
                newpop = []
                xbest, = toolbox.best(subpop)
                # Apply regular DE to the first part of the population
                for individual in subpop[:regular]:
                    idxs = np.random.choice(len(subpop), size=4)
                    x1, x2, x3, x4 = subpop[idxs[0]], subpop[idxs[1]], \
                        subpop[idxs[2]], subpop[idxs[3]]
                    offspring = toolbox.clone(individual)
                    index = np.random.randint(individual_size)
                    for i, _ in enumerate(individual):
                        if i == index or np.random.random() < self.cr:
                            offspring[i] = xbest[i] + self.f \
                                * (x1[i] + x2[i] - x3[i] - x4[i])
                    offspring.fitness.values = toolbox.evaluate(offspring)
                    if offspring.fitness >= individual.fitness:
                        newpop.append(offspring)
                    else:
                        newpop.append(individual)

                # Apply Brownian to the last part of the population
                newpop.extend(toolbox.brownian_individual(xbest)
                              for _ in range(brownian))

                # Evaluate the brownian individuals
                for individual in newpop[-brownian:]:
                    individual.fitness.value = toolbox.evaluate(individual)

                # Replace the population
                populations[idx] = newpop

        self.cleanup()
        return self

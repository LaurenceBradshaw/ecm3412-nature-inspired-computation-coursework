# This work evaluates the genetic algorithm on two bin-packing variants that differ only in how item weights scale. In BPP1, item i has weight i, resulting in a linear progression of weights across the 500 items. In BPP2, item i has weight 
# 𝑖^2/2, producing a quadratic growth pattern and greatly amplifying the disparity between light and heavy items. Consequently, BPP2 is substantially more difficult to balance than BPP1.
# Each candidate solution is represented as a fixed-length array of 500 integers. The i-th entry in the array corresponds to item i, and its value specifies the bin assignment for that item. Gene values range from 1 to b, where b = 10 for BPP1 and b = 50 for BPP2.
# The algorithm begins by generating a population of 100 chromosomes, with every gene initialised to a randomly selected valid bin. This provides a broad starting pool from which evolutionary search can proceed.
# Solution quality is evaluated by computing the total weight in each bin and measuring the difference d between the heaviest and lightest bins. Fitness is then calculated as
# fitness = 100/ 1 + 𝑑
# which rewards solutions with more balanced bin loads. The run terminates once 10,000 fitness evaluations have been completed.
# Parent selection uses tournament selection. For each parent, t individuals are drawn uniformly at random and the one with the highest fitness advances. Tournament sizes of 3 and 7 are used to compare lower and higher selection pressure.
# Uniform crossover is applied with probability 0.8. Each gene in the offspring is independently inherited from one of the two parents with equal likelihood. If crossover does not occur, the parents pass unchanged to the next generation.
# Mutation operates independently on each gene, randomly reassigning the corresponding item to any valid bin. Two mutation rates are examined: 0.01 and 0.05, enabling comparison between lower and higher exploration levels.
# The algorithm employs generational replacement, where the entire population is refreshed each iteration. To retain progress, elitism is used to carry the single best chromosome from the previous generation into the next without modification.
# For every configuration of mutation rate and tournament size, five independent trials are conducted using different random seeds. This ensures robust evaluation and avoids artefacts arising from specific initial conditions.

import numpy as np
import matplotlib.pyplot as plt
import os
import copy


class Bin_packing_problem:

    def __init__(self, 
                 num_bins: int,
                 mutation_rate: float,
                 tournament_size: int,
                 weight_function: callable,
                 fitness_function: callable,
                 crossover_rate: float=0.8,
                 num_items: int=500,
                 population_size: int=100,
                 max_evaluations: int=10000,
                 num_trials: int=5):
        self.num_bins = num_bins
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.weight_function = weight_function
        self.fitness_function = fitness_function
        self.crossover_rate = crossover_rate
        self.num_items = num_items
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.has_run = False
        self.num_trials = num_trials
        num_gens = max_evaluations // population_size
        self.fitness_history = np.zeros((num_trials, num_gens ), dtype=float)
        self.best_solutions = np.zeros((num_trials, num_items), dtype=int)
        self.best_bin_weights = np.zeros((num_trials, num_bins), dtype=float)
        self.rng = np.random.default_rng() # will be seeded later per trail
        self.population = np.zeros((population_size, num_items), dtype=int)


    def _initialize_population(self) -> np.ndarray:
        '''
        Initialize the population with random bin assignments.
        Each individual is represented as an array of integers, where each 
        integer indicates the bin assignment for the corresponding item.
        '''
        return self.rng.integers(1, self.num_bins + 1, size=(self.population_size, self.num_items))
    
    def _bin_weights(self, individual) -> np.ndarray:
        '''
        Calculate the weights of each bin for a given individual.
        '''
        bin_weights = np.zeros(self.num_bins)
        for item_index in range(self.num_items):
            bin_index = individual[item_index] - 1
            bin_weights[bin_index] += self.weight_function(item_index + 1)
        return bin_weights
    
    def _evaluate_fitness(self, individual) -> float:
        '''
        Evaluate the fitness of an individual based on the weight function
        and fitness function provided.
        '''
        bin_weights = self._bin_weights(individual)
        
        fitness = self.fitness_function(bin_weights)
        self.evaluations += 1
        return fitness
    
    def _evaluate_population(self) -> np.ndarray:
        '''
        Evaluate the fitness of the entire population.
        '''
        fitnesses = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitnesses[i] = self._evaluate_fitness(self.population[i, :])
        return fitnesses
    
    def _best_individual(self, fitnesses: np.ndarray) -> np.ndarray:
        '''
        Return the best individual in the population based on fitnesses.
        '''
        best_index = np.argmax(fitnesses)
        return self.population[best_index, :]
    
    def _tournament_selection(self, fitnesses: np.ndarray) -> np.ndarray:
        '''
        Select an individual using tournament selection.
        '''
        tournament_indices = self.rng.choice(self.population_size, size=self.tournament_size, replace=False)
        tournament_fitnesses = fitnesses[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return self.population[winner_index, :]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        '''
        Perform uniform crossover between two parents to produce an offspring.
        '''
        if self.rng.random() < self.crossover_rate:
            mask = self.rng.integers(0, 2, size=self.num_items).astype(bool)
            offspring1 = np.where(mask, parent1, parent2)
            offspring2 = np.where(mask, parent2, parent1)
            return offspring1, offspring2
        else:
            return parent1.copy(), parent2.copy()
        
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        '''
        Mutate an individual by randomly reassigning items to bins.
        '''
        for item_index in range(self.num_items):
            if self.rng.random() < self.mutation_rate:
                individual[item_index] = self.rng.integers(1, self.num_bins + 1)
        return individual
    
    def run(self):
        '''
        Run the genetic algorithm until the maximum number of evaluations is reached.
        '''
        for i in range(self.num_trials):
            # unqiue seed based on trail num and other params
            seed = (i + 1) * 1e8 + int(self.mutation_rate * 1e6) + self.tournament_size * 1e4 + self.num_bins * 1e2
            self.rng = np.random.default_rng(int(seed))
            self.population = self._initialize_population()
            self.evaluations = 0
            gen = 0
            while self.evaluations <= self.max_evaluations - self.population_size:
                fitnesses = self._evaluate_population()
                self.fitness_history[i, gen] = np.max(fitnesses)
                best_individual = self._best_individual(fitnesses)
                self.best_solutions[i, :] = best_individual

                if self.evaluations < self.max_evaluations:
                    new_population = np.zeros_like(self.population)
                    # Elitism: carry the best individual to the new population
                    new_population[0, :] = best_individual
                    for j in range(1, self.population_size, 2):
                        parent1 = self._tournament_selection(fitnesses)
                        parent2 = self._tournament_selection(fitnesses)

                        offspring1, offspring2 = self._crossover(parent1, parent2)

                        new_population[j, :] = self._mutate(offspring1)

                        if j + 1 < self.population_size:
                            new_population[j + 1, :] = self._mutate(offspring2)

                    self.population = new_population
                    gen += 1
            
            assert self.evaluations <= self.max_evaluations
            self.best_bin_weights[i, :] = self._bin_weights(self.best_solutions[i, :])
        
        self.has_run = True

    def get_history(self) -> np.ndarray:
        '''
        Get the fitness history of evaluations.
        '''
        if not self.has_run:
            raise RuntimeError("The algorithm must be run before getting history.")
        return self.fitness_history, self.best_solutions, self.best_bin_weights
    
class Gga_bin_packing_problem:
    
    def __init__(self,
                 num_bins: int,
                 mutation_rate: float,
                 tournament_size: int,
                 weight_function: callable,
                 fitness_function: callable,
                 crossover_rate: float = 0.8,
                 num_items: int = 500,
                 population_size: int = 100,
                 max_evaluations: int = 10000,
                 num_trials: int = 5):
        
        self.num_bins = num_bins
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.weight_function = weight_function
        self.fitness_function = fitness_function
        self.crossover_rate = crossover_rate
        self.num_items = num_items
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.num_trials = num_trials
        
        self.rng = np.random.default_rng()
        
        num_gens = max_evaluations // population_size
        self.fitness_history = np.zeros((num_trials, num_gens))
        self.best_solutions = np.zeros((num_trials, num_items), dtype=int)
        self.best_bin_weights = np.zeros((num_trials, num_bins), dtype=float)
        self.has_run = False

    def _random_group_solution(self):
        """Create random grouping (GGA representation)."""
        bins = [[] for _ in range(self.num_bins)]
        items = np.arange(1, self.num_items + 1)
        self.rng.shuffle(items)
        for item in items:
            b = self.rng.integers(0, self.num_bins)
            bins[b].append(item)
        return bins

    def _encode_solution(self, grouped):
        """Convert grouping to flat chromosome for logging."""
        sol = np.zeros(self.num_items, dtype=int)
        for b, group in enumerate(grouped):
            for item in group:
                sol[item - 1] = b + 1
        return sol

    def _bin_weights(self, grouped):
        weights = np.zeros(self.num_bins)
        for b, group in enumerate(grouped):
            for item in group:
                weights[b] += self.weight_function(item)
        return weights

    def _evaluate_fitness(self, grouped):
        bin_w = self._bin_weights(grouped)
        return self.fitness_function(bin_w)

    def _tournament_selection(self, population, fitnesses):
        idxs = self.rng.choice(self.population_size, size=self.tournament_size, replace=False)
        best_idx = idxs[np.argmax(fitnesses[idxs])]
        return copy.deepcopy(population[best_idx])
    
    def _crossover(self, parent1, parent2):
        if self.rng.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child = [[] for _ in range(self.num_bins)]
        used = set()

        # Step 1: randomly transfer groups from parent1
        for bin_group in parent1:
            if self.rng.random() < 0.5 and bin_group:
                new_group = [i for i in bin_group if i not in used]
                if new_group:
                    child.append(new_group)
                    used.update(new_group)

        # Step 2: fill missing items with parent2 ordering
        flat_p2 = [item for group in parent2 for item in group]
        remaining = [i for i in flat_p2 if i not in used]

        for item in remaining:
            self.rng.shuffle(child)
            for group in child:
                if len(group) < self.num_items:  # sanity constraint
                    group.append(item)
                    break

        # Ensure correct number of bins
        child = [g for g in child if len(g) > 0]
        while len(child) < self.num_bins:
            child.append([])

        return child[:self.num_bins], copy.deepcopy(child[:self.num_bins])
    
    def _mutate(self, grouped):
        for _ in range(int(self.mutation_rate * self.num_items)):
            src = self.rng.integers(0, self.num_bins)
            if grouped[src]:
                item = grouped[src].pop(self.rng.integers(0, len(grouped[src])))
                dest = self.rng.integers(0, self.num_bins)
                grouped[dest].append(item)
        return grouped
    
    def run(self):
        for trial in range(self.num_trials):
            seed = (trial + 1) * 1e8 + int(self.mutation_rate * 1e6) + self.tournament_size * 1e4 + self.num_bins * 1e2
            self.rng = np.random.default_rng(int(seed))

            population = [self._random_group_solution() for _ in range(self.population_size)]
            evaluations = 0
            gen = 0

            while evaluations < self.max_evaluations:

                fitnesses = np.array([self._evaluate_fitness(ind) for ind in population])
                evaluations += self.population_size

                best = population[np.argmax(fitnesses)]
                self.fitness_history[trial, gen] = np.max(fitnesses)
                self.best_solutions[trial, :] = self._encode_solution(best)
                gen += 1

                new_pop = [copy.deepcopy(best)]  # elitism

                while len(new_pop) < self.population_size:
                    p1 = self._tournament_selection(population, fitnesses)
                    p2 = self._tournament_selection(population, fitnesses)
                    c1, c2 = self._crossover(p1, p2)
                    new_pop.append(self._mutate(c1))
                    if len(new_pop) < self.population_size:
                        new_pop.append(self._mutate(c2))

                population = new_pop

            # record best bin loads
            self.best_bin_weights[trial, :] = self._bin_weights(best)

        self.has_run = True
    

    def get_history(self):
        if not self.has_run:
            raise RuntimeError("Must run algorithm first.")
        return self.fitness_history, self.best_solutions, self.best_bin_weights
    
def plot_history(BPP_instance: Bin_packing_problem, title: str, save_location: str=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    plt.rcParams.update({
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    for idx, instance in enumerate(BPP_instance):
        fitness_history, _, _ = instance.get_history()
        
        ax = axes[idx]
        
        for trial in range(fitness_history.shape[0]):
            ax.plot(fitness_history[trial, :])
        
        ax.set_title(f"pm={instance.mutation_rate}, t={instance.tournament_size}", fontsize=16)
        ax.set_xlabel("Generation", fontsize=14)
        ax.set_ylabel("Fitness", fontsize=14)

    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    file_name = f"{save_location}/fitness_history.png"
    fig.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    # Example fitness function
    def fitness_function(bin_weights):
        d = np.max(bin_weights) - np.min(bin_weights)
        return 100 / (1 + d)

    # Create an instance of the Bin Packing Problem for BPP1
    bpp1 = [
        Bin_packing_problem(
            num_bins=10,
            mutation_rate=0.01,
            tournament_size=3,
            weight_function=lambda i: i,
            fitness_function=fitness_function
        ),
        Bin_packing_problem(
            num_bins=10,
            mutation_rate=0.05,
            tournament_size=3,
            weight_function=lambda i: i,
            fitness_function=fitness_function
        ),
        Bin_packing_problem(
            num_bins=10,
            mutation_rate=0.01,
            tournament_size=7,
            weight_function=lambda i: i,
            fitness_function=fitness_function
        ),
        Bin_packing_problem(
            num_bins=10,
            mutation_rate=0.05,
            tournament_size=7,
            weight_function=lambda i: i,
            fitness_function=fitness_function
        )
    ]

    for instance in bpp1:
        # Run the genetic algorithm
        instance.run()

    # Print statistics for BPP1
    print("-" * 44 + " Results for BPP1 " + "-"*43)
    for instance in bpp1:
        fitness_history, best_solutions, best_bin_weights = instance.get_history()
        last_fitnesses = fitness_history[:, -1]
        fitnesses = np.max(last_fitnesses, axis=0)
        avg_fitness = np.mean(last_fitnesses)
        std_fitness = np.std(last_fitnesses)
        
        best_idx = np.argmax(last_fitnesses, axis=0)
        std_weight = np.std(best_bin_weights[best_idx], axis=0)
        best_solution = best_solutions[best_idx]
        bin_weights = np.zeros(instance.num_bins)
        for item_index in range(instance.num_items):
            bin_index = best_solution[item_index] - 1
            bin_weights[bin_index] += instance.weight_function(item_index + 1)
        d = np.max(bin_weights) - np.min(bin_weights)

        print(f"BPP1 - Bins: {instance.num_bins}, Mutation Rate: {instance.mutation_rate}, "
              f"Tournament Size: {instance.tournament_size} => Best: {np.max(fitnesses):.5f}, "
              f"Average: {avg_fitness:.5f}, Std: {std_fitness:.5f}\n" + " " * 40 + 
              f"For Best Fitness => Std Bin Weight: {std_weight:.5f}, Difference: {d}")
    print("-" * 105)

    bpp2 = [
        Bin_packing_problem(
            num_bins=50,
            mutation_rate=0.01,
            tournament_size=3,
            weight_function=lambda i: (i ** 2) / 2,
            fitness_function=fitness_function
        ),
        Bin_packing_problem(
            num_bins=50,
            mutation_rate=0.05,
            tournament_size=3,
            weight_function=lambda i: (i ** 2) / 2,
            fitness_function=fitness_function
        ),
        Bin_packing_problem(
            num_bins=50,
            mutation_rate=0.01,
            tournament_size=7,
            weight_function=lambda i: (i ** 2) / 2,
            fitness_function=fitness_function
        ),
        Bin_packing_problem(
            num_bins=50,
            mutation_rate=0.05,
            tournament_size=7,
            weight_function=lambda i: (i ** 2) / 2,
            fitness_function=fitness_function
        )
    ]

    for instance in bpp2:
        # Run the genetic algorithm
        instance.run()

    # Print statistics for BPP1
    print("-" * 44 + " Results for BPP2 " + "-"*43)
    for instance in bpp2:
        fitness_history, best_solutions, best_bin_weights = instance.get_history()
        last_fitnesses = fitness_history[:, -1]
        fitnesses = np.max(last_fitnesses, axis=0)
        avg_fitness = np.mean(last_fitnesses)
        std_fitness = np.std(last_fitnesses)
        
        best_idx = np.argmax(last_fitnesses, axis=0)
        std_weight = np.std(best_bin_weights[best_idx], axis=0)
        best_solution = best_solutions[best_idx]
        bin_weights = np.zeros(instance.num_bins)
        for item_index in range(instance.num_items):
            bin_index = best_solution[item_index] - 1
            bin_weights[bin_index] += instance.weight_function(item_index + 1)
        d = np.max(bin_weights) - np.min(bin_weights)

        print(f"BPP2 - Bins: {instance.num_bins}, Mutation Rate: {instance.mutation_rate}, "
              f"Tournament Size: {instance.tournament_size} => Best: {np.max(fitnesses):.5f}, "
              f"Average: {avg_fitness:.5f}, Std: {std_fitness:.5f}\n" + " " * 40 + 
              f"For Best Fitness => Std Bin Weight: {std_weight:.5f}, Difference: {d}")
    print("-" * 105)

    # gga = [
    #     Gga_bin_packing_problem(
    #         num_bins=10,
    #         mutation_rate=0.01,
    #         tournament_size=3,
    #         weight_function=lambda i: i,
    #         fitness_function=fitness_function,
    #         max_evaluations=20000
    #     ),
    #     Gga_bin_packing_problem(
    #         num_bins=10,
    #         mutation_rate=0.05,
    #         tournament_size=3,
    #         weight_function=lambda i: i,
    #         fitness_function=fitness_function,
    #         max_evaluations=20000
    #     ),
    #     Gga_bin_packing_problem(
    #         num_bins=10,
    #         mutation_rate=0.01,
    #         tournament_size=7,
    #         weight_function=lambda i: i,
    #         fitness_function=fitness_function,
    #         max_evaluations=20000
    #     ),
    #     Gga_bin_packing_problem(
    #         num_bins=10,
    #         mutation_rate=0.05,
    #         tournament_size=7,
    #         weight_function=lambda i: i,
    #         fitness_function=fitness_function,
    #         max_evaluations=20000
    #     )
    # ]
    # for instance in gga:
    #     # Run the genetic algorithm
    #     instance.run()

    # # Print statistics for BPP1
    # print("-" * 44 + " Results for GGA " + "-"*43)
    # for instance in gga:
    #     fitness_history, best_solutions, best_bin_weights = instance.get_history()
    #     last_fitnesses = fitness_history[:, -1]
    #     fitnesses = np.max(last_fitnesses, axis=0)
    #     avg_fitness = np.mean(last_fitnesses)
    #     std_fitness = np.std(last_fitnesses)
        
    #     best_idx = np.argmax(last_fitnesses, axis=0)
    #     std_weight = np.std(best_bin_weights[best_idx], axis=0)

    #     print(f"GGA - Bins: {instance.num_bins}, Mutation Rate: {instance.mutation_rate}, "
    #           f"Tournament Size: {instance.tournament_size} => Best: {np.max(fitnesses):.5f}, "
    #           f"Average: {avg_fitness:.5f}, Std: {std_fitness:.5f}\n" + " " * 39 + 
    #           f"For Best Fitness => Std Bin Weight: {std_weight:.5f}")
    # print("-" * 105)


    plot_history(bpp1, title="Fitness History for BPP1", save_location="results/bpp1")
    plot_history(bpp2, title="Fitness History for BPP2", save_location="results/bpp2")
    # plot_history(gga, title="Fitness History for GGA", save_location="results/gga")

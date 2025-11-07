import numpy as np
import matplotlib.pyplot as plt
import os


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
        Initialize the population with random bin assignments
        Each individual is represented as an array of integers, where each 
        integer indicates the bin assignment for the corresponding item
        '''
        return self.rng.integers(1, self.num_bins + 1, size=(self.population_size, self.num_items))
    
    def _bin_weights(self, individual) -> np.ndarray:
        '''
        Calculate the weights of each bin for a given individual
        '''
        bin_weights = np.zeros(self.num_bins)
        for item_index in range(self.num_items):
            bin_index = individual[item_index] - 1
            bin_weights[bin_index] += self.weight_function(item_index + 1)

        return bin_weights
    
    def _evaluate_fitness(self, individual) -> float:
        '''
        Evaluate the fitness of an individual based on the weight function
        and fitness function provided
        '''
        bin_weights = self._bin_weights(individual)
        
        fitness = self.fitness_function(bin_weights)
        self.evaluations += 1

        return fitness
    
    def _evaluate_population(self) -> np.ndarray:
        '''
        Evaluate the fitness of the entire population
        '''
        fitnesses = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitnesses[i] = self._evaluate_fitness(self.population[i, :])

        return fitnesses
    
    def _best_individual(self, fitnesses: np.ndarray) -> np.ndarray:
        '''
        Return the best individual in the population based on fitnesses
        '''
        best_index = np.argmax(fitnesses)

        return self.population[best_index, :]
    
    def _tournament_selection(self, fitnesses: np.ndarray) -> np.ndarray:
        '''
        Select an individual using tournament selection
        '''
        tournament_indices = self.rng.choice(self.population_size, size=self.tournament_size, replace=False)
        tournament_fitnesses = fitnesses[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]

        return self.population[winner_index, :]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        '''
        Perform uniform crossover between two parents to produce an offspring
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
        Mutate an individual by randomly reassigning items to bins
        '''
        for item_index in range(self.num_items):
            if self.rng.random() < self.mutation_rate:
                individual[item_index] = self.rng.integers(1, self.num_bins + 1)

        return individual
    
    def run(self):
        '''
        Run the genetic algorithm until the maximum number of evaluations is reached
        for the specified number of trials
        '''
        for i in range(self.num_trials):
            # unqiue seed based on trail num and other params
            seed = (i + 1) * 1e8 + int(self.mutation_rate * 1e6) + self.tournament_size * 1e4 + self.num_bins * 1e2
            self.rng = np.random.default_rng(int(seed))
            self.population = self._initialize_population()
            self.evaluations = 0
            gen = 0

            # Main GA loop
            while self.evaluations <= self.max_evaluations - self.population_size:
                # Evaluate population and record best fitness
                fitnesses = self._evaluate_population()
                self.fitness_history[i, gen] = np.max(fitnesses)
                best_individual = self._best_individual(fitnesses)
                self.best_solutions[i, :] = best_individual

                # Split into separate if to ensure fitness for last gen is recorded without running antother gen
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
        Get the fitness history, best solutions, and best bin weights after running the algorithm
        '''
        if not self.has_run:
            raise RuntimeError("The algorithm must be run before getting history")
        return self.fitness_history, self.best_solutions, self.best_bin_weights
    
def print_statistics(BPP_instance: Bin_packing_problem):
    """
    Print statistics for a given BPP_instance after running the GA
    """
    total_weight = sum([BPP_instance.weight_function(i) for i in range(1, BPP_instance.num_items + 1)])
    fitness_history, best_solutions, best_bin_weights = BPP_instance.get_history()
    last_fitnesses = fitness_history[:, -1] # (num_trials, gen)
    fitnesses = np.max(last_fitnesses, axis=0)
    avg_fitness = np.mean(last_fitnesses)
    std_fitness = np.std(last_fitnesses)
    
    best_idx = np.argmax(last_fitnesses, axis=0)
    std_weight = np.std(best_bin_weights[best_idx], axis=0)
    best_solution = best_solutions[best_idx]
    bin_weights = np.zeros(BPP_instance.num_bins)
    for item_index in range(BPP_instance.num_items):
        bin_index = best_solution[item_index] - 1
        bin_weights[bin_index] += BPP_instance.weight_function(item_index + 1)
    d = np.max(bin_weights) - np.min(bin_weights)

    fitness_lienar = 1 - (d / total_weight)
    fitness_dev = 100 * d / (2 * total_weight)

    print(f"Bins: {BPP_instance.num_bins}, Mutation Rate: {BPP_instance.mutation_rate}, "
          f"Tournament Size: {BPP_instance.tournament_size} => Best: {np.max(fitnesses):.5f}, "
          f"Average: {avg_fitness:.5f}, Std: {std_fitness:.5f}\n" + " " * 33 + 
          f"For Best Fitness => Std Bin Weight: {std_weight:.5f}, Difference: {d}\n" + " " * 50 +
          f"=> f_lin: {fitness_lienar:.4f}, f_dev: {fitness_dev:.4f}")
    
def plot_history(BPP_instance: Bin_packing_problem, title: str, save_location: str=""):
    """
    Plot fitness history for each trial in a 2x2 grid
    Each BPP_instance in the list represents a different GA parameter config,
    containing multiple trials internally
    """
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

def plot_average_history(bpp_instances: list[Bin_packing_problem], title: str, save_location: str = ""):
    """
    Plot mean +- range fitness history across multiple trials for each experiment setup
    Each BPP_instance in the list represents a different GA parameter config,
    containing multiple trials internally
    """
    plt.figure(figsize=(12, 7))
    plt.rcParams.update({
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    for instance in bpp_instances:
        fitness_history, _, _ = instance.get_history()

        mean_history = fitness_history.mean(axis=0)
        min_history = fitness_history.min(axis=0)
        max_history = fitness_history.max(axis=0)

        generations = range(fitness_history.shape[1])

        label = f"pm={instance.mutation_rate}, t={instance.tournament_size}"

        # mean curve
        plt.plot(generations, mean_history, linewidth=2, label=label)

        # shading
        plt.fill_between(
            generations,
            min_history,
            max_history,
            alpha=0.25
        )

    plt.title(title, fontsize=20)
    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Fitness", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if not os.path.exists(save_location):
        os.makedirs(save_location)
    file_name = f"{save_location}/fitness_history_avg.png"
    plt.savefig(file_name)

    plt.show()

def fitness_function(bin_weights):
    """
    Fitness function from project brief
    """
    d = np.max(bin_weights) - np.min(bin_weights)
    return 100 / (1 + d)

if __name__ == "__main__":

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
        print_statistics(instance)
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
        print_statistics(instance)
    print("-" * 105)

    extension = Bin_packing_problem(
            num_bins=10,
            mutation_rate=0.001,
            tournament_size=10,
            weight_function=lambda i: i,
            fitness_function=fitness_function
    )

    extension.run()

    print("-" * 41 + " Results for Extension " + "-"*40)
    print_statistics(extension)
    print("-" * 105)

    # Plotting
    plot_history(bpp1, title="Fitness History for BPP1", save_location="results/bpp1")
    plot_history(bpp2, title="Fitness History for BPP2", save_location="results/bpp2")
    plot_average_history(bpp1, title="Average Fitness History for BPP1", save_location="results/bpp1")
    plot_average_history(bpp2, title="Average Fitness History for BPP2", save_location="results/bpp2")
    plot_average_history([extension], title="Average Fitness History for Extension", save_location="results/extension")
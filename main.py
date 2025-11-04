import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable
import os


def fitness_diff_based(chromosome: np.ndarray, weights: np.ndarray, b: int) -> float:
    '''
    Compute the difference between the min and max bin sums, then compute the fitness

    This is the fitness function specified in the coursework.

    Parameters:
        chromosome: 1d-array of length k with values 1..b
        weights: 1d-array of length k
        b: number of bins
    Returns:
        fitness: fitness value of the chromosome
        d: difference between max and min bin sums
    '''
    # bins indexed 1..b -> shift to 0..b-1
    sums = np.bincount(chromosome - 1, weights=weights, minlength=b)
    d = float(sums.max() - sums.min())
    fitness = 100.0 / (1.0 + d)

    return fitness

def fitness_cv_based(chromosome: np.ndarray, weights: np.ndarray, b: int) -> float:
    sums = np.bincount(chromosome - 1, weights=weights, minlength=b)
    mean_load = float(np.mean(sums))
    std_dev = float(np.std(sums))
    cv = std_dev / mean_load if mean_load > 0 else 0.0
    fitness = 100.0 / (1.0 + 50.0 * cv)

    return fitness # scale factor (50) adjusts sensitivity

def fitness_mse_based(chromosome: np.ndarray, weights: np.ndarray, b: int) -> float:
    sums = np.bincount(chromosome - 1, weights=weights, minlength=b)
    mean_load = float(np.mean(sums))
    mse = float(np.mean((sums - mean_load) ** 2))
    fitness = 100.0 / (1.0 + mse)

    return fitness

def fitness_range_normalised(chromosome: np.ndarray, weights: np.ndarray, b: int) -> float:
    sums = np.bincount(chromosome - 1, weights=weights, minlength=b)
    mean_load = float(np.mean(sums))
    d = float(sums.max() - sums.min())
    rel_diff = d / mean_load if mean_load > 0 else 0.0
    fitness = 100.0 / (1.0 + rel_diff)

    return fitness

def calc_statistics(chromosome: np.ndarray, weights: np.ndarray, b: int) -> dict:
    '''
    Calculate statistics for a given chromosome.

    Parameters:
        chromosome: 1d-array of length k with values 1..b
        weights: 1d-array of length k
        b: number of bins
    Returns:
        stats: dictionary with keys:
            "min_load": minimum bin load (float)
            "max_load": maximum bin load (float)
            "std_dev": standard deviation of bin loads (float)
            "cv": coefficient of variation (float)
            "d": difference between max and min bin loads (float)
    '''
    sums = np.bincount(chromosome - 1, weights=weights, minlength=b)
    min_load = float(sums.min())
    max_load = float(sums.max())
    mean_load = float(np.mean(sums))
    std_dev = float(np.std(sums))
    cv = std_dev / mean_load if mean_load > 0 else 0.0
    d = float(max_load - min_load)
    return {
        "min_load": min_load,
        "max_load": max_load,
        "std_dev": std_dev,
        "cv": cv,
        "d": d
    }
    
def evaluate_population(population: np.ndarray, weights: np.ndarray, b: int, eval_counter: list[int], fitness_fn: Callable[[np.ndarray, np.ndarray, int], float]) -> tuple[np.ndarray, dict]:
    '''
    Evaluate the entire population and return fitnesses and d values.
    Also increment eval_counter[0] by the number of evaluations performed.

    Parameters:
        population: 2d-array shape (p, k) of chromosomes
        weights: 1d-array of length k
        b: number of bins
        eval_counter: list with single int element to track number of evaluations (pseudo-pass by reference)
    Returns:
        fitnesses: 1d-array of length p with fitness values
        ds: 1d-array of length p with d values from fitness calculation
    '''
    p = population.shape[0]
    fitnesses = np.empty(p, dtype=float)
    stats_list = []

    for i in range(p):
        fitnesses[i] = fitness_fn(population[i], weights, b)
        assert 0.0 <= fitnesses[i] <= 100.0, f'Invalid fitness {fitnesses[i]}'
        stats_list.append(calc_statistics(population[i], weights, b))
        eval_counter[0] += 1

    # Transpose list of dicts to dict of lists
    stats_dict = {key: [d[key] for d in stats_list] for key in stats_list[0]}

    return fitnesses, stats_dict

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray, pc: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    '''
    1. For each gene position, flip a coin
    2. If heads, take gene from parent 1; if tails, take gene from parent 2
    3. Apply crossover with probability pc (crossover rate)
    
    Parameters:
        parent1,parent2: chromosomes to crossover
        pc: float, crossover probability (0 <= pc <= 1)
        rng: numpy random generator instance
    Returns:
        child1,child2: offspring chromosomes
    '''
    num_genes = parent1.shape[0]

    if rng.random() < pc:
        mask = rng.random(size=num_genes) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()

    return child1, child2

def mutate(chromosome: np.ndarray, pm: float, b: int, rng: np.random.Generator) -> np.ndarray:
    '''
    1. For each gene, with probability pm (mutation rate)
    2. Randomly change the bin assignment to any valid bin (1 to b)

    Parameters:
        chromosome: 1d-array of gene values (1..b)
        pm: mutation probability (0 <= pm <= 1)
        b: number of bins
        rng: numpy random generator instance
    Returns:
        mutated_chromosome: mutated copy of input chromosome
    '''
    num_genes = chromosome.shape[0]
    mutated_chromosome = chromosome.copy()
    # for each gene, with probability pm, reassign to random bin 1..b
    mutation_mask = rng.random(size=num_genes) < pm

    if mutation_mask.any():
        # random new bins
        new_bins = rng.integers(1, b+1, size=mutation_mask.sum())
        mutated_chromosome[mutation_mask] = new_bins

    return mutated_chromosome

def tournament_select(fitnesses: np.ndarray, tournament_size: int, rng: np.random.Generator) -> int:
    '''
    1. Randomly select t chromosomes from the population (tournament size)
    2. Choose the chromosome with the best fitness from this tournament

    Parameters:
        fitnesses: 1d-array of fitness values for the population
        tournament_size: int, number of contenders in each tournament (t >= 2)
        rng: numpy random generator instance
    Returns:
        winner_idx: index of chosen parent in the population
    '''
    p = fitnesses.shape[0]

    assert tournament_size < p, 'Tournament size must less than population size'
    
    contenders = rng.choice(p, size=tournament_size, replace=False)
    
    winner_idx = contenders[np.argmax(fitnesses[contenders])]
    return winner_idx

def initialise_balanced_population(p: int, k: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Create p chromosomes, each with roughly equal representation of bins 1..b."""
    base = np.repeat(np.arange(1, b + 1), k // b)
    remainder = k % b
    population = np.empty((p, k), dtype=int)

    for i in range(p):
        chrom = base.copy()
        # Distribute the remainder randomly
        if remainder > 0:
            extras = rng.choice(np.arange(1, b + 1), size=remainder, replace=False)
            chrom = np.concatenate([chrom, extras])
        rng.shuffle(chrom)
        population[i] = chrom

    return population

def run_ga(weights: np.ndarray, b: int, p: int, pm: float, tournament_size: int, pc: float, max_evaluations: int, seed: int, fitness_fn: Callable[[np.ndarray, np.ndarray, int], float]) -> dict:
    '''
    Run the genetic algorithm to solve the bin packing problem as follows:
    1. Initialize a population of p randomly generated chromosomes.
    2. Evaluate the fitness of each chromosome in the population.
    3. Select parents for reproduction using selection method.
    4. Create offspring through crossover and mutation operations.
    5. Replace the old population with the new population (or subset).
    6. If a termination criterion has been reached, then stop. Otherwise return to step 2.

    Parameters:
        weights: 1d-array of item weights
        b: number of bins
        p: population size
        pm: mutation probability (0 <= pm <= 1)
        tournament_size: number of contenders in tournament selection (t >= 2)
        pc: crossover probability (0 <= pc <= 1)
        max_evaluations: maximum number of fitness evaluations to perform
        seed: random seed for reproducibility
    Returns:
        results: dictionary with keys:
            "best_fitness": best fitness found  (float)
            "best_d": best d value found (float)
            "evaluations": total number of fitness evaluations performed (int)
            "generations": total number of generations completed (int)
            "best_chromosome": 1d-array of best chromosome found
            "best_fitness_history": 1d-array of length max_evaluations with best fitness found at each evaluation step
            "time_sec": total elapsed time in seconds (float)
    '''
    rng = np.random.default_rng(seed)
    k = weights.shape[0]
    population = rng.integers(1, b+1, size=(p, k))
    # population = initialise_balanced_population(p, k, b, rng)
    eval_counter = [0]  # pseudo-pass by reference
    best_fitness_history = np.zeros(max_evaluations, dtype=float)
    start_time = time.time()

    # initial evaluation
    fitnesses, stats = evaluate_population(population, weights, b, [0], fitness_fn)  # don't count initial evals
    best_idx = np.argmax(fitnesses)
    best_fitness = float(fitnesses[best_idx])
    best_chromosome = population[best_idx].copy()
    best_d = float(stats["d"][best_idx])

    # record initial population fitness evaluations
    for i in range(len(fitnesses)):
        best_fitness = max(best_fitness, fitnesses[i])
        best_fitness_history[i] = best_fitness

    generation = 0
    while eval_counter[0] < max_evaluations:
        generation += 1
        new_population = np.empty_like(population)

        # carry over best chromosome (elitism)
        new_population[0] = best_chromosome.copy()

        # fill the rest of new_population
        i = 1
        while i < p:
            # select parents
            idx1 = tournament_select(fitnesses, tournament_size, rng)
            idx2 = tournament_select(fitnesses, tournament_size, rng)
            parent1, parent2 = population[idx1], population[idx2]
            # crossover
            child1, child2 = uniform_crossover(parent1, parent2, pc, rng)
            # mutation
            child1 = mutate(child1, pm, b, rng)
            child2 = mutate(child2, pm, b, rng)
            # add to new population
            new_population[i] = child1
            i += 1
            if i < p:  # case that population is already full
                new_population[i] = child2
                i += 1

        # evaluate new population
        fitnesses_new, stats_new = evaluate_population(new_population, weights, b, eval_counter, fitness_fn)
        # update best_fitness and history
        for j in range(len(fitnesses_new)):
            best_fitness = max(best_fitness, fitnesses_new[j])
            best_fitness_history[eval_counter[0] - len(fitnesses_new) + j] = best_fitness

            # break in the case that we have reached max evaluations
            if eval_counter[0] - len(fitnesses_new) + j + 1 >= max_evaluations:
                break

        # replace population for next generation
        population = new_population
        fitnesses, stats = fitnesses_new, stats_new
        best_idx = np.argmax(fitnesses)
        best_chromosome = population[best_idx].copy()
        best_d = float(stats["d"][best_idx])

    elapsed = time.time() - start_time
    return {
        "best_fitness": best_fitness,
        "best_d": best_d,
        "min_load": stats["min_load"][best_idx],
        "max_load": stats["max_load"][best_idx],
        "std_dev": stats["std_dev"][best_idx],
        "cv": stats["cv"][best_idx],
        "evaluations": eval_counter[0],
        "generations": generation,
        "best_chromosome": best_chromosome,
        "best_fitness_history": best_fitness_history,
        "time_sec": elapsed
    }

def run_experiments(operation_settings: dict, trail_settings: list[dict], problems: list[dict]) -> pd.DataFrame:
    '''
    Run experiments for all combinations of problems and trail settings.
    
    Parameters:
        operation_settings: dictionary with keys:
            "num_trials": int, number of trials per configuration
            "max_evals": int, maximum number of fitness evaluations per run
            "pc": float, crossover probability (0 <= pc <= 1)
            "base_seed": int, base random seed for reproducibility
        trail_settings: list of dictionaries, each with keys:
            "p": int, population size
            "pm": float, mutation probability (0 <= pm <= 1)
            "tournament_size": int, number of contenders in tournament selection (t >= 2)
        problems: list of dictionaries, each with keys:
            "name": str, name of the problem instance
            "b": int, number of bins
            "weights": 1d-array of item weights
    Returns:
        df: pandas DataFrame with results of all runs
    '''
    results = []
    for prob_idx, prob in enumerate(problems):
        for setting_idx, setting in enumerate(trail_settings):
            for trial in range(operation_settings["num_trials"]):
                seed = operation_settings["base_seed"] + trial + int(setting["pm"]*1000) + setting["tournament_size"]*100
                res = run_ga(weights=prob["weights"],
                             b=prob["b"],
                             p=setting["p"],
                             pm=setting["pm"],
                             tournament_size=setting["tournament_size"],
                             pc=operation_settings["pc"],
                             max_evaluations=operation_settings["max_evals"],
                             seed=seed,
                             fitness_fn=operation_settings["fitness_fn"])
                results.append({
                    "problem": prob["name"],
                    "config_ident": (prob["name"], setting_idx),
                    "b": prob["b"],
                    "setting_p": setting["p"],
                    "setting_pm": setting["pm"],
                    "setting_tournament": setting["tournament_size"],
                    "trial": trial+1,
                    "best_fitness": res["best_fitness"],
                    "best_d": res["best_d"],
                    "min_load": res["min_load"],
                    "max_load": res["max_load"],
                    "std_dev": res["std_dev"],
                    "cv": res["cv"],
                    # "evaluations": res["evaluations"],
                    # "generations": res["generations"],
                    "time_sec": res["time_sec"],
                    "fitness_history": res["best_fitness_history"]
                })
                print(f'Completed {prob["name"]} p={setting["p"]} pm={setting["pm"]} t={setting["tournament_size"]} trial={trial+1} ' \
                      f'-> best_fitness={res["best_fitness"]:.6f} best_d={res["best_d"]:.6f} gens={res["generations"]}')

    df = pd.DataFrame(results)
    return df

def plot_histories(df_results: pd.DataFrame, operation_settings: dict, trail_settings: list[dict], problems: list[dict]) -> None:
    '''
    Plot fitness histories for each problem and trail setting in a 2x2 grid of subplots.
    
    Parameters:
        df_results: results from run_experiments
        operation_settings: dictionary with key "max_evals" for x-axis limit
        trail_settings: list of dictionaries with keys "pm" and "tournament_size" for subplot titles
        problems: list of dictionaries with key "name" for filtering results
    Returns:
        None
    '''
    max_evals = operation_settings["max_evals"]

    for prob in problems:
        prob_results = df_results[df_results["problem"] == prob["name"]]
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharey=True)
        axes = axes.flatten()  # flatten to 1d-array for easy iteration

        for ax, setting in zip(axes, trail_settings):
            setting_results = prob_results[
                (prob_results["setting_pm"] == setting["pm"]) &
                (prob_results["setting_tournament"] == setting["tournament_size"])
            ]
            for _, r in setting_results.iterrows():
                ax.plot(range(max_evals), r["fitness_history"], alpha=0.7, label=f"trial {r['trial']}")

            ax.set_title(f"p={setting['p']}, pm={setting['pm']}, t={setting['tournament_size']}")
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Best fitness")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"results/d_history_{prob['name']}.png")
        plt.show()

def plot_average_histories_single_plot(df_results: pd.DataFrame, operation_settings: dict, trail_settings: list[dict], problems: list[dict]) -> None:
    '''
    Plot average fitness histories with std deviation shading for each problem in a single plot.

    Parameters:
        df_results: results from run_experiments
        operation_settings: dictionary with key "max_evals" for x-axis limit
        trail_settings: list of dictionaries with keys "pm" and "tournament_size" for legend
        problems: list of dictionaries with key "name" for filtering results
    Returns:
        None
    '''
    max_evals = operation_settings["max_evals"]

    for prob in problems:
        prob_results = df_results[df_results["problem"] == prob["name"]]

        plt.figure(figsize=(12, 6))
        plt.title(f"Average Convergence for {prob['name']}", fontsize=16)

        for setting in trail_settings:
            # Filter results for this setting
            setting_results = prob_results[
                (prob_results["setting_pm"] == setting["pm"]) &
                (prob_results["setting_tournament"] == setting["tournament_size"])
            ]

            # Stack all fitness_history arrays: shape = (num_trials, max_evals)
            fitness_histories = np.stack(setting_results["fitness_history"].values)
            mean_history = fitness_histories.mean(axis=0)
            std_history = fitness_histories.std(axis=0)

            label = f"p={setting['p']}, pm={setting['pm']}, t={setting['tournament_size']}"
            plt.plot(range(max_evals), mean_history, lw=2, label=label)
            plt.fill_between(range(max_evals),
                             mean_history - std_history,
                             mean_history + std_history,
                             alpha=0.2)

        plt.xlabel("Evaluations")
        plt.ylabel("Best fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/d_history_avg_{prob['name']}.png")
        plt.show()

if __name__ == "__main__":
    trail_settings = [
        {"p": 100, "pm": 0.01, "tournament_size": 3},
        {"p": 100, "pm": 0.05, "tournament_size": 3},
        {"p": 100, "pm": 0.01, "tournament_size": 7},
        {"p": 100, "pm": 0.05, "tournament_size": 7}
    ]
    problems = [
        {"name": "BPP1", "b": 10, "weights": np.arange(1,501, dtype=float)},            # weight i = i
        {"name": "BPP2", "b": 50, "weights": (np.arange(1,501, dtype=float)**2) / 2.0}  # weight i = i^2 / 2
    ]
    operation_settings = {
        "num_trials": 5,
        "max_evals": 10000,
        "pc": 0.8,
        "base_seed": 12345,
        "fitness_fn": fitness_diff_based
    }

    df_results = run_experiments(operation_settings, trail_settings, problems)

    df_results_grouped = df_results.groupby(["config_ident"])
    for config_ident, group_df in df_results_grouped:
        problem_name, setting_idx = config_ident[0]
        assert len(group_df) == operation_settings["num_trials"], \
            f'Group (problem={problem_name}, setting_i={setting_idx}) has {len(group_df)} rows instead of {operation_settings["num_trials"]}'

        print(f"\nGroup: problem={problem_name}, setting_i={setting_idx}")
        print(group_df.drop(columns=["fitness_history"], inplace=False))

    avg_df = df_results_grouped.mean(numeric_only=True).reset_index()
    avg_df["problem"] = avg_df["config_ident"].apply(lambda x: x[0])
    # move "problem" to the front for nicer printing
    cols = ["problem"] + [c for c in avg_df.columns if c != "problem"]
    avg_df = avg_df[cols]

    print("\nAverage")
    print(avg_df)

    if not os.path.exists("results"):
        os.mkdir("results")

    plot_histories(df_results, operation_settings, trail_settings, problems)
    plot_average_histories_single_plot(df_results, operation_settings, trail_settings, problems)
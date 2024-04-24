import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Not from asignment imports, but needed for evolutionary algorithm
import random



# Calculate intercept and slope using scipy
def scipy(dataset):

    x = dataset['x']
    y = dataset['y']
    slope, intercept, _, _, _ = stats.linregress(x, y)

    return intercept, slope



# Calculate mean squared error (MSE) for intercept and slope values
def get_error(dataset, intercept, slope):

    y_pred = intercept + slope * dataset['x']
    y_actual = dataset['y']
    error = ((y_pred - y_actual) ** 2).mean( )

    return error



# Make 2D plot
def plot_2D(dataset):

    plt.scatter(dataset["x"], dataset["y"])
    plt.xlabel('X-values')
    plt.ylabel('Y-values')

    plt.show( )

    return



# Make 3D plot
def plot_3D(dataset):

    # Range of intercept and slope values (minimum, maximum, number of steps)
    intercepts = np.linspace(100, 200, 100)
    slopes = np.linspace(10, 50, 100)

    # Make meshgrid of intercepts and slopes
    intercepts, slopes = np.meshgrid(intercepts, slopes)

    # Error values for combinations of intercepts and slopes
    errors = np.zeros_like(intercepts)

    # Loop trough all combinations of intercept and slope values, and calculate corresponding errors
    for i in range(len(intercepts)):
        for j in range(len(slopes)):
            errors[i,j] = get_error(dataset, intercepts[i,j], slopes[i,j])

    # Make a 3D-plot, x-axis = intercept, y-axis = slope, z-axis = error
    fig = plt.figure( )
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(intercepts, slopes, errors)
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    ax.set_zlabel('Error')

    plt.show( )

    return



# Run evolutionary algorithm to calculate best intercept and slope values
def run_evo_alg(dataset, pop_size, gens, epsilon):

    # Start timer
    tic = time.perf_counter()

    # Parameters for the randmization of first population
    min_i = 0
    max_i = 500
    min_s = 0
    max_s = 150

    # Number of elite children and mutation children
    numb_elite = pop_size * 0.15
    numb_elite = round(numb_elite)

    # Initial population
    population = [(random.uniform(min_i, max_i), random.uniform(min_s, max_s)) for _ in range(pop_size)]

    if not investigate:
        print(f"Intercept range: {min_i} - {max_i}")
        print(f"Slope range:     {min_s} - {max_s}\n")

    global reached_gen
    global time_spend
    global error_list
    error_list = []
    
    # Loop trough generations
    for cur_gen in range(gens):

        # Calculate fitness, lower error score is better
        fitness = [get_error(dataset, *individual) for individual in population]
        best_error = min(fitness)

        # Get the best intercept and slope values
        best_intercept, best_slope = population[fitness.index(best_error)]

        error_list.append(best_error)

        # Stop loop if threshold is passed, return best values
        if best_error <= epsilon:
            params = [best_intercept, best_slope]

            toc = time.perf_counter()

            if not investigate:
                print(f"Evolution completed in {toc - tic:0.1f} seconds.")

            reached_gen = cur_gen
            time_spend = toc - tic

            return params
        
        # Sort population based on fitness scores
        sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])

        # Pick 15 % of population with best scores
        elite_children = [individual[0] for individual in sorted_population[:numb_elite]]

        # Make a population excisting only of elite children
        elite_population = [elite_children[i % len(elite_children)] for i in range(pop_size)]

        new_population = []

        # Mutate crossover population
        for individual in elite_population:
            mutated_individual = list(individual)

            for i in range(len(mutated_individual)):
                mutated_individual[i] = mutated_individual[i] + np.random.normal(0, 10)

            mutated_individual = tuple(mutated_individual)

            new_population.append(mutated_individual)

        # Make sure that the elite children of the prev pop are not mutated
        new_population = elite_children + new_population[len(elite_children):]

        # Update population
        population = new_population

        if not investigate:
            print(f"Generation {cur_gen} of size {pop_size}; best error {best_error:.0f}; \
            intercept {best_intercept:.2f}; slope {best_slope:.2f}")

    
    # If threshold is not passed, check population for best values

    # Calculate fitness of last population, lower error score is better
    fitness = [get_error(dataset, *individual) for individual in population]
    best_error = min(fitness)

    # Get the best intercept and slope values of last population
    intercept, slope = population[fitness.index(best_error)]
    params = [intercept, slope]

    # Stop timer and print time
    toc = time.perf_counter()

    if not investigate:
        print(f"Evolution completed in {toc - tic:0.1f} seconds.")
    
    reached_gen = gens
    time_spend = toc - tic

    return params



# Investigate pop size and numb of gens
def investigate(dataset):
    
    pop_size = []
    run_time = []
    gen_reached = []

    epsilon = 16.2
    gens = 100

    investigate = True

    print(f'Start running tests...')

    # First test
    for i in range(1, 21):
        i = i * 10
        run_evo_alg(dataset, i, gens, epsilon)
        pop_size.append(i)
        run_time.append(time_spend)
        gen_reached.append(reached_gen)


    _, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].plot(pop_size, run_time, color='blue')
    axs[0].set_xlabel('Population size')
    axs[0].set_ylabel('Run time')
    axs[0].set_title(f'Plot 1 (gens = {gens}, epsilon = {epsilon})')

    axs[1].plot(pop_size, gen_reached, color='red')
    axs[1].set_xlabel('Population size')
    axs[1].set_ylabel('Number of generations')
    axs[1].set_title(f'Plot 2 (gens = {gens}, epsilon = {epsilon})')

    plt.tight_layout()
    plt.savefig('Figure1.pdf')
    plt.show()

    # Second test
    for j in range(1, 4):
        pop = j * 100
        run_evo_alg(dataset, pop, gens, epsilon)
        plt.plot(range(0, len(error_list)), error_list, label = f'Pop_size: {pop}')

    plt.xlabel('Number of generations')
    plt.ylabel('Error value')
    plt.title('Line plot with errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure2.pdf')
    plt.show()

    print(f'Done testing!')

    return



# Main function to run all other functions
def main():

    # Set seed for reproducibility
    random.seed(1)

    data = pd.read_csv("datasets/dataset_group_2.csv")

    global investigate

    investigate(data)

    investigate = False

    # Plot data in 2D-plot, x-axis = x-values, y-axis = y-values
    plot_2D(data)

    # Plot data in 3D-plot, x-axis = intercept, y-axis = slope, z-axis = error
    plot_3D(data)

    # Calculate intercept, slope and error using scipy library
    intercept, slope = scipy(data)
    error = get_error(data, intercept, slope)
    print(f"Values using scipy library:")
    print(f"Intercept = {intercept}\nSlope = {slope}\nError = {error}\n")

    # Calculate intercept, slope and error using evolutionary algorithm
    evo_algo = run_evo_alg(data, 200, 100, 10)
    error = get_error(data, evo_algo[0], evo_algo[1])
    print(f"\nValues using evolutionary algorithm:")
    print(f"Intercept = {evo_algo[0]}\nSlope = {evo_algo[1]}\nError = {error}")
    


# Run main function
if __name__ == "__main__":
    main( )


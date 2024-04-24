import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LinearRegression

def scikit(dataset):
    X = dataset[['x']]
    y = dataset['y']

    model = LinearRegression( )
    model.fit(X, y)

    intercept = model.intercept_
    slope = model.coef_[0]

    return intercept, slope





def get_error(dataset, intercept, slope):

    y_pred = intercept + slope * dataset['x']
    y_actual = dataset['y']
    error = ((y_pred - y_actual) ** 2).mean( )

    return error





def plot_3D(dataset):

    intercepts = np.linspace(150, 200, 100) # (minimum, maximum, number of steps)
    slopes = np.linspace(20, 40, 100) # (minimum, maximum, number of steps)
    intercepts, slopes = np.meshgrid(intercepts, slopes)

    errors = np.zeros_like(intercepts)

    for i in range(len(intercepts)):
        for j in range(len(slopes)):
            errors[i,j] = get_error(dataset, intercepts[i,j], slopes[i,j])

    fig = plt.figure( )
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(intercepts, slopes, errors)
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    ax.set_zlabel('Error')

    plt.show( )

    return





def run_evo_alg(dataset, pop_size, gens, epsilon):
    tic = time.perf_counter()
    pop1_slope = np.linspace(0,50,1)
    pop2_intercept = np.linspace(0,400,1)
    worst_slope = 0
    worst_intercept = 0
    i = 0
    best_pop1 = 0
    best_pop2 = 0
    worsterror = 10000  
    for cur_gen in gens:
       for i in range(len(pop2_intercept)):
           for j in range(len(pop1_slope)):
               error =  get_error(dataset, pop1_slope[j], pop2_intercept[i])
               if (error < worsterror):
                   worsterror = error
                   worst_slope = j
                   worst_intercept= i
        pop1_slope = pop1_slope[-worst_slope,:]
       pop2_intercept[-worst_intercept,:]
       best_error = worsterror
       best_intercept = pop2_intercept
       best_slope = pop1_slope
       if ()
       print(f"Generation {cur_gen} of size {pop_size}; best error {best_error:.0f}; \
    intercept {best_intercept:.2f}; slope {best_slope:.2f}")
    return pop1_slope, pop2_intercept
       
       
        

        
    toc = time.perf_counter()
    print(f"Evolution completed in {toc - tic:0.1f} seconds.")
    params = [intercept, slope]
    return params





def main( ):
    data = pd.read_csv("dataset/dataset_group_5.csv")
    # plt.scatter(data["x"], data["y"])
    # plt.show()

    plot_3D(data)

    intercept, slope = scikit(data)
    print("Intercept:", intercept)
    print("Slope:", slope)

    error = get_error(data, intercept, slope)
    print("Error:", error)

    run_evo_alg(data, 10, 10, 10)
    print(f"Generation {cur_gen} of size {pop_size}; best error {best_error:.0f}; \
    intercept {best_intercept:.2f}; slope {best_slope:.2f}")





if __name__ == "__main__":
    main( )


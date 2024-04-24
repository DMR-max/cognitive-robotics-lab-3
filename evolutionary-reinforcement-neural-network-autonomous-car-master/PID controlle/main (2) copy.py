import math
import pygame
import pid
from utils import scale_image, blit_rotate_center, blit_text_center
# import time #NOT SURE OF MAG
# Not from asignment imports, but needed for evolutionary algorithm
import random #NOT SURE OF MAG
import numpy as np
import matplotlib.pyplot as plt


pygame.init()
pygame.font.init()

clock = pygame.time.Clock()
debug = False

Twiddle = False

wind = False
steer_bias = True

FrameHeight = 400
FrameWidth = 1200

pygame.display.set_caption("PID controller simulation")
screen = pygame.display.set_mode((FrameWidth, FrameHeight))

bg = pygame.image.load("background_small.png").convert()
RED_CAR = scale_image(pygame.image.load("imgs/red-car_small.png"), 1.0)

MAIN_FONT = pygame.font.SysFont("courier", 35)


def draw(win, player_car, scroll):

    i = 0
    while(i < tiles):
        screen.blit(bg, (bg.get_width()*i + scroll, 0))
        i += 1

    # RESET THE SCROLL FRAME
    if abs(scroll) > bg.get_width():
        scroll = 0

    if debug:
        level_text = MAIN_FONT.render(
            f"CTE {player_car.y - 266}", 1, (255, 255, 255))
        win.blit(level_text, (10, FrameHeight - level_text.get_height() - 70))

        steer_text = MAIN_FONT.render(
            f"Steering angle: {player_car.steering_angle}", 1, (255, 255, 255))
        win.blit(steer_text, (10, FrameHeight - steer_text.get_height() - 40))

        vel_text = MAIN_FONT.render(
            f"Vel: {round(player_car.vel, 1)} px/s", 1, (255, 255, 255))
        win.blit(vel_text, (10, FrameHeight - vel_text.get_height() - 10))
        print(player_car.x)


    player_car.draw(win)
    pygame.display.update()

    return scroll


def move_player(player_car):

    keys = pygame.key.get_pressed()
    moved = False

    current_CTE = player_car.y - 266
    # print(f"CTE = {current_CTE}")

    player_car.steering_angle = controller.process(current_CTE)

    if steer_bias:
        player_car.steering_angle += .3

    player_car.rotate()

    if debug:
        if keys[pygame.K_w]:
            moved = True
            player_car.move_forward()
        if keys[pygame.K_s]:
            moved = True
            player_car.move_backward()
        if not moved:
            player_car.reduce_speed()

    if not debug:
        player_car.move_forward()



class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.max_steering_angle = 4.0
        self.steering_angle = 0.0
        self.angle = 220
        self.x, self.y = self.START_POS
        self.prev_x, self.prev_y = self.START_POS
        self.acceleration = 0.1

    def rotate(self):
        if self.steering_angle > self.max_steering_angle:
            self.steering_angle = self.max_steering_angle
        if self.steering_angle < -self.max_steering_angle:
            self.steering_angle = -self.max_steering_angle

        # test for velocity-related steering speed, uncomment for original
        self.angle -= (self.vel / self.max_vel) * self.steering_angle
        

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.prev_x = self.x
        self.prev_y = self.y
        self.y -= vertical
        self.x -= horizontal

        if wind:
            self.y -= .2

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0




class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (45, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()




player_car = PlayerCar(1, 4)
controller = pid.PIDcontroller()
scroll = 0
tiles = math.ceil(FrameWidth / bg.get_width()) + 1
coords = np.array([[player_car.x, player_car.y]])



# special run function for Twiddle without drawing the simulation and higher clock ticks
def run_with_params(p, scroll):
        player_car = PlayerCar(1, 4)
        controller.p_p = p[0]
        controller.p_i = p[1]
        controller.p_d = p[2]
        counter = 0
        
        while 1:
            clock.tick(1000000)

            if counter > 2000:
                return 10000

            move_player(player_car)

            if player_car.x > 1200 or player_car.x < 0 or player_car.y < 0 or player_car.y > 400:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            counter += 1

        return controller.CTE_total



# if twiddle is off, or the end of twiddle is reached this function is used, draws simulation saves coordinates.
def run(scroll):

    while 1:
        clock.tick(200)

        scroll = draw(screen, player_car, scroll)

        move_player(player_car)

        global coords
        coords = np.append(coords, [[player_car.x, player_car.y]], axis=0)

        if player_car.x > 1200 or player_car.x < 0 or player_car.y < 0 or player_car.y > 400:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()




if Twiddle == False:
    run(scroll)
else:
    tol = 0.000002
    p = [controller.p_p, controller.p_i, controller.p_d]
    dp = [1, 1, 1]
    it = 0

    best_err = run_with_params(p, scroll)

    while sum(dp) > tol:

        if it % 50 == 0: 
            print("Iteration {}, best error = {}".format(it, best_err))

        for i in range(len(p)):
           
            p[i] += dp[i]
            controller.CTE_total = 0
            err = run_with_params(p, scroll)

            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] -= 2 * dp[i]
                controller.CTE_total = 0
                err = run_with_params(p, scroll)

                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] += dp[i]
                    dp[i] *= 0.9

        it += 1

    controller.p_p = p[0]
    controller.p_i = p[1]
    controller.p_d = p[2]

    print("\nTotal iterations = {}, best error = {}".format(it, best_err))
    print("\nP: " + str(controller.p_p) + "\nI: " + str(controller.p_i) + "\nD: " + str(controller.p_d))

    controller.CTE_total = 0
    player_car = PlayerCar(1, 4)

    run(scroll)
# Make 3D plot
def plot_3D(p_var, i_var, d_var):

    # Range of PID values (minimum, maximum, number of steps)
    P = p_var
    I = i_var 
    D = d_var

    if isinstance(p_var, np.ndarray) and isinstance(i_var, np.ndarray):
        # Make meshgrid of PI
        P, I = np.meshgrid(P, I)

        # Error values for combinations of PID
        errors = np.zeros_like(P)

        # Loop trough all combinations of P and I values, and calculate corresponding errors
        for i in range(len(P)):
            for j in range(len(I)):
                p = [P[i, j], I[i, j], D]
                controller.CTE_total = 0
                errors[i,j] = run_with_params(p, scroll)

        # Make a 3D-plot, x-axis = P, y-axis = I, z-axis = error
        fig = plt.figure( )
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(P, I, errors)
        ax.set_xlabel('P')
        ax.set_ylabel('I')
        ax.set_zlabel('CTE')

    if isinstance(i_var, np.ndarray) and isinstance(d_var, np.ndarray):
        # Make meshgrid of ID
        I, D = np.meshgrid(I, D)

        # Error values for combinations of PID
        errors = np.zeros_like(I)

        # Loop trough all combinations of I and D values, and calculate corresponding errors
        for i in range(len(I)):
            for j in range(len(D)):
                p = [P, I[i, j], D[i, j]]
                controller.CTE_total = 0
                errors[i,j] = run_with_params(p, scroll)

        # Make a 3D-plot, x-axis = I, y-axis = D, z-axis = error
        fig = plt.figure( )
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(I, D, errors)
        ax.set_xlabel('I')
        ax.set_ylabel('D')
        ax.set_zlabel('CTE')

    if isinstance(p_var, np.ndarray) and isinstance(d_var, np.ndarray):
        # Make meshgrid of PD
        P, D = np.meshgrid(P, D)

        # Error values for combinations of PID
        errors = np.zeros_like(P)

        # Loop trough all combinations of P and D values, and calculate corresponding errors
        for i in range(len(P)):
            for j in range(len(D)):
                p = [P[i, j], I, D[i, j]]
                controller.CTE_total = 0
                errors[i,j] = run_with_params(p, scroll)

        # Make a 3D-plot, x-axis = P, y-axis = D, z-axis = error
        fig = plt.figure( )
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(P, D, errors)
        ax.set_xlabel('P')
        ax.set_ylabel('D')
        ax.set_zlabel('CTE')

    plt.show( )

    return

# plot_3D(np.linspace(-20, 20), np.linspace(-0.1, 0.1), 3.5) #ga naar - 
# plot_3D(np.linspace(-20, 20), 0.00009, np.linspace(-5, 5))
# plot_3D(0.1, np.linspace(-0.1, 0.1), np.linspace(-5, 5))
#IMPORT TIME KAN NIET OMDAT DAT DE SIMULATION KILLED
# Run evolutionary algorithm to calculate best intercept and slope values
def optimize_evo(pop_size, gens, epsilon):
    random.seed(1)

    # # Start timer
    # tic = time.perf_counter()

    # Parameters for the randmization of first population
    min_p = 0.01
    max_p = 0.5
    min_i = -0.000001
    max_i = 0.0001
    min_d = 2
    max_d = 5
    # min_p = 0
    # max_p = 1
    # min_i = 0
    # max_i = 1
    # min_d = 0
    # max_d = 1

    # Number of elite children and mutation children
    numb_elite = pop_size * 0.15
    numb_elite = round(numb_elite)
    numb_mutate = pop_size - numb_elite

    # Initial population
    population = [(random.uniform(min_p, max_p), random.uniform(min_i, max_i), random.uniform(min_d, max_d)) for _ in range(pop_size)]


    print(f"p range: {min_p} - {max_p}")
    print(f"i range:     {min_i} - {max_i}\n")
    print(f"d range:     {min_d} - {max_d}\n")

    global cur_generation
    global time_spend
    
    # Loop trough generations
    for cur_gen in range(gens):

        # print(f"Pop: {cur_gen}\n{population}\n\n")

        # Calculate fitness, lower error score is better
        fitness = []
        for individual in population:
            controller.CTE_total = 0
            fitness.append(run_with_params(individual, scroll))
        best_error = min(fitness)

        # Get the best intercept and slope values
        P, I, D = population[fitness.index(best_error)]

        # Stop loop if threshold is passed, return best values
        if best_error <= epsilon:
            params = [P, I, D]
            return params

        sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
        elite_children = [individual[0] for individual in sorted_population[:numb_elite]]
        elite_population = [elite_children[i % len(elite_children)] for i in range(pop_size)]

        new_population = []

        # Mutate crossover population
        for individual in elite_population:
            mutated_individual = list(individual)

            for i in range(len(mutated_individual)):
                mutated_individual[i] = mutated_individual[i] + np.random.normal(0, 2)

            mutated_individual = list(mutated_individual)

            new_population.append(mutated_individual)

        new_population = elite_children + new_population[len(elite_children):]

        # Update population
        population = new_population
 
        print(f"Generation {cur_gen} of size {pop_size}; best error {best_error:.0f}; \
        P {P:.6f}; I {I:.6f}; D {D:.6f}")

    
    # If threshold is not passed, check population for best values
    fitness = []

    # Calculate fitness of last population, lower error score is better
    for individual in population:
        controller.CTE_total = 0
        fitness.append(run_with_params(individual, scroll))
    best_error = min(fitness)

    # Get the best intercept and slope values of last population
    P, I, D = population[fitness.index(best_error)]
    params = [P, I, D]

    # Stop timer and print time
    # toc = time.perf_counter()
    # print(f"Evolution completed in {toc - tic:0.1f} seconds.")
    
    cur_generation = gens
    # time_spend = toc - tic

    return params
params = optimize_evo(20, 1000, 0.01)
print("----------------------")
print(params)
# creates plot
x = coords[:, 0]
y = coords[:, 1]

y_cte = 266

plt.plot(x, y, color="red", label='Car')
plt.hlines(y = [y_cte], xmin=[0], xmax=[1200], color="black", linestyles='--', lw=2, label='Road center')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('PID-controller (With Twiddle)')
plt.legend()

plt.tight_layout()
plt.savefig('PID_Twiddle.pdf')
plt.show()

pygame.quit()

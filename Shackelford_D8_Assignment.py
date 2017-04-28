# Sarah Shackelford - D8 Assignment
import csv
import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
import time

start_time = time.time()  # starting a timer!


def read_color_file(file_name):
    '''
    This function reads each line of a txt file into a list containing a
    string. It then changes each string in the lists into a list of floats.
    This is so that the actual numbers can be used later as floats.
    parameter: string - name of the file you want to import
    returns: colors - a list containing lists of floats
    '''
    colors = []
    with open('colors.txt') as c:
        reader = csv.reader(c)
        for line in reader:
            color = line[0].split()
            color = map(float, color)
            colors.append(color)
    return colors


def evaluate(colors, num_colors):
    '''
    This function evaluates the total three-dimensional distance of the colors
    in the specific order they're given in.
    parameters: colors - list of color coords, num_colors - # of colors
    returns: fitness - the total 3D distance b/w all the colors
    '''
    distances = []
    for i in range(num_colors):
        if i == 0:
            continue
        else:
            color1 = colors[i-1]
            color2 = colors[i]
        element = 0
        i += 1
        distance = 0
        for c in range(3):
            element += (color1[c] - color2[c])**2
        distance += np.sqrt(element)
        distances.append(distance)

    return sum(distances)


def best_of_100(colors, num_colors):
    '''
    This function just runs through 100 random color orderings and picks the
    one with the smallest total distance between them (best fitness)
    parameters: colors: list of color coords, num_colors - # of colors
    returns: fitness - tuple, minimum total 3D distance as per the evaluate
    func
    '''
    copy = colors[:]
    fitness = []
    for i in range(100):
        order = []
        for i in np.random.permutation(copy):
            order.append(list(i))
        fitness.append((order, evaluate(order, num_colors)))

    return sorted(fitness, key=lambda x: x[1])[0]


def two_swap_move_operator(sol, num_colors):
    '''
    This function is a move operator. It swaps two colors.
    parameters: sol - a random given color ordering, num_colors - # of colors
    returns: sol - the same color ordering that was put in with two of the
    colors swapped with each other
    '''
    while True:
        pos1 = randint(0, num_colors-1)
        pos2 = randint(0, num_colors-1)
        sol_copy = sol[:]
        if pos1 != pos2:
            sol[pos1] = sol_copy[pos2]
            sol[pos2] = sol_copy[pos1]
            return sol
        else:
            continue


def inversion_move(colors, size):
    '''
    This function is a move operator. It inverts the colors between two
    randomly selected points.
    parameters: colors - a random given color ordering, num_color - # of colors
    returns: colors - the same color ordering that was put in but with an
    inverted section
    '''
    while True:
        pos1 = randint(0, size-1)
        pos2 = randint(0, size-1)
        colors_copy = colors[:]
        if pos1 != pos2:
            colors[pos1+1:pos2+1] = colors_copy[pos2:pos1:-1]
            return colors
        else:
            continue


def hill_climber(colors, fitness, size):
    '''
    This function is a hill-climber using the first improvement method.
    parameters: colors - a list of a random given color ordering, fitness - the
    total 3D distance b/w all the colors in that order, size - # of colors
    returns: best - a tuple including the colors in the order which gives the
    best fitness & then that fitness, counter - # times loop ran to find that
    best ordering & fitness, results - a list of tuples with best fitness as
    first value & the iteration # it was achieved on as the second value
    '''

    r = 100 if size == 1000 else 500  # change itr # depending on sample size

    copy1 = colors[:]
    best = (copy1, fitness)
    results = [(best[1], 1)]
    counter = 1
    for i in range(r):  # loops the hill climber r times
        two_color_swap = inversion_move(best[0], size)
        # two_color_swap = two_swap_move_operator(best[0], size)
        new_fitness = evaluate(two_color_swap, size)
        if new_fitness < best[1]:
            counter += 1
            best = (two_color_swap, new_fitness)
            results.append((best[1], i+1))
        else:
            continue

    return best, counter, results


def perturbation(colors, size):
    '''
    This function performs two consecutive two-swaps (i.e., a perturbation)
    parameters: colors - list of lists of color coords, size - # colors
    returns: order2 - perturbed list of lists of color coords, fitness2 - int,
    fitness of the perturbed order
    '''
    copy2 = colors[:]
    order1 = inversion_move(copy2, size)
    # order1 = two_swap_move_operator(copy2, size)
    fitness1 = evaluate(order1, size)
    order2 = inversion_move(order1, size)
    # order2 = two_swap_move_operator(order1, size)
    fitness2 = evaluate(order2, size)
    return order2, fitness2


def iterated_local_search(colors, fitness, size):
    '''
    This function performs an iterated local search (100 iterations) on a set
    of colors.
    parameters: colors - list, a random given color ordering, fitness - the
    total 3D distance b/w all the colors in that order, size - # of colors
    returns: best - a tuple including a random color ordering that gives the
    best fitness and then that best fitness, results - a list of tuples with
    best fitness as first value & the iteration # it was achieved on as the
    second value
    '''

    r = 100 if size == 1000 else 250  # change itr # depending on sample size

    copy3 = colors[:]
    first_sol = hill_climber(copy3, fitness, size)
    best = first_sol[0]
    results = [(best[1], 1)]
    for i in range(r):  # loops the local search 100 times
        perturbed = perturbation(copy3, size)
        new_soln = hill_climber(perturbed[0], perturbed[1], size)
        hc = new_soln[0]
        if hc[1] < best[1]:
            best = hc
            results.append((best[1], i+1))
        else:
            continue

    return best, results


def evolution(colors, size):
    '''
    This function improves the total 3D distance between the set of given
    colors by evolution. We consider each color ordering as a person, we
    evaluate each of those people's fitnesses, and then we pick two of them
    randomly to have children several times. We then replace the least fit
    people in the poplation with those children.
    parameters: colors - list, a random given color ordering, size - # of
    colors
    returns: tuple, the best order and fitness after so many evolutions
    '''
    copy4 = colors[:]
    population = []
    for gen in range(100):  # loops the evolution 100 times
        for p in range(50):  # creates a population of size 50
            person = []
            for i in np.random.permutation(copy4):
                person.append(list(i))
            fitness = evaluate(person, size)
            population.append((fitness, person))

        mom = tournament_selection(population)
        dad = tournament_selection(population)
        child1 = one_point_crossover(mom, dad)
        child2 = one_point_crossover(mom, dad)
        mutated1 = inversion_move(child1, size)
        mutated2 = inversion_move(child2, size)
        # mutated1 = two_swap_move_operator(child1, size)
        # mutated2 = two_swap_move_operator(child2, size)
        fitness1 = evaluate(mutated1, size)
        fitness2 = evaluate(mutated2, size)
        child1 = (fitness1, mutated1)
        child2 = (fitness2, mutated2)
        next_gen = replace_worst(population, child1, child2, size)

    return sorted(next_gen, key=lambda x: x[0])[0]


def tournament_selection(pop):
    '''
    This function selects the best of two random "people" in a "population."
    parameter: pop - a list of tuples containing "people"...(order, fitness)
    returns: tuple, the color ordering with the best fitness
    '''

    # contestants = random.sample(pop, 2)  # contestants are 2 random people from pop
    contestants = sorted(pop, key=lambda x: x[0])[:2]  # contestants are 2 best in the pop
    option1 = contestants[0]
    option2 = contestants[1]
    if option1[0] >= option2[0]:
        return option2
    else:
        return option1


def one_point_crossover(mom, dad):
    '''
    This function takes the first bit of one person and the last bit of another
    to make a "child."
    parameter: mom, dad - lists of listss containing the 3D color coords
    returns: list, a child
    '''
    mom = mom[1]
    dad = dad[1]
    crossover_pt = randint(0, len(mom)-1)
    return mom[:crossover_pt] + dad[crossover_pt:]


def replace_worst(population, child1, child2, size):
    '''
    This function sorts the population, cuts out the people with the worst two
    fitnesses, then puts in two new-born childredn (wether the children are
    better or not)
    parameter: population - list of tuples (order, fitness), child1, child2 -
    tuple of a color ordering and that ordering's fitness, size - # of colors
    returns: population - list of tuples (order, fitness)
    '''

    pop_wo_worst2 = sorted(population, key=lambda x: x[0])[:size-2]
    pop_wo_worst2.append(child1)
    pop_wo_worst2.append(child2)
    population = pop_wo_worst2
    return population


def plot_colors(colors, local_search, iter_local, evolution):
    '''
    This function plots the colors (mostly inspired by Gabriella's code)
    parameters: colors - list of lists of & color coords, local_search - list
    of lists of color coords from filsm, iter_local - list of lists of color
    coords from ils, evolution - list of lists of color coords from genetic
    alg, x_width - int, changes width of pop-up window depending on # colors
    using
    '''

    img = np.zeros((10, len(colors), 3))

    if len(colors) == 1000:  # this controls the img width based on # of colors
        x_width = 16
    else:
        x_width = 8

    for i in range(len(colors)):
        img[:, i, :] = colors[i]

    fig, axes = plt.subplots(nrows=4, figsize=(x_width, 8))
    fig.canvas.set_window_title('100 colors, inversion move, best of 30')
    axes[0].set_title('Original Colors')
    axes[0].imshow(img, interpolation='nearest')
    axes[0].axis('off')

    for i in range(len(colors)):
        img[:, i, :] = local_search[i]

    axes[1].set_title('Hill climber: first improvement local search method')
    axes[1].imshow(img, interpolation='nearest')
    axes[1].axis('off')

    for i in range(len(colors)):
        img[:, i, :] = iter_local[i]

    axes[2].set_title('Iterated Local Search')
    axes[2].imshow(img, interpolation='nearest')
    axes[2].axis('off')

    for i in range(len(colors)):
        img[:, i, :] = evolution[i]

    axes[3].set_title('Evolution')
    axes[3].imshow(img, interpolation='nearest')
    axes[3].axis('off')

    plt.show()


def plot_run(results1, results2, size):
    '''
    This function plots the best runs for the first improvement hill climber &
    the iterated local search algorithms.
    parameters: results1, results2 - a list of tuples with best fitness as
    first value & the iteration # it was achieved on as the second value
    '''

    if size == 1000:  # determine y ranges for different color sample sizes
        x, y, z = 620, 670, 10
    elif size == 100:
        x, y, z = 55, 70, 5
    elif size == 10:
        x, y, z = 3.5, 7.1, 0.4

    x_data1 = [i[1] for i in results1]
    y_data1 = [fit[0] for fit in results1]
    x_data2 = [i[1] for i in results2]
    y_data2 = [fit[0] for fit in results2]

    # plot the best run
    plt.subplot(221)
    plt.plot(x_data1, y_data1, 'b', marker='o', label='Hill climber: first improvement local search method')
    plt.xticks(x_data1)
    plt.yticks(np.arange(x, y, z))
    plt.legend()
    plt.ylabel('fitness')
    plt.xlabel('number iterations')

    plt.subplot(222)
    plt.plot(x_data2, y_data2, 'g', marker='o', label='Iterated Local Search')
    plt.xticks(x_data2)
    plt.yticks(np.arange(x, y, z))
    plt.legend()
    plt.ylabel('fitness')
    plt.xlabel('number iterations')

    plt.show()


def plot_stats(alg1, alg2, alg3, size):

    if size == 1000:  # determine y ranges for different color sample sizes
        x1, y1, z1 = 615, 665, 10  # for plot1
        x2, y2, z2 = 625, 665, 5  # for box plots
    elif size == 100:
        x1, y1, z1 = 50, 65, 5
        x2, y2, z2 = 50, 65, 5
    elif size == 10:
        x1, y1, z1 = 3, 7, 0.4
        x2, y2, z2 = 4, 6.5, 0.3

    # plot of all fitnesses over all 30 iterations
    plt.plot(alg1, 'b', marker='o', label='first improvement')
    plt.plot(alg2, 'g', marker='s', label='iterated local search')
    plt.plot(alg3, 'm', marker='^', label='evolution')
    plt.yticks(np.arange(x1, y1, z1))
    plt.legend()
    plt.show()

    # boxplots of fitnesses of each algorithm
    fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
    axes[0].boxplot(alg1, vert=False)
    axes[0].set_title('Hill climber: first improvement local search method')
    axes[0].xaxis.set_ticks(np.arange(x2, y2, z2))
    axes[1].boxplot(alg2, vert=False)
    axes[1].set_title('Iterated Local Search')
    axes[1].xaxis.set_ticks(np.arange(x2, y2, z2))
    axes[2].boxplot(alg3, vert=False)
    axes[2].set_title('Evolution')
    axes[2].xaxis.set_ticks(np.arange(x2, y2, z2))

    plt.show()


def stats(colors, rand_colors, fitness, size):
    '''
    This function calcs the mean, min, and standard dev for all the algorithms
    over 30 separate runs.
    parameters: colors - list of lists of original color coords, rand_colors -
    list of lists of rand color coords, fitness - int, fitness of rand color
    ordering, size - int, # colors
    returns: best_alg1_run, best_alg2_run, best_alg3_run - list including a
    tuple, int, & list of tuples; tuple contains the best colors with their
    fitness, int is the # times loop ran to find that tuple, and the list of
    tuples is best fitness as first value & the iteration # it was achieved on
    as the second value, alg1_fit, alg2_fit, alg3_fit - lists of all fitness
    over all the runs
    '''

    itr = 10 if size == 1000 else 30  # changes # itrs based on sample size

    copy5 = colors[:]
    copy6 = rand_colors[:]
    alg1, alg2, alg3 = [], [], []
    for i in range(itr):  # loops all 3 algorithms itr times
        alg1.append(hill_climber(copy6, fitness, size))
        alg2.append(iterated_local_search(copy6, fitness, size))
        alg3.append(evolution(copy5, size))

    alg1_fit = [a1[0][1] for a1 in alg1]
    alg2_fit = [a2[0][1] for a2 in alg2]
    alg3_fit = [a3[0] for a3 in alg3]

    best_alg1_run = sorted(alg1, key=lambda x: x[0][1])[0]
    best_alg2_run = sorted(alg2, key=lambda x: x[0][1])[0]
    best_alg3_run = sorted(alg3, key=lambda x: x[0])[0]

    means = [np.mean(alg1_fit), np.mean(alg2_fit), np.mean(alg3_fit)]
    mins = [best_alg1_run[0][1], best_alg2_run[0][1], best_alg3_run[0]]
    stds = [np.std(alg1_fit), np.std(alg2_fit), np.std(alg3_fit)]

    print '\n', 'Average, best (min), and standard devs after %d loops:' % (itr)
    print 'First Improv. average:', means[0]
    print 'First Improv. best (min):', mins[0]
    print 'First Improv. standard dev:', stds[0]
    print 'Iterated average:', means[1]
    print 'Iterated best (min):', mins[1]
    print 'Iterated standard dev:', stds[1]
    print 'Evolution average:', means[2]
    print 'Evolution best (min):', mins[2]
    print 'Evolution standard dev:', stds[2]

    return best_alg1_run, best_alg2_run, best_alg3_run, alg1_fit, alg2_fit, alg3_fit


imported_file = read_color_file('colors.txt')  # import the file colors.txt
# num_colors = int(imported_file[0][0])  # the first line of the file
print 'How many colors would you like to use?'
response = raw_input('Enter either 10, 100, or 1000:')  # user decides # colors to use
while True:  # check if input is what I want it to be before passing it through
    if response  not in ['10', '100', '1000']:
        print 'Sorry, try again. You must enter 10, 100, or 1000.'
        response = raw_input('Enter either 10, 100, or 1000:')
        continue
    else:
        r = int(response) + 1
        break

colors = imported_file[1:r]  # only use some of the colors
num_colors = len(colors)
print '\n', 'This run uses %d colors.' % (num_colors)
rand_color_order = []  # loop creates a list of lists instead of arrays
for i in np.random.permutation(colors):
    rand_color_order.append(list(i))

# One random ordering
print 'Randomly generated solutions:'
fitness = evaluate(rand_color_order, num_colors)
print 'Fitness of randomly generated solution:', fitness


# Random best solution from 100-iteration loop
rand_search = best_of_100(colors, num_colors)
print 'Best fitness of 100 random color orderings:', rand_search[1]

# Hill-climb: first improvement local search method (filsm)
print '\n', 'First Improvement Hill-climber:'
# via inversion move operator
filsm = hill_climber(rand_color_order, fitness, num_colors)
print 'Fitness using an inversion move operator:', filsm[0][1]
print 'The hill-climber ran %d times to find this solution.' % (filsm[1])

# Iterated local search (ils)
print '\n', 'Iterated local search:'
ils = iterated_local_search(rand_color_order, fitness, num_colors)
print 'Fitness via iterated local search:', ils[0][1]

# Evolutionary Algorithm
print '\n', 'Evolutionary Algorithm:'
genetic_algorithm = evolution(colors, num_colors)
print 'Fitness via evolutionary algorithm:', genetic_algorithm[0]

# Average & standard deviation
all_stats = stats(colors, rand_color_order, fitness, num_colors)
print("--- %s seconds ---" % (time.time() - start_time))  # ending my timer
plot_colors(rand_color_order, all_stats[0][0][0], all_stats[1][0][0], all_stats[2][1])
plot_stats(all_stats[3], all_stats[4], all_stats[5], num_colors)
plot_run(all_stats[0][2], all_stats[1][1], num_colors)

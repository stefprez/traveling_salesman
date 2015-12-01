#!/usr/bin/env python

import math
import numpy
import sys
import copy
import csv
import matplotlib.pyplot as plot
import Queue
import time
import random
import pprint
import cProfile
import numpy.linalg as linalg

sqrt = math.sqrt


def main():
    cities = loadInCities()
    # greedy_tour = greedy_algorithm(cities)
    # greedy_tour.plot_tour()
    # for num_cities in range(16, 20):
    #     ich_tour = in_class_heuristic(cities, num_cities)
    #     ich_tour.save_tour("ICH_{0}.pdf".format(num_cities))
    #     uniform_cost_tour = uniform_cost(cities, num_cities)
    #     uniform_cost_tour.save_tour("uniform_{0}.pdf".format(num_cities))
    #     uniform_cost_tour.print_tour()

    # uniform_cost_tour = uniform_cost(cities, 2)
    # uniform_cost_tour.save_tour("uniform_{0}.pdf".format(2))
    # uniform_cost_tour.print_tour()

#     genectic_tour = genetic_algorithm(cities, 100)
#     genectic_tour.plot_tour()
    # genectic_tour.save_tour("genetic_algorithm.pdf")
#     genectic_tour.print_tour()


    simulated_annealing_tour = simulated_annealing(cities)

def loadInCities():
    cities = []
    with open('cities.csv', 'r') as city_file:
        reader = csv.reader(city_file)
        for index, line in enumerate(reader):
            city = City(line[0], line[1], index)
            cities.append(city)
    return cities


class Tour(object):
    # distance_map = {}

    def __init__(self, tour=None):
        if tour is None:
            self._tour = []
        else:
            self._tour = tour
        self.update_distance()

    def __cmp__(self, other):
        if self is None:
            return False
        elif other is None:
            return False
        else:
            return cmp(self._distance, other._distance)

    def __len__(self):
        return len(self._tour)

    def get_distance(self):
        # if self._distance == 0.0:
        #     tot_dist = 0.0
        #     end_tour_index = len(self._tour) - 1
        #     for index in xrange(0, end_tour_index):
        #         city1 = self._tour[index]
        #         city2 = self._tour[index + 1]
        #         tot_dist += Tour.distance_between_cities(city1, city2)

        #     tot_dist += Tour.distance_between_cities(self._tour[0],
        #                                              self._tour[end_tour_index])

        #     self._distance = tot_dist
        return self._distance

    def get_tour(self):
        return self._tour

    def add_city_and_update(self, city):
        self._tour.append(city)
        self.update_distance_old()

    def add_city(self, city):
        self._tour.append(city)

    def update_distance_old(self):
        old_last_city = self._tour[-2]
        new_last_city = self._tour[-1]
        self._distance += Tour.distance_between_cities(old_last_city,
                                                       new_last_city)

    def update_distance(self):
        tour = self._tour
        tot_dist = 0.0
        if tour:
            end_tour_index = len(tour) - 1
            for index in xrange(0, end_tour_index):
                city1 = tour[index]
                city2 = tour[index + 1]
                tot_dist += Tour.distance_between_cities(city1, city2)

            tot_dist += Tour.distance_between_cities(tour[0], tour[end_tour_index])

        self._distance = tot_dist

    @staticmethod
    def distance_between_cities(city_1, city_2):
        return sqrt(
            ((city_2.x-city_1.x)**2) +
            ((city_2.y - city_1.y)**2))

    def print_tour(self):
        print "Tour Distance: {}".format(self.get_distance())
        print "Tour: "
        for city in self._tour:
            print "{0}".format(city.index)

    def get_tour_string(self):
        string = ""
        string += "Tour Distance: {}\n".format(self.get_distance())
        string += "Tour: \n"
        for city in self._tour:
            string += "{0}\n".format(city.index)
        return string

    def plot_tour(self):
        self._plot_setup()
        plot.show()

    def _plot_setup(self):
        cities_x = []
        cities_y = []
        for city in self._tour:
            cities_x.append(city.x)
            cities_y.append(city.y)
        cities_x.append(self._tour[0].x)
        cities_y.append(self._tour[0].y)
        plot.axis([-5, 105, -5, 105])
        plot.title("Tour Distance: {0}".format(self.get_distance()))
        plot.plot(cities_x, cities_y, marker="o")

    def save_tour(self, filename="default.png"):
        self._plot_setup()
        plot.savefig(filename)
        plot.clf()  # Clear figure state

    def is_valid(self):
        return len(self._tour) == len(set(self._tour))


class City(object):
    def __init__(self, x, y, index):
        self.x = int(x)
        self.y = int(y)
        self.index = int(index)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.index


def greedy_algorithm(cities):
    # cities = cities[:50]
    start_time = time.time()
    starting_cities = copy.deepcopy(cities)
    min_tour = Tour()
    min_tour_distance = sys.maxint
    for best_start, starting_city in enumerate(starting_cities):
        bad_tour = False
        current_tour = Tour()
        current_tour.add_city(starting_city)
        temp_cities = copy.deepcopy(cities)
        temp_cities.pop(best_start)
        min_city_index = -1
        running_total = 0
        last_city = starting_city
        while temp_cities:
            min_distance = sys.maxint
            for counter, city in enumerate(temp_cities):
                distance = Tour.distance_between_cities(last_city, city)
                if distance < min_distance:
                    min_distance = distance
                    min_city_index = counter
            closest_city = temp_cities.pop(min_city_index)
            current_tour.add_city_and_update(closest_city)
            last_city = closest_city
            running_total += min_distance
            if running_total > min_tour_distance:
                bad_tour = True
                break
        current_tour.add_city_and_update(starting_city)
        if (not bad_tour) and (running_total < min_tour_distance):
            min_tour_distance = running_total
            min_tour = current_tour

    end_time = time.time()
    total_time = end_time - start_time
    print "Run Time for Greedy Algorithm: {0}".format(total_time)
    return min_tour


def genetic_algorithm(cities, pop_size, num_cities=119):
    random.seed(12345)
    population = get_starting_population(cities, pop_size)
    generation = 0

    start_time = time.time()

    while generation < 20000:
        curr_min_tour = get_min_tour(population)

        if generation % 250 == 0:
            min_dist = curr_min_tour.get_distance()
            time_gen = (time.time() - start_time) / 250
            print "Time per gen: {}".format(time_gen)

            # curr_min_tour.print_tour()
            # curr_min_tour.plot_tour()
            print "Generation: {0} Min tour distance: {1}".format(generation,
                                                                  min_dist)
            start_time = time.time()

        new_population = []

        for index in xrange(0, len(population), 2):
            tour1, tour2 = get_parents_for_breeding(population, curr_min_tour)

            # Mutate pairs and generate new children
            child1, child2 = mutate_and_breed(tour1, tour2)

            # New children are new population
            new_population.append(child1)
            new_population.append(child2)

        population = new_population
        generation += 1
    return get_min_tour(population)


def get_parents_for_breeding(population, min_tour):
    random_parent = population[random.randint(1, len(population) - 1)]

    chance_for_alpha = .25

    if random.uniform(0, 1) > chance_for_alpha:
        # Alpha not selected, pick random parent
        random_alpha = population[random.randint(1, len(population) - 1)]
    else:
        # Alpha selected
        random_alpha = min_tour

    return random_parent, random_alpha


def fitness(tour):
    return float(1) / tour._distance


def mutate_and_breed(parent1, parent2):
    tour_len = len(parent1)
    p1_tour = parent1.get_tour()
    p2_tour = parent2.get_tour()

    # Get random start and stop values
    start, stop = get_two_rand_ints(tour_len)

   

    # Get slices
    slice1 = p1_tour[start:stop]
    slice2 = p2_tour[start:stop]

    # Make maps of slices
    slice1_map = {city: slice_ind
                  for slice_ind, city
                  in enumerate(slice1)}

    slice2_map = {city: slice_ind
                  for slice_ind, city
                  in enumerate(slice2)}

    # Make map from a city in a slice to it's new city
    child1_dupe_map = {}
    for index, city in enumerate(slice2):
        if city not in slice1_map:
            target_city = slice1[index]
            while target_city in slice2_map:
                index = slice2_map[target_city]
                target_city = slice1[index]
            child1_dupe_map[city] = target_city

    child2_dupe_map = {}
    for index, city in enumerate(slice1):
        if city not in slice2_map:
            target_city = slice1[index]
            while target_city in slice1_map:
                index = slice1_map[target_city]
                target_city = slice2[index]
            child2_dupe_map[city] = target_city

    child1 = [slice2[index - start]
              if index >= start and index < stop  # insert slice
              else child1_dupe_map[city]
              if city in child1_dupe_map  # get rid of duplicates
              else city  # otherwise use parent tour
              for index, city
              in enumerate(p1_tour)]

    child2 = [slice1[index - start]
              if index >= start and index < stop  # insert slice
              else child2_dupe_map[city]
              if city in child2_dupe_map  # get rid of duplicates
              else city  # otherwise use parent tour
              for index, city
              in enumerate(p2_tour)]

    # Mutate
    child1 = mutate(child1)
    child2 = mutate(child2)

    return child1, child2


def get_two_rand_ints(max_value):
    low = 0
    high = 0
    while low == high:
        rand1 = random.randint(0, max_value)
        rand2 = random.randint(0, max_value)
        low = min(rand1, rand2)
        high = max(rand1, rand2)
    return low, high


def mutate(tour, alpha=None):
    if alpha is None:
        alpha = 0.4

    if random.random() <= alpha:
        # Mutate
        low, high = get_two_rand_ints(len(tour) - 1)

        temp = tour[low]
        tour[low] = tour[high]
        tour[high] = temp

    return Tour(tour)


def get_starting_population(cities, pop_size):
    start_tour = greedy_algorithm(cities)
    start_tour._tour.pop()
    start_tour.update_distance()
    start_tour_list = start_tour.get_tour()

    tour_list_len = len(start_tour_list)
    start_pop = [Tour(random.sample(start_tour_list, tour_list_len))
                 for _
                 in xrange(0, pop_size)]

    # start_pop = [mutate(start_tour_list, 1)
    #              for _
    #              in xrange(0, pop_size - 1)]

    # start_pop.insert(0, start_tour)

    return start_pop


def get_min_tour(tours):
    min_tour = None
    min_tour_distance = sys.maxint

    for tour in tours:
        tour_dist = tour._distance
        if tour_dist < min_tour_distance:
            min_tour = tour
            min_tour_distance = tour_dist

    return min_tour

def get_random_tour(cities):
    seed_tour = greedy_algorithm(cities)
    seed_tour._tour.pop()
    seed_tour.update_distance()
    seed_tour_list = seed_tour._tour
    
    tour_list_len = len(seed_tour_list)
    
    return Tour(random.sample(seed_tour_list, tour_list_len))

def simulated_annealing(cities):
    start_time = time.time()
    tour = get_random_tour(cities)
    tour_list = tour._tour
    print tour._distance
    
    finished = False
    swaps = 0
    while not finished:
        
        for _ in xrange(0, 15000):
#             if count % 500 == 0:
#                 print "Count: ", count
            finished = True
            swapped, new_tour = two_opt_swap(tour)
            if swapped:
                tour = new_tour
                swaps += 1
                finished = False
                break
    finished_tour = tour
    elapsed = time.time() - start_time
    print "Time: ", elapsed
    print swaps
    print finished_tour._distance
    finished_tour.plot_tour()

def two_opt_swap(tour):
    start, stop = get_two_rand_ints(len(tour) - 1)
    start_dist = tour._distance
    working_tour = [city for city in tour._tour]

    while start < stop:
        temp = working_tour[start]
        working_tour[start] = working_tour[stop]
        working_tour[stop] = temp
        start += 1
        stop -= 1
        
    end_tour = Tour(working_tour)
    end_dist = end_tour._distance
    
    if end_dist < start_dist:
#         print "End: ", end_dist
#         print "Start: ", start_dist
        short_tour = end_tour
        swapped = True
    else:
        short_tour = tour
        swapped = False
    return swapped, short_tour

if __name__ == "__main__":
    # cProfile.run('main()')
    main()

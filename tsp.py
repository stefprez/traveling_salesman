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

    genectic_tour = genetic_algorithm(cities, 30)
    genectic_tour.plot_tour()
    # genectic_tour.save_tour("genetic_algorithm.pdf")
    genectic_tour.print_tour()


def loadInCities():
    cities = []
    with open('cities.csv', 'r') as city_file:
        reader = csv.reader(city_file)
        counter = 0
        for index, line in enumerate(reader):
            city = City(line[0], line[1], index)
            cities.append(city)
            counter += 1
    return cities


class LineSegment(object):
    def __init__(self, city1, city2):
        self.city1 = city1
        self.city2 = city2
        self.x1 = city1.x
        self.y1 = city1.y
        self.x2 = city2.x
        self.y2 = city2.y
        self._minx = None
        self._maxx = None
        self._miny = None
        self._maxy = None

    def get_min_x(self):
        if self._minx is None:
            self._minx = min(self.x1, self.x2)

        return self._minx

    def get_max_x(self):
        if self._maxx is None:
            self._maxx = max(self.x1, self.x2)

        return self._maxx

    def get_min_y(self):
        if self._miny is None:
            self._miny = min(self.y1, self.y2)

        return self._miny

    def get_max_y(self):
        if self._maxy is None:
            self._maxy = max(self.y1, self.y2)

        return self._maxy

    def on_segment(self, city):
        city_x = city.x
        city_y = city.y
        if (city_x <= self.get_max_x() and
           city_x >= self.get_min_x() and
           city_y <= self.get_max_y() and
           city_y >= self.get_min_y()):
            return True
        else:
            return False

    def orientation(self, city):
        val = ((city.y - self.y1) * (self.x2 - city.x) -
               (city.x - self.x1) * (self.y2 - city.y))

        if val > 0:
            # Clockwise
            return 1
        elif val < 0:
            # Counterclockwise
            return 2
        else:
            # Colinear
            return 0


class Tour(object):
    def __init__(self, tour=None, distance=None, complete=None):
        if tour is None:
            self._tour = []
        else:
            self._tour = tour

        if distance is None:
            self._distance = 0.0
            if self._tour:
                self.get_distance()
        else:
            self._distance = distance

        if complete is None:
            self._complete = False
        else:
            self._complete = complete

    def __cmp__(self, other):
        return cmp(self.get_distance(), other.get_distance())

    def __len__(self):
        return len(self._tour)

    def get_distance(self):
        if self._distance == 0.0:
            tot_dist = 0.0
            for index in xrange(0, len(self._tour) - 1):
                city1 = self._tour[index]
                city2 = self._tour[index + 1]
                tot_dist += Tour.distance_between_cities(city1, city2)

            self._distance = tot_dist

        return self._distance

    def get_tour(self):
        return self._tour

    def add_city_and_update(self, city):
        self._tour.append(city)
        self.update_distance()

    def add_city(self, city):
        self._tour.append(city)

    def update_distance(self):
        old_last_city = self._tour[-2]
        new_last_city = self._tour[-1]
        self._distance += Tour.distance_between_cities(old_last_city,
                                                       new_last_city)

    def _get_lines_to_check(self):
        lines_to_check = []
        tour = self.get_tour()
        for index in range(len(tour) - 3):
            city_1 = tour[index]
            city_2 = tour[index + 1]

            if (city_1.x == city_2.x):
                # ignore vertical line segment
                continue

            line = LineSegment(city_1, city_2)
            lines_to_check.append(line)

        return lines_to_check

    def has_intersecting_lines(self):
        tour = self.get_tour()

        last_city = tour[-1]
        next_to_last_city = tour[-2]
        new_line = LineSegment(last_city, next_to_last_city)

        lines_to_check = self._get_lines_to_check()

        for old_line in lines_to_check:
            o1 = new_line.orientation(old_line.city1)
            o2 = new_line.orientation(old_line.city2)
            o3 = old_line.orientation(new_line.city1)
            o4 = old_line.orientation(new_line.city2)

            if (o1 != o2) and (o3 != o4):
                return True
            elif (o1 == 0 and new_line.on_segment(old_line.city1)):
                return True
            elif (o2 == 0 and new_line.on_segment(old_line.city2)):
                return True
            elif (o3 == 0 and old_line.on_segment(new_line.city1)):
                return True
            elif (o4 == 0 and old_line.on_segment(new_line.city2)):
                return True
        return False

    @staticmethod
    def distance_between_cities(city_1, city_2):
        x1 = city_1.x
        y1 = city_1.y
        x2 = city_2.x
        y2 = city_2.y

        return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

    def print_tour(self):
        print "Tour Distance: {}".format(self._distance)
        print "Tour: "
        for city in self._tour:
            print "{0}".format(city.index)

    def plot_tour(self):
        self._plot_setup()
        plot.show()

    def _plot_setup(self):
        cities_x = []
        cities_y = []
        for city in self._tour:
            cities_x.append(city.x)
            cities_y.append(city.y)
        plot.axis([-5, 105, -5, 105])
        plot.title("Tour Distance: {0}".format(self._distance))
        plot.plot(cities_x, cities_y, marker="o")

    def save_tour(self, filename="default.png"):
        self._plot_setup()
        plot.savefig(filename)
        plot.clf()  # Clear figure state

    def get_copy(self):
        return Tour(list(self.get_tour()), self.get_distance())

    def is_complete(self):
        return self._complete

    def set_complete(self):
        self._complete = True


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


def greedy_algorithm(cities):
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


def uniform_cost(cities, num_cities=119):
    start_time = time.time()

    new_cities = copy.deepcopy(cities)
    new_cities = new_cities[:num_cities - 1]

    start_city = new_cities.pop(0)
    start_tour = Tour()
    start_tour.add_city(start_city)

    remaining_cities = list(new_cities)

    queue = Queue.PriorityQueue()
    queue.put((start_tour, remaining_cities))

    while True:
        current_tour, remaining_cities = queue.get()

        if not remaining_cities:
            if current_tour.is_complete():
                end_time = time.time()
                total_time = end_time - start_time
                print "Run Time for {0} cities: {1}".format(num_cities,
                                                            total_time)
                return current_tour
            else:
                current_tour.add_city(start_city)
                current_tour.set_complete()
                queue.put((current_tour, remaining_cities))
        else:
            for index, city in enumerate(remaining_cities):
                partial_tour = current_tour.get_copy()
                partial_tour.add_city(city)

                temp_remaining_cities = list(remaining_cities)
                temp_remaining_cities.pop(index)

                queue.put((partial_tour, temp_remaining_cities))


def in_class_heuristic(cities, num_cities=119):
    start_time = time.time()

    new_cities = copy.deepcopy(cities)
    new_cities = new_cities[:num_cities - 1]

    start_city = new_cities.pop(0)
    start_tour = Tour()
    start_tour.add_city(start_city)

    remaining_cities = list(new_cities)

    queue = Queue.PriorityQueue()
    queue.put((start_tour, remaining_cities))

    while True:
        current_tour, remaining_cities = queue.get()

        if not remaining_cities:
            if current_tour.is_complete():
                end_time = time.time()
                total_time = end_time - start_time
                print "Run Time for {0} cities: {1}".format(num_cities,
                                                            total_time)
                return current_tour
            else:
                current_tour.add_city_and_update(start_city)
                current_tour.set_complete()
                queue.put((current_tour, remaining_cities))
        else:
            for index, city in enumerate(remaining_cities):
                partial_tour = current_tour.get_copy()
                partial_tour.add_city(city)

                if partial_tour.has_intersecting_lines():
                    continue
                else:
                    partial_tour.update_distance()
                    temp_remaining_cities = list(remaining_cities)
                    temp_remaining_cities.pop(index)
                    queue.put((partial_tour, temp_remaining_cities))


def genetic_algorithm(cities, pop_size, num_cities=119):
    population = get_starting_population(cities, pop_size)
    generation = 0

    while generation < 1000000:

        if generation % 100 == 0:
            curr_min_tour = get_min_tour(population)
            min_dist = curr_min_tour.get_distance()
            print "Generation: {0} Min tour distance: {1}".format(generation,
                                                                  min_dist)

        # random.shuffle(population)
        new_population = []
        choices = get_choices(population)
        weight_total = sum(weight for choice, weight in choices)

        for index in xrange(0, len(population), 2):
            # tour1 = population[index]
            # tour2 = population[index + 1]
            tour1 = get_parent_for_breeding(choices, weight_total)
            tour2 = get_parent_for_breeding(choices, weight_total)

            # Mutate pairs and generate new children
            child1, child2 = mutate_and_breed(tour1, tour2)

            # New children are new population
            new_population.append(child1)
            new_population.append(child2)

        population = new_population
        generation += 1


def get_parent_for_breeding(choices, total):
    # O(n)
    r = random.uniform(0, total)
    upto = 0

    for choice, weight in choices:
        upto += weight
        if upto >= r:
            return choice

    assert False, "Shouldn't get here."


def get_choices(population):
    # O(n)
    choices = []
    for parent in population:
        weight = fitness(parent)
        choice = (parent, weight)
        choices.append(choice)
    return choices


def fitness(tour):
    return float(1) / tour.get_distance()**2


# def get_slice_vals(slice1, slice2):
#     slice_vals = {}
#     for cur_slice in [slice1, slice2]:
#         for val in cur_slice:
#             if val not in slice_vals:
#                 slice_vals[val] = True

#     return slice_vals


def mutate_and_breed(parent1, parent2):
    p1 = parent1
    p2 = parent2
    tour_len = len(parent1)
    # if tour_len == 0:
    #     print "Length of tour list is 0!"
    #     sys.exit(1)

    parent1 = parent1.get_tour()
    parent2 = parent2.get_tour()

    # Get random start and stop values
    start, stop = get_two_rand_ints(tour_len)

    # Get slices
    slice1 = parent1[start:stop]
    slice2 = parent2[start:stop]
    empty_slice = [None for _ in slice1]

    # slice_vals = get_slice_vals(slice1, slice2)

    # Set slice areas to None for duplicate checking
    parent1[start:stop] = empty_slice
    parent2[start:stop] = empty_slice

    # Replace would-be duplicates in parent2
    for slice_ind, val in enumerate(slice1):
        try:
            par_ind = parent2.index(val)
            parent2[par_ind] = slice2[slice_ind]
        except:
            # Both slices contain value
            pass

    # Replace would-be duplicates in parent1
    for slice_ind, val in enumerate(slice2):
        try:
            par_ind = parent1.index(val)
            parent1[par_ind] = slice1[slice_ind]
        except:
            # Both slices contain value
            pass

    parent1[start:stop] = slice2
    parent2[start:stop] = slice1

    # Mutate?

    # Now children
    child1 = Tour(parent1)
    child2 = Tour(parent2)

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


def mutate(child):
    alpha = 0.001

    if random.random() <= alpha:
        # Mutate
        tour = child.get_tour()
        low, high = get_two_rand_ints(len(tour) - 1)

        temp = tour[low]
        tour[low] = tour[high]
        tour[high] = temp

        return Tour(tour)
    else:
        # No mutation
        return child


def get_starting_population(cities, pop_size):
    start_tour = greedy_algorithm(cities)
    start_tour_list = start_tour.get_tour()
    tour_list_len = len(start_tour_list)

    start_pop = [Tour(random.sample(start_tour_list, tour_list_len))
                 for _
                 in xrange(0, pop_size - 1)]

    start_pop.append(start_tour)

    return start_pop


def get_min_tour(tours):
    min_tour = None
    min_tour_distance = sys.maxint

    for tour in tours:
        tour_dist = tour.get_distance()
        if tour_dist < min_tour_distance:
            min_tour = tour
            min_tour_distance = tour_dist

    return min_tour

if __name__ == "__main__":
    main()

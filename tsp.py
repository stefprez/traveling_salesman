#!/usr/bin/env python

import math
import numpy
import pylab
import sys
import copy
import csv
import matplotlib.pyplot as plot
import Queue
import time


def main():
    cities = loadInCities()
    # greedy_tour = greedy_algorithm(cities)
    # greedy_tour.plot_tour()
    for num_cities in range(2, 15):
        uniform_cost_tour = uniform_cost(cities, num_cities)
        uniform_cost_tour.save_tour("uniform_{0}.pdf".format(num_cities))
        uniform_cost_tour.print_tour()

    # uniform_cost_tour = uniform_cost(cities, 2)
    # uniform_cost_tour.save_tour("uniform_{0}.pdf".format(2))
    # uniform_cost_tour.print_tour()


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


class Tour(object):
    def __init__(self, distance=None, tour=None, complete=None):
        if distance is None:
            self._distance = 0.0
        else:
            self._distance = distance

        if tour is None:
            self._tour = []
        else:
            self._tour = tour

        if complete is None:
            self._complete = False
        else:
            self._complete = complete

    def __cmp__(self, other):
        return cmp(self.get_distance(), other.get_distance())

    def get_distance(self):
        return self._distance

    def get_tour(self):
        return self._tour

    def add_city(self, city):
        self._tour.append(city)
        self._update_distance()

    def _update_distance(self):
        if len(self._tour) >= 2:
            old_last_city = self._tour[-2]
            new_last_city = self._tour[-1]
            self._distance += Tour.distance_between_cities(old_last_city,
                                                           new_last_city)

    @staticmethod
    def distance_between_cities(city_1, city_2):
        x1 = city_1.get_x()
        y1 = city_1.get_y()
        x2 = city_2.get_x()
        y2 = city_2.get_y()

        return math.sqrt(((x2-x1)**2)+((y2-y1)**2))

    def print_tour(self):
        print "Tour Distance: {}".format(self._distance)
        print "Tour: "
        for city in self._tour:
            # print "{0}, {1}".format(city.x, city.y)
            print "{0}".format(city.get_index())

    def plot_tour(self):
        self._plot_setup()
        plot.show()

    def _plot_setup(self):
        cities_x = []
        cities_y = []
        for city in self._tour:
            cities_x.append(city.get_x())
            cities_y.append(city.get_y())
        plot.axis([-5, 105, -5, 105])
        plot.title("Tour Distance: {0}".format(self._distance))
        plot.plot(cities_x, cities_y, marker="o")

    def save_tour(self, filename="default.png"):
        self._plot_setup()
        plot.savefig(filename)
        plot.clf()  # Clear figure state

    def get_copy(self):
        return Tour(self.get_distance(), list(self.get_tour()))

    def is_complete(self):
        return self._complete

    def set_complete(self):
        self._complete = True


class City(object):
    def __init__(self, x, y, index):
        self._x = int(x)
        self._y = int(y)
        self._index = int(index)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.get_index() == other.get_index()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_index(self):
        return self._index

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y


def greedy_algorithm(cities):
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
            current_tour.add_city(closest_city)
            last_city = closest_city
            running_total += min_distance
            if running_total > min_tour_distance:
                bad_tour = True
                break
        current_tour.add_city(starting_city)
        if (not bad_tour) and (running_total < min_tour_distance):
            min_tour_distance = running_total
            min_tour = current_tour

    return min_tour


def uniform_cost(cities, num_cities=119):
    start_time = time.time()

    new_cities = copy.deepcopy(cities)
    new_cities = new_cities[:num_cities - 1]

    start_city = new_cities.pop(0)
    start_tour = None
    start_tour = Tour()
    print "Empty Starting Tour: "
    start_tour.print_tour()
    start_tour.add_city(start_city)

    remaining_cities = list(new_cities)

    queue = Queue.PriorityQueue()
    print "Starting city: {0}".format(start_city.get_index())
    print "Adding starting tour to Queue: ",  # TODO
    start_tour.print_tour()  # TODO
    print "Remaining cities: {0}".format(len(remaining_cities))  # TODO

    queue.put((start_tour, remaining_cities))

    counter = 0  # TODO
    while True:
        # raw_input("Press Enter to Continue...")  # TODO
        current_tour, remaining_cities = queue.get()
        print "Pop Queue"  # TODO
        current_tour.print_tour()  # TODO
        print "Remaining cities: {0}".format(len(remaining_cities))  # TODO

        if not remaining_cities:
            print "No remaining cities!"  # TODO
            if current_tour.is_complete():
                print "Tour is complete"  # TODO
                end_time = time.time()
                total_time = end_time - start_time
                print "Run Time for {0} cities: {1}".format(num_cities, total_time)
                return current_tour
            else:
                print "Tour is NOT complete"   # TODO
                current_tour.print_tour()  # TODO

                current_tour.add_city(start_city)
                current_tour.set_complete()

                print "Putting completed tour back onto Queue"  # TODO
                current_tour.print_tour()  # TODO

                queue.put((current_tour, remaining_cities))
        else:
            for index, city in enumerate(remaining_cities):
                print "Index: {0} City: {1}".format(index, city.get_index())
                partial_tour = current_tour.get_copy()
                partial_tour.add_city(city)

                temp_remaining_cities = list(remaining_cities)
                temp_remaining_cities.pop(index)

                print "Adding partial tour to queue"
                partial_tour.print_tour()
                print "partial tour's remaining_cities: {0}".format(len(temp_remaining_cities))
                queue.put((partial_tour, temp_remaining_cities))

if __name__ == "__main__":
    main()

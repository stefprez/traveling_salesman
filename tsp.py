#!/usr/bin/env python

import math
import numpy
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
    for num_cities in range(14, 25):
        ich_tour = in_class_heuristic(cities, num_cities)
        ich_tour.save_tour("ICH_{0}.pdf".format(num_cities))
        # uniform_cost_tour = uniform_cost(cities, num_cities)
        # uniform_cost_tour.save_tour("uniform_{0}.pdf".format(num_cities))
        # uniform_cost_tour.print_tour()

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


class LineSegment(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.slope = self.calculate_slope()
        self._b = None
        self._minx = None
        self._maxx = None

    def calculate_slope(self):
        return (self.y2 - self.y1) / float(self.x2 - self.x1)

    def get_b(self):
        if self._b is None:
            self._b = self.y1 - (self.slope * self.x1)

        return self._b

    def get_min_x(self):
        if self._minx is None:
            self._minx = min(self.x1, self.x2)

        return self._minx

    def get_max_x(self):
        if self._maxx is None:
            self._maxx = max(self.x1, self.x2)

        return self._maxx


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

    def _get_lines_to_check(self):
        lines_to_check = []
        tour = self.get_tour()
        for index in range(len(tour) - 3):
            city_1 = tour[index]
            city_2 = tour[index + 1]

            if (city_1.get_x() == city_2.get_x()):
                # ignore vertical line segment
                continue

            line = LineSegment(city_1.get_x(),
                               city_1.get_y(),
                               city_2.get_x(),
                               city_2.get_y())
            lines_to_check.append(line)

        return lines_to_check

    def has_intersecting_lines(self):
        tour = self.get_tour()
        if len(tour) < 4:
            return False

        last_city = tour[-1]
        next_to_last_city = tour[-2]
        new_line = LineSegment(last_city.get_x(),
                               last_city.get_y(),
                               next_to_last_city.get_x(),
                               next_to_last_city.get_y())

        lines_to_check = self._get_lines_to_check()

        for old_line in lines_to_check:
            if (old_line.get_max_x() < new_line.get_min_x()):
                # Line segment x intervals don't overlap
                continue

            # Commented out due to infrequency of occurences
            # if (old_line.slope == new_line.slope):
            #     print "Same slope"
            #     # lines parallel
            #     continue

            Xa = (new_line.get_b() - old_line.get_b()) / (old_line.slope - new_line.slope)

            if (Xa < max(old_line.get_min_x(), new_line.get_min_x())) or (Xa > min(old_line.get_max_x(), new_line.get_max_x)):
                # X value of intersection point is outside line segment intervals
                continue

            Ya1 = old_line.slope * Xa + old_line.get_b()
            Ya2 = new_line.slope * Xa + new_line.get_b()

            if (Ya1 != Ya2):
                continue

            if (Ya1 > min(max(new_line.y1, new_line.y2), max(old_line.y1, old_line.y2))):
                continue

            if (Ya1 < max(min(new_line.y1, new_line.y2), min(old_line.y1, old_line.y2))):
                continue

            return True

        return False

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
                current_tour.add_city(start_city)
                current_tour.set_complete()
                queue.put((current_tour, remaining_cities))
        else:
            for index, city in enumerate(remaining_cities):
                partial_tour = current_tour.get_copy()
                partial_tour.add_city(city)

                if partial_tour.has_intersecting_lines():
                    print "INTERSECTING LINES"
                    partial_tour.plot_tour()
                    continue
                else:
                    if len(partial_tour.get_tour()) > 3:
                        print "No Intersections"
                        partial_tour.plot_tour()
                    temp_remaining_cities = list(remaining_cities)
                    temp_remaining_cities.pop(index)
                    queue.put((partial_tour, temp_remaining_cities))


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import math
import numpy
import pylab
import sys
import copy
import csv
import matplotlib.pyplot as plot


def main():
    cities = loadInCities()
    greedyAlgorithm(cities)


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


def print_tour(tour):
    for city in tour:
        print "{0}, {1}".format(city.x, city.y)


class Tour(object):
    def __init__(self):
        self._distance = 0.0
        self._tour = []

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
        cities_x = []
        cities_y = []
        for city in self._tour:
            cities_x.append(city.get_x())
            cities_y.append(city.get_y())
        plot.axis([-5, 105, -5, 105])
        plot.title("Tour Distance: {0}".format(self._distance))
        plot.plot(cities_x, cities_y, marker="o")
        plot.show()


class City(object):
    def __init__(self, x, y, index):
        self._x = int(x)
        self._y = int(y)
        self._index = int(index)

    def get_index(self):
        return self._index

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y


def greedyAlgorithm(cities):
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

    min_tour.print_tour()
    min_tour.plot_tour()

if __name__ == "__main__":
    main()

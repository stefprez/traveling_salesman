#!/usr/bin/env python

import math
import numpy
import pylab
import sys
import copy
import csv


def main():
    cities = loadInCities()
    greedyAlgorithm(cities)


def distance(x1, y1, x2, y2):
    return math.sqrt(((x2-x1)**2)+((y2-y1)**2))


def distance_between_cities(city_1, city_2):
    return distance(city_1.x, city_1.y, city_2.x, city_2.y)


def loadInCities():
    cities = []
    with open('cities.csv', 'r') as city_file:
        reader = csv.reader(city_file)
        counter = 0
        for line in reader:
            city = City(line[0], line[1])
            cities.append(city)
            counter += 1
    return cities


class City(object):

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


def greedyAlgorithm(cities):
    starting_cities = copy.deepcopy(cities)
    min_tour = None
    min_tour_distance = sys.maxint
    for best_start, starting_city in enumerate(starting_cities):
        bad_tour = False
        current_tour = []
        temp_cities = copy.deepcopy(cities)
        temp_cities.pop(best_start)
        min_city_index = -1
        running_total = 0
        last_city = starting_city
        while temp_cities:
            min_distance = sys.maxint
            for counter, city in enumerate(temp_cities):
                distance = distance_between_cities(last_city, city)
                if distance < min_distance:
                    min_distance = distance
                    min_city_index = counter
            closest_city = temp_cities.pop(min_city_index)
            current_tour.append(closest_city)
            last_city = closest_city
            running_total += min_distance
            if running_total > min_tour_distance:
                bad_tour = True
                break
        if not bad_tour and running_total < min_tour_distance:
            min_tour_distance = running_total
            min_tour = current_tour

    print "Minimum Tour Distance: {}".format(min_tour_distance)
    print "Minimum Tour"
    for city in min_tour:
        print "{0}, {1}".format(city.x, city.y)

if __name__ == "__main__":
    main()

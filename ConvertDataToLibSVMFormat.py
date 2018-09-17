#!/usr/bin/env python3


filename = input('Input file name that you want the converted data to be stored in.')
datafile = 'dota2Train.csv'

file = open(filename + '.txt', 'w')

for i in open(datafile, 'r'):
	list = i.split(',')
	if (list[0] == '-'):
		file.write(list[0] + list[1] + ' ')
	else:
		file.write(list[0] + ' ')

	counter = 1;

	while counter <= len(list) - 2:
		file.write(str(counter) + ':' + list[counter] + ' ')
		counter += 1

	file.write(str(counter) + ':' + list[counter])

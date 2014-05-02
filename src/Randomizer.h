/*
 * Randomizer.h
 * 	Given by the instructor
 *  Created on: May 1, 2014
 */

#ifndef RANDOMIZER_H_
#define RANDOMIZER_H_
#include<iostream>
struct Randomizer {
	int a, b;

	Randomizer() {
		a = 1234, b = 4321;
	}

	double rand(double range_start, double range_end) {
		const int RANGE = 100;
		double c = (a + b) % (RANGE + 1);
		a = b, b = (int) c;
		return range_start + (c / RANGE) * (range_end - range_start);
	}
};


#endif /* RANDOMIZER_H_ */

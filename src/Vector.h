/*
 * Vector
 *
 *  Created on: May 1, 2014
 *      Author: Mohamed Kamal
 */

#ifndef VECTOR_
#define VECTOR_

#include "Matrix.h"

// TODO: integrate the use of Vector instead of Matrix when applicable

// TODO: make matrix functions to be template to enable matrix operations that return vector

class ColumnVector : public Matrix{
public:
	ColumnVector(int n) : Matrix(n, 1){

	}

	double& operator [](int row){
		Matrix &base = *this;
		return base[row][0];
	}


//	friend ColumnVector operator*(const Matrix &m, const ColumnVector &v){
//		ColumnVector res = m * v;
//		return res;
//	}
};


#endif /* VECTOR_ */

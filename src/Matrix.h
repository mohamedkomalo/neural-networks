/*
 * Matrix.h
 *
 *  Created on: May 1, 2014
 *      Author: Mohamed Kamal
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "Randomizer.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
using namespace std;

// Define a definition "Function" type that takes an input double and returns an output double
// Used in "Matrix.ApplyFunction"
typedef double(*Function)(double value);

// TODO: change matrix implementation

// TODO: enable changing dimension in Matrix

// TODO: add exception to wrong matrix operation instead of exit(0) error handling

class Matrix{
	class MatrixRow;			//Used in accessing the matrix elements using double subscripts
	class MatrixRowConst;		//

	int rows;
	int columns;

	vector<vector<double> > arr;

public:
	Matrix(int r, int c) : rows(r),
						   columns(c),
						   arr(vector<vector<double> >(r, vector<double>(c, 0))){
	}

	MatrixRow operator [](int i){
		return MatrixRow(*this, i);
	}

	const MatrixRowConst operator [](int i) const{
		return MatrixRowConst(*this, i);
	}

	int getRows() const{
		return rows;
	}

	int getColumns() const{
		return columns;
	}

	double get(int row, int col){
		return arr[row][col];
	}

	void set(int row, int col, double value){
		arr[row][col] = value;
	}

	void fill(double value){
		Matrix &thisMatrix = *this;
		for(int i=0; i<rows; i++){
			for(int j=0; j<columns; j++){
				thisMatrix[i][j] = value;
			}
		}
	}


	void fillRandomly(double rangeStart, double rangeEnd){
		Matrix &thisMatrix = *this;
		Randomizer r;

		for(int i=0; i<rows; i++){
			for(int j=0; j<columns; j++){
				thisMatrix[i][j] = r.rand(rangeStart, rangeEnd);
			}
		}
	}

	Matrix applyFunction(Function function) const{
		Matrix res(rows, columns);
		for(int i=0; i<rows; i++){
			for(int j=0; j<columns; j++){
				res[i][j] = function(arr[i][j]);
			}
		}
		return res;
	}

	Matrix transpose() const{
		Matrix res(columns, rows);
		for(int i=0; i<rows; i++){
			for(int j=0; j<columns; j++){
				res[j][i] = arr[i][j];
			}
		}
		return res;
	}

	double calculateMeanSquareErrorWith(const Matrix &otherMatrix) const{
		const Matrix &thisMatrix = *this;

		if(thisMatrix.rows != otherMatrix.rows || thisMatrix.columns != otherMatrix.columns){
			cout << "Mean square error must take 2 equal matrices";
			exit(0);
		}

		double res = 0;
		for(int i=0; i<thisMatrix.rows; i++){
			for(int j=0; j<thisMatrix.columns; j++){
				double t = thisMatrix[i][j] - otherMatrix[i][j];
				res += t * t;
			}
		}
		return res;
	}

	/*
	 *  Also know as Hadamard product
	 */
	Matrix elementWiseProduct(const Matrix &otherMatrix) const{
		const Matrix &thisMatrix = *this;

		if(thisMatrix.rows != otherMatrix.rows || thisMatrix.columns != otherMatrix.columns){
			cout << "Element Wise Product must take 2 equal matrices";
			exit(0);
		}

		Matrix res(rows, columns);
		for(int i=0; i<thisMatrix.rows; i++){
			for(int j=0; j<thisMatrix.columns; j++){
				res[i][j] = thisMatrix[i][j] * otherMatrix[i][j];
			}
		}
		return res;
	}

	/*
	 * Operator overloading using friend function to allow nested expressions
	 * for example:
	 * 	cout << MatrixA << MatrixB << MatrixC;
	 * 	MatrixD = MatrixA * MatrixB * MatrixC;
	 */

	friend Matrix operator*(const Matrix &m1, const Matrix &m2){
		if(m1.columns != m2.rows){
			cout << "Matrix multiply require row to be equal col";
			exit(0);
		}

		Matrix res(m1.rows, m2.columns);

		for(int i=0; i<res.rows; i++){
			for(int j=0; j<res.columns; j++){
				res[i][j] = 0;
				for(int k=0; k<m1.columns; k++){
					res[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}

		return res;
	}

	friend Matrix operator-(const Matrix &m1, const Matrix &m2){
		Matrix res(m1.rows, m2.columns);
		for(int i=0; i<res.rows; i++){
			for(int j=0; j<res.columns; j++){
				res[i][j] = m1[i][j] - m2[i][j];
			}
		}
		return res;
	}

	bool friend operator==(const Matrix &m1, const Matrix &m2){
		if(m1.rows != m2.rows || m1.columns != m2.columns){
			cout << "Matrix equivlence require the two matrices of same dimension";
			return false;
		}

		for(int i=0; i<m1.rows; i++){
			for(int j=0; j<m2.columns; j++){
				if(m1[i][j] != m2[i][j])
					return false;
			}
		}

		return true;
	}

	friend ostream& operator <<(ostream& os, const Matrix &m){
		os << m.rows << " " << m.columns;
		for(int i=0; i<m.rows; i++){
			os << endl;
			for(int j=0; j<m.columns; j++){
					os << m[i][j] << " ";
			}
		}
		return os;
	}

	friend istream& operator >>(istream& is, Matrix &m){
		is >> m.rows >> m.columns;
		m.arr.resize(m.rows, vector<double>(m.columns));

		for(int i=0; i<m.rows; i++){
			for(int j=0; j<m.columns; j++){
				is >> m[i][j];
			}
		}

		return is;
	}

private:
	class MatrixRow{
		Matrix &m;
		int currentRow;
	public:
		MatrixRow(Matrix &_m, int _currentRow) : m(_m), currentRow(_currentRow){}

		double& operator[](int col){
			if(col < 0 || col >= m.columns)
				throw exception();

			return m.arr[currentRow][col];
		}
	};

	class MatrixRowConst{
		const Matrix &m;
		int currentRow;
	public:
		MatrixRowConst(const Matrix &_m, int _currentRow) : m(_m), currentRow(_currentRow){
		}

		const double& operator[](int col) const{
			if(col < 0 || col >= m.columns)
				throw exception();

			return m.arr[currentRow][col];
		}
	};
};

#endif /* MATRIX_H_ */

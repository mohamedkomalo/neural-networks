/*
 * NeuralNetwork.h
 *
 *  Created on: May 1, 2014
 *      Author: Mohamed Kamal
 */

#include "Matrix.h"
#include "Vector.h"

#include <vector>
#include <climits>
using namespace std;

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

// TODO: make as AbstractBackprogationNeuralNetwork and inherit a matrix implementation and normal loops implementation
// TODO: enabling changing layer neurons in NN
// TODO: variable number of hidden layers

// TODO: add exceptions for wrong network parameters

class NeuralNetwork{
	double learningRate;


	Matrix weightsHiddenLayer;			// Matrix represents the weights of the connections
										// from the hidden layer (rows) to the input layer (cols) (not the reverse)

	Matrix weightsOutputLayer;			// Matrix represents the weights of the connections
										// from the output layer (rows) to the hidden layer (cols) (not the reverse)

	Function ActivationFunction;
	Function ActivationFunctionDerivative;

public:

	// TODO: add initial weights matrices constructor

	NeuralNetwork(int _inputNeurons,
				  int _hiddenNeurons,
				  int _outputNeurons,
				  double _learningRate,
				  Function _ActivationFunction,
				  Function _ActivationFunctionDerivative) : learningRate(_learningRate),
		  	  	  	 	 	 	 	 	 	 	 	 weightsHiddenLayer(Matrix(_hiddenNeurons, _inputNeurons)),
										  	  	  	 weightsOutputLayer(Matrix(_outputNeurons, _hiddenNeurons)),
										  	  	  	 ActivationFunction(_ActivationFunction),
										  	  	  	 ActivationFunctionDerivative(_ActivationFunctionDerivative){
	}

	int getInputNeuronsNo() const{
		return weightsHiddenLayer.getColumns();
	}

	int getHiddenNeuronsNo() const{
		return weightsHiddenLayer.getRows();
	}

	int getOutputNeuronsNo() const{
		return weightsOutputLayer.getRows();
	}

	Matrix feedforward(const Matrix &input) const{
		// Calculate the input signal to hidden layer that come from the input layer
		Matrix hiddenInputSignal = weightsHiddenLayer * input;

		// Calculate output signal from the hidden layer
		Matrix hiddenOutputSignal = hiddenInputSignal.applyFunction(ActivationFunction);

		// Calculate input signal for output layer
		// We will use the output signal of the hidden layer as an input signal to the output layer
		Matrix outputInputSignal = weightsOutputLayer * hiddenOutputSignal;

		// Calculate the output signal from the out layer
		// which is the output of the NN
		Matrix Output = outputInputSignal.applyFunction(ActivationFunction);

		return Output;
	}

	/**
	 * Note 1:
	 * The range of the activitation function must be taken in consideration seriously
	 * For example, the range of the Sigmoind function is between [0, 1]
	 * So the output signal of each neuron will be a number between the range of [0,1]
	 * So for input/output numbers larger than the range of the activation function, some
	 * kind of scaling must be done, but I don't know what is it =D
	 *
	 * Note 2:
	 * bias is not used in the algorithm yet.
	 *
	 * TODO:try to apply feedforward for all trainig data and compute the output error vector of each
	 * and compute an avarege output error vector, then propagate with this vector.
	 *
	 * TODO: make the input as a training pair and define a struct or class for it
	 */

	void train(const vector<ColumnVector> &allInput, const vector<ColumnVector> &allOutput){
		int inputNeurons = getInputNeuronsNo();
		int hiddenNeurons = getHiddenNeuronsNo();
		int outputNeurons = getOutputNeuronsNo();

		/**
		 * The randomizer used inside will generate the same random
		 * weights in each time the function fillRandomly is called
		 * This is due to the nature of the algorithim
		 * You can force the randomizer to generate different weights
		 * by the Randomizer variable in fillRandomly "static"
		 * But after testing, I found out that this slows the learning process
		 */
		weightsOutputLayer.fillRandomly(-0.5, 0.5);
		weightsHiddenLayer.fillRandomly(-0.5, 0.5);

		const int loopsBoundary = 3000;
		const double errorTarget = 0.01;

		int loops = 0;
		double error = 0;

		do{
			error = 0;

			for(unsigned int i=0; i<allInput.size(); i++){
				const Matrix &input = allInput[i];
				const Matrix &expectedOutput = allOutput[i];

				// Feedforward
				// Calculate the input signal to hidden layer that come from the input layer
				Matrix hiddenInputSignal = weightsHiddenLayer * input;

				// Calculate output signal from the hidden layer
				Matrix hiddenOutputSignal = hiddenInputSignal.applyFunction(ActivationFunction);

				// Calculate input signal for output layer
				// We will use the output signal of the hidden layer as an input to the output layer
				// but the input from each hidden neuron will be multiplied by the weights of the
				// connection to the output neuron
				Matrix outputInputSignal = weightsOutputLayer * hiddenOutputSignal;

				// Calculate the output signal from the output layer
				// which is the output of the Neural Network
				Matrix ActualOutput = outputInputSignal.applyFunction(ActivationFunction);

				// Add the error to monitor if we reached the target error
				error += expectedOutput.calculateMeanSquareErrorWith(ActualOutput);

				// Back-propagation
				// Compute the error for the output layer
				Matrix errorOutput = (expectedOutput - ActualOutput);
				errorOutput = errorOutput.elementWiseProduct(outputInputSignal.applyFunction(ActivationFunctionDerivative));

				// Skipping the weight update of the output layer to later.

				/* Calculate the error for the hidden layer
				 * Error output is a column vector with dimension (outputNeurons x 1), we transpose it to be (1 x outputNeurons)
				 * The weightsOutputLayer has a dimension of (outputNeurons x hiddenNeurons)
				 * So the result will be a row vector
				 *  	(1 x outputNeurons) * (outputNeurons x hiddenNeurons) = (1xhiddenNeurons)
				 *  transposing again to be a column vector because all the operations are done
				 *  with assumption of column vectors.
				 *  What we are doing is that we propagating the error of the output
				 *  neurons to hidden neurons based on the weight of each connection between them.
				 */
				Matrix errorHiddenPropagated = (errorOutput.transpose() * weightsOutputLayer).transpose();
				Matrix errorDerivative = hiddenInputSignal.applyFunction(ActivationFunctionDerivative);
				Matrix errorHidden = errorHiddenPropagated.elementWiseProduct(errorDerivative);

				// calculate the weights of the next iteration

				// for the output layer
				for(int k=0; k<outputNeurons; k++){
					for(int j=0; j<hiddenNeurons; j++){
						weightsOutputLayer[k][j] = weightsOutputLayer[k][j] + learningRate * errorOutput[k][0] * hiddenOutputSignal[j][0];
					}
				}

				// for hidden layer
				for(int k=0; k<hiddenNeurons; k++){
					for(int j=0; j<inputNeurons; j++){
						weightsHiddenLayer[k][j] = weightsHiddenLayer[k][j] + learningRate * errorHidden[k][0] * input[j][0];
					}
				}
			}

		// Cout how much loops we did so far
		int finishedLoops = loops*100/loopsBoundary;
		system("cls");
		cout << "Loops so far: " << loops << ", % of Loops finished: " << finishedLoops << "%, Error=" << error << endl;

		}while(error > errorTarget && loops++ < loopsBoundary);

		cout << "Completed learning with respect to the target error or the loops boundary\n"
			 << "Check the previous line to see the error achieved and no of loops done.\n";
	}


	friend ostream& operator <<(ostream& os, const NeuralNetwork &neuralNetwork){

		os << neuralNetwork.learningRate << endl
		   << neuralNetwork.weightsHiddenLayer << endl
		   << neuralNetwork.weightsOutputLayer << endl;

		return os;
	}

	friend istream& operator >>(istream& is, NeuralNetwork &neuralNetwork){

		is >> neuralNetwork.learningRate
		   >> neuralNetwork.weightsHiddenLayer
		   >> neuralNetwork.weightsOutputLayer;

		return is;
	}
};

#endif /* NEURALNETWORK_H_ */

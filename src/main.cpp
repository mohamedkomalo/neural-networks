#include "NeuralNetwork.h"
#include "Vector.h"

#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

// Serious error may be done here by making the numerator 1 not 1.0
// This will cause the division to return an integer and fail the NN learning process.
// However in the denumerator, 1.0 was chosen instead of 1 to avoid
// implicit conversion from int to double, but it won't affect the result
// TODO: check if there will be an implicit conversion from float to double
double Sigmoind(double x){
	return 1.0 / ( 1.0 + exp(-x) );
}

double SigmoindDerivative(double x){
	return Sigmoind(x) * (1.0 - Sigmoind(x));
}

double BipolarSigmoind(double x){
	return 2.0 / ( 1.0 + exp(-x) ) - 1;
}

double BipolarSigmoindDerivative(double x){
	return 0.5 * (1.0 + BipolarSigmoind(x)) * (1.0 - BipolarSigmoind(x));
}

void train_and_save(string trainfilename, string networkfilename){
	ifstream in(trainfilename.c_str());
	if(!in){
		cout << "Error, " << trainfilename << " file not found" << endl;
		exit(0);
	}

	int inputNeurons, hiddenNeurons, outputNeurons, noOfTrainings;
	in >> inputNeurons >> hiddenNeurons >> outputNeurons >> noOfTrainings;

	vector<ColumnVector> allX(noOfTrainings, ColumnVector(inputNeurons));
	vector<ColumnVector> allY(noOfTrainings, ColumnVector(outputNeurons));

	for(int i=0; i<noOfTrainings; i++){
		for(int j=0; j<inputNeurons; j++)
			in >> allX[i][j];


		for(int j=0; j<outputNeurons; j++)
			in >> allY[i][j];

	}
	in.close();

	NeuralNetwork neuralNetwork(inputNeurons, hiddenNeurons, outputNeurons,
								0.7,
								Sigmoind, SigmoindDerivative);

	neuralNetwork.train(allX, allY);

	ofstream we(networkfilename.c_str());
	we << neuralNetwork;
	we.close();
}

void load_and_test(string testsfilename, string networkfilename){
	ifstream networkfile(networkfilename.c_str());
	if(!networkfile){
		cout << "Error, " << networkfilename << " file not found" << endl;
		exit(0);
	}

	NeuralNetwork neuralNetwork(0, 0, 0, 0, Sigmoind, BipolarSigmoindDerivative);
	networkfile >> neuralNetwork;
	networkfile.close();

	int noOfTests;
	ifstream testsfile(testsfilename.c_str());

	if(!testsfile){
		cout << "Error, " << testsfilename << " file not found" << endl;
		exit(0);
	}

	testsfile >> noOfTests;

	double totalMSE = 0;

	for(int i=0; i<noOfTests; i++){
		Matrix input(neuralNetwork.getInputNeuronsNo(), 1);
		Matrix output(neuralNetwork.getOutputNeuronsNo(), 1);

		for(int j=0; j<neuralNetwork.getInputNeuronsNo(); j++){
			testsfile >> input[j][0];
		}

		for(int j=0; j<neuralNetwork.getOutputNeuronsNo(); j++){
			testsfile >> output[j][0];
		}

		Matrix actualOutput = neuralNetwork.feedforward(input);

		// Special case for the letter recognition assignment.
		Matrix roundedOutput = actualOutput.applyFunction(round);
		double error = roundedOutput.calculateMeanSquareErrorWith(output);

		cout << "Test Case " << i+1 << ": (Error=" << error << ")\n" <<  roundedOutput << endl;

		totalMSE += error;
	}

	totalMSE /= noOfTests;

	cout << "Mean Square Error of All: " << totalMSE << endl;

}

int main(){
	// The test cases are from the same training set

	short op;
	cout << "What do you want to do?\n"
		 << "1. train the NN using train.txt and save the weights to weights.txt\n"
		 << "2. load the NN using weights.txt and test the NN using test.txt file\n"
		 << "Enter option: ";
	cin >> op;

	if(op == 1)
		train_and_save("train.txt", "NeuralNetwork.txt");
	else if(op == 2)
		load_and_test("test.txt", "NeuralNetwork.txt");

	cout << "\nDone, exiting now\n";

	return 0;
}

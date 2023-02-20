#ifndef MYNEURO_H
#define MYNEURO_H

#include <iostream>
#include <vector>
#include <array>

namespace NN {

	struct Example {
		std::vector<float> input;
		std::vector<float> expectedOutput;
	};

	struct QueryResult {
		float avgError;
		std::vector<float> output;
	};

	class myNeuro {
	public:
		myNeuro(std::vector<int> layers);

		float train(const Example& trainExample);
		QueryResult query(const Example& testExample);

	private:
		struct nnLay;
		std::vector<nnLay> layers;

		void feedForward(const std::vector<float>& input);
		void backPropagate(const std::vector<float>& input, const std::vector<float>& expectedOutput);
	};

	struct myNeuro::nnLay {
		int in;
		int out;
		float** matrix;
		float* hidden;
		float* errors;
		int getInCount() { return in; }
		int getOutCount() { return out; }
		float** getMatrix() { return matrix; }

		void updMatrix(const float* enteredVal);
		void setIO(int inputs, int outputs);
		void makeHidden(const float* inputs);
		float* getHidden();
		void calcOutError(const float* targets);
		void calcHidError(const float* targets, float** outWeights, int inS, int outS);
		float* getErrors();
		float sigmoida(float val);
		float sigmoidasDerivate(float val);

	};

}

#endif // MYNEURO_H#pragma once

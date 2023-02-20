#include "myNeuro.h"

#include <cmath>

#define learnRate 0.1 
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))

using namespace std;

NN::myNeuro::myNeuro(std::vector<int> layersInfo) {
	layers.resize(layersInfo.size() - 1);

	// слои
	for (auto i = 0; i < layersInfo.size() - 1; i++) {
		layers[i].setIO(layersInfo[i], layersInfo[i + 1]);
	}
}

// прямой проход по слоям
void NN::myNeuro::feedForward(const std::vector<float>& input) {
	layers.front().makeHidden(input.data());
	for (int i = 1; i < layers.size(); i++) {
		layers[i].makeHidden(layers[i - 1].getHidden());
	}
}

void NN::myNeuro::backPropagate(const std::vector<float>& input, const std::vector<float>& expectedOutput) {
	
	layers.back().calcOutError(expectedOutput.data());
	for (int i = layers.size() - 2; i >= 0; i--) {
		layers[i].calcHidError(layers[i + 1].getErrors(), layers[i + 1].getMatrix(),
			layers[i + 1].getInCount(), layers[i + 1].getOutCount());
	}
	for (int i = layers.size() - 1; i > 0; i--)
		layers[i].updMatrix(layers[i - 1].getHidden());
	layers.front().updMatrix(input.data());
}

float NN::myNeuro::train(const Example& trainExample) {
	feedForward(trainExample.input);
	backPropagate(trainExample.input, trainExample.expectedOutput);

	layers.back().calcOutError(trainExample.expectedOutput.data());
	float Err_sum = 0;
	for (int i = 0; i < layers.size() - 1; i++) {

		Err_sum += std::abs(layers.back().errors[i]);
	}
	Err_sum /= layers.back().getOutCount();
	return Err_sum;
}

//трестирование
NN::QueryResult NN::myNeuro::query(const Example& testExample) {
	feedForward(testExample.input);
	auto& output = layers.back();
	float err = 0;
	for (int i = 0; i < testExample.expectedOutput.size(); i++) {
		err += std::abs(testExample.expectedOutput[i] - output.getHidden()[i]);
	}
	return {
		.avgError = err / testExample.expectedOutput.size(),
		.output = std::vector<float>(output.getHidden(), output.getHidden() + output.getOutCount())
	};
}

// матрица весов
void NN::myNeuro::nnLay::updMatrix(const float* enteredVal)
{
	for (int ou = 0; ou < out; ou++)
	{

		for (int hid = 0; hid < in; hid++)
		{
			matrix[hid][ou] += (learnRate * errors[ou] * enteredVal[hid]);
		}
		matrix[in][ou] += (learnRate * errors[ou]);
	}
};

void NN::myNeuro::nnLay::setIO(int inputs, int outputs)
{
	in = inputs;
	out = outputs;
	hidden = (float*)malloc((out) * sizeof(float));
	errors = (float*)malloc((out) * sizeof(float));

	matrix = (float**)malloc((in + 1) * sizeof(float*));
	for (int inp = 0; inp < in + 1; inp++)
	{
		matrix[inp] = (float*)malloc(out * sizeof(float));
	}
	for (int inp = 0; inp < in + 1; inp++)
	{
		for (int outp = 0; outp < out; outp++)
		{
			matrix[inp][outp] = randWeight;
		}
	}
};

void NN::myNeuro::nnLay::makeHidden(const float* inputs)
{
	for (int hid = 0; hid < out; hid++)
	{
		float tmpS = 0.0;
		for (int inp = 0; inp < in; inp++)
		{
			tmpS += inputs[inp] * matrix[inp][hid];
		}
		tmpS += matrix[in][hid];
		hidden[hid] = sigmoida(tmpS);
	}
};

float* NN::myNeuro::nnLay::getHidden()
{
	return hidden;
};

void NN::myNeuro::nnLay::calcOutError(const float* targets)
{
	errors = (float*)malloc((out) * sizeof(float));
	for (int ou = 0; ou < out; ou++)
	{
		errors[ou] = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);
		//cout << "errors_o: " << errors[ou] << endl;
	}

};

void NN::myNeuro::nnLay::calcHidError(const float* targets, float** outWeights, int inS, int outS)
{
	errors = (float*)malloc((inS) * sizeof(float));
	for (int hid = 0; hid < inS; hid++)
	{
		errors[hid] = 0.0;
		for (int ou = 0; ou < outS; ou++)
		{
			errors[hid] += targets[ou] * outWeights[hid][ou];
		}
		errors[hid] *= sigmoidasDerivate(hidden[hid]);

	}
};

float* NN::myNeuro::nnLay::getErrors()
{
	return errors;
};
float NN::myNeuro::nnLay::sigmoida(float val)
{
	return (1.0 / (1.0 + exp(-val)));
}
float NN::myNeuro::nnLay::sigmoidasDerivate(float val)
{
	return (val * (1.0 - val));
};

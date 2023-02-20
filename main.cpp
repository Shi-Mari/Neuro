#include <iostream>
#include <memory>
#include <algorithm>
#include <exception>
#include <fstream>
#include <numeric>
#include <random>
#include <iomanip>

#include "myNeuro.h"

#define EPS 0.001  // ошибка до которой тренероуем 

using Dataset = std::vector<NN::Example>;

std::shared_ptr<Dataset> loadDataset_mnist(const std::string& dataPath, const std::string& labelsPath, int categoriesCount);


// перегрузка оператора << дл€ вывода вектора в консоль
std::ostream& operator<< (std::ostream& out, const std::vector<float>& v) {
	out << std::setprecision(4) << std::fixed;
	for (size_t i = 0; i < v.size(); ++i) {
		out << v[i] << " ";
	}
	return out;
}

int main() {

	NN::myNeuro* bb = new NN::myNeuro({ 28 * 28, 128, 64, 10 });

	auto trainSet = loadDataset_mnist("dataset\\emnist-digits-train-images-idx3-ubyte", "dataset\\emnist-digits-train-labels-idx1-ubyte", 10);
	auto testSet = loadDataset_mnist("dataset\\emnist-digits-test-images-idx3-ubyte", "dataset\\emnist-digits-test-labels-idx1-ubyte", 10);

	std::vector<size_t> trainingSetAccessOrder(1000);
	std::iota(trainingSetAccessOrder.begin(), trainingSetAccessOrder.end(), 0); // заполн€етс€ последовательно числами от 0 до 999
	std::random_device rd;
	std::mt19937 gen(rd());

	
	int epochNum = 1;
	float err = 1;
	do {
		err = 0;
		std::shuffle(trainingSetAccessOrder.begin(), trainingSetAccessOrder.end(), gen); // перемешивание
		for (auto i : trainingSetAccessOrder) {
			err += bb->train((*trainSet)[i]);  // бирЄм картинку с рандомным номером и отправл€ем на тренировку
		}

		// считаем и выводим ошибку, преращение эпохи
		err /= trainingSetAccessOrder.size();
		std::cout << "Epoch " << epochNum << ": " << err << std::endl;
		epochNum++;
	} while (err > EPS);

	std::cout << "___________________RESULT_____________" << std::endl;


	//тестирование
	NN::QueryResult res;
	float avgErr = 0;
	for (auto& example : *(testSet)) { // проходит по всем элементам testSet и отправл€ет на обучение
		res = bb->query(example);
		avgErr += res.avgError;  
	}
	avgErr /= testSet->size(); // средн€€ ошибка
	std::cout << "Average Error: " << avgErr << std::endl << std::endl; // вывод ошибки
	std::cout << "Some examples:" << std::endl;
	std::uniform_int_distribution<> distrib(0, testSet->size() - 1); // рандомное распределение
	for (int num = 0; num < 10; num++) {  // 10 тестовых примеров
		auto i = distrib(gen);    // достаЄм тестовую картинку по рандомному значению и отправл€етс€ на тест
		auto& test = (*trainSet)[i];
		res = bb->query(test);
		auto guessed = std::distance(res.output.begin(), std::max_element(res.output.begin(), res.output.end()));  // находим максимальное элемент в том что вернула нейронка, находим номер элемента
		auto answer = std::distance(test.expectedOutput.begin(), std::max_element(test.expectedOutput.begin(), test.expectedOutput.end())); // то же самое дл€ ожидаемого
		std::cout << "example " << i << " : " << (guessed == answer ? "Correct" : "Error") << std::endl; // сравниваем с ожидаемыми значени€ми
		std::cout << "got     : " << res.output << std::endl;
		std::cout << "expected: " << test.expectedOutput << std::endl;
		std::cout << std::endl;
	}

	std::cout << "_______________THE____END_______________" << std::endl;

	return 0;
}

const float normalizationMap[256] = { 0.000000000f, 0.003921569f, 0.007843137f, 0.011764706f, 0.015686275f, 0.019607843f, 0.023529412f, 0.027450980f, 0.031372549f, 0.035294118f, 0.039215686f, 0.043137255f, 0.047058824f, 0.050980392f, 0.054901961f, 0.058823529f,
									  0.062745098f, 0.066666667f, 0.070588235f, 0.074509804f, 0.078431373f, 0.082352941f, 0.086274510f, 0.090196078f, 0.094117647f, 0.098039216f, 0.101960784f, 0.105882353f, 0.109803922f, 0.113725490f, 0.117647059f, 0.121568627f,
									  0.125490196f, 0.129411765f, 0.133333333f, 0.137254902f, 0.141176471f, 0.145098039f, 0.149019608f, 0.152941176f, 0.156862745f, 0.160784314f, 0.164705882f, 0.168627451f, 0.172549020f, 0.176470588f, 0.180392157f, 0.184313725f,
									  0.188235294f, 0.192156863f, 0.196078431f, 0.200000000f, 0.203921569f, 0.207843137f, 0.211764706f, 0.215686275f, 0.219607843f, 0.223529412f, 0.227450980f, 0.231372549f, 0.235294118f, 0.239215686f, 0.243137255f, 0.247058824f,
									  0.250980392f, 0.254901961f, 0.258823529f, 0.262745098f, 0.266666667f, 0.270588235f, 0.274509804f, 0.278431373f, 0.282352941f, 0.286274510f, 0.290196078f, 0.294117647f, 0.298039216f, 0.301960784f, 0.305882353f, 0.309803922f,
									  0.313725490f, 0.317647059f, 0.321568627f, 0.325490196f, 0.329411765f, 0.333333333f, 0.337254902f, 0.341176471f, 0.345098039f, 0.349019608f, 0.352941176f, 0.356862745f, 0.360784314f, 0.364705882f, 0.368627451f, 0.372549020f,
									  0.376470588f, 0.380392157f, 0.384313725f, 0.388235294f, 0.392156863f, 0.396078431f, 0.400000000f, 0.403921569f, 0.407843137f, 0.411764706f, 0.415686275f, 0.419607843f, 0.423529412f, 0.427450980f, 0.431372549f, 0.435294118f,
									  0.439215686f, 0.443137255f, 0.447058824f, 0.450980392f, 0.454901961f, 0.458823529f, 0.462745098f, 0.466666667f, 0.470588235f, 0.474509804f, 0.478431373f, 0.482352941f, 0.486274510f, 0.490196078f, 0.494117647f, 0.498039216f,
									  0.501960784f, 0.505882353f, 0.509803922f, 0.513725490f, 0.517647059f, 0.521568627f, 0.525490196f, 0.529411765f, 0.533333333f, 0.537254902f, 0.541176471f, 0.545098039f, 0.549019608f, 0.552941176f, 0.556862745f, 0.560784314f,
									  0.564705882f, 0.568627451f, 0.572549020f, 0.576470588f, 0.580392157f, 0.584313725f, 0.588235294f, 0.592156863f, 0.596078431f, 0.600000000f, 0.603921569f, 0.607843137f, 0.611764706f, 0.615686275f, 0.619607843f, 0.623529412f,
									  0.627450980f, 0.631372549f, 0.635294118f, 0.639215686f, 0.643137255f, 0.647058824f, 0.650980392f, 0.654901961f, 0.658823529f, 0.662745098f, 0.666666667f, 0.670588235f, 0.674509804f, 0.678431373f, 0.682352941f, 0.686274510f,
									  0.690196078f, 0.694117647f, 0.698039216f, 0.701960784f, 0.705882353f, 0.709803922f, 0.713725490f, 0.717647059f, 0.721568627f, 0.725490196f, 0.729411765f, 0.733333333f, 0.737254902f, 0.741176471f, 0.745098039f, 0.749019608f,
									  0.752941176f, 0.756862745f, 0.760784314f, 0.764705882f, 0.768627451f, 0.772549020f, 0.776470588f, 0.780392157f, 0.784313725f, 0.788235294f, 0.792156863f, 0.796078431f, 0.800000000f, 0.803921569f, 0.807843137f, 0.811764706f,
									  0.815686275f, 0.819607843f, 0.823529412f, 0.827450980f, 0.831372549f, 0.835294118f, 0.839215686f, 0.843137255f, 0.847058824f, 0.850980392f, 0.854901961f, 0.858823529f, 0.862745098f, 0.866666667f, 0.870588235f, 0.874509804f,
									  0.878431373f, 0.882352941f, 0.886274510f, 0.890196078f, 0.894117647f, 0.898039216f, 0.901960784f, 0.905882353f, 0.909803922f, 0.913725490f, 0.917647059f, 0.921568627f, 0.925490196f, 0.929411765f, 0.933333333f, 0.937254902f,
									  0.941176471f, 0.945098039f, 0.949019608f, 0.952941176f, 0.956862745f, 0.960784314f, 0.964705882f, 0.968627451f, 0.972549020f, 0.976470588f, 0.980392157f, 0.984313725f, 0.988235294f, 0.992156863f, 0.996078431f, 1.000000000f };

#define BYTESWAP_32(number) (((number) >> 24) | (((number) & 0x00FF0000) >> 8) | (((number) & 0x0000FF00) << 8) | ((number) << 24))

struct MnistDataHeader {
	uint32_t magic;
	uint32_t count;
	uint32_t rows;
	uint32_t cols;

	void swapBytes() {
		magic = BYTESWAP_32(magic);
		count = BYTESWAP_32(count);
		rows = BYTESWAP_32(rows);
		cols = BYTESWAP_32(cols);
	}
};

struct MnistLabelsHeader {
	uint32_t magic;
	uint32_t count;

	void swapBytes() {
		magic = BYTESWAP_32(magic);
		count = BYTESWAP_32(count);
	}
};


std::shared_ptr<Dataset> loadDataset_mnist(const std::string& dataPath, const std::string& labelsPath, int categoriesCount) {
	std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>();


	// Open file в бинарном виде
	std::ifstream dataFile{ dataPath, std::ios::binary };
	if (!dataFile.is_open()) throw std::runtime_error("Can't open data file");
	std::ifstream labelsFile{ labelsPath, std::ios::binary };
	if (!labelsFile.is_open()) throw std::runtime_error("Can't open labels file");

	// Reading header information
	MnistDataHeader dataHeader;
	MnistLabelsHeader labelsHeader;
	dataFile.read((char*)&dataHeader, sizeof(MnistDataHeader));
	labelsFile.read((char*)&labelsHeader, sizeof(MnistLabelsHeader));
	if constexpr (std::endian::native == std::endian::little) {
		dataHeader.swapBytes();
		labelsHeader.swapBytes();
	}
	if (dataHeader.count != labelsHeader.count)
		throw std::runtime_error("Data and labels file have different number of elements");

	// Reading dataset считываем картинки в виде байтов и сохран€ютс€ в массив
	dataset->resize(dataHeader.count); // расшир€етс€ до необходимого кол-ва элементов
	uint64_t inputCount = dataHeader.rows * dataHeader.cols;
	char* tmpImageData = new char[inputCount];
	char tmpLabelData = 0;
	for (uint32_t i = 0; i < dataHeader.count; i++) {
		dataFile.read(tmpImageData, inputCount);
		labelsFile.read(&tmpLabelData, 1);
		if (!dataFile.good() || !labelsFile.good())
			throw std::runtime_error("Unexpected error when reading.");

		(*dataset)[i].input.resize(inputCount);
		std::transform(tmpImageData, tmpImageData + inputCount, (*dataset)[i].input.data(), [](uint8_t in) {return normalizationMap[in]; }); // 4 парам. л€мбда функци€ преобразует число от 0 до 255, результат сохран€етс€ в вектор input
		
		//ожидаемый выход
		(*dataset)[i].expectedOutput.resize(categoriesCount); 
		//(*dataset)[i].expectedOutput.setZero();
		(*dataset)[i].expectedOutput[tmpLabelData] = 1;
	}

	dataFile.close();
	labelsFile.close();

	return dataset;
}

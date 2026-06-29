#include <stdio.h>
#include "ConvLayer.cpp"
#include "ReLU.cpp"
#include "Tensor.h"
#include "MaxPooling.cpp"
#include "FullyConnected.cpp"
#include "Loss.cpp"

using namespace std;

class Network {
private:
	// Свёрточные слои
	ConvolutionalLayer conv1;
	ReLU relu1;
	MaxPooling pool1;

	ConvolutionalLayer conv2;
	ReLU relu2;
	MaxPooling pool2;

	ConvolutionalLayer conv3;
	ReLU relu3;
	MaxPooling pool3;
	// Слои полносвязной сети
	FullyConnected fc1;
	ReLU fc_relu;
	FullyConnected fc2;

public:
	Network() {}
	Tensor forward(Tensor& input) {
		Tensor out;
		// Первый свёрточный слой
		out = conv1.forward(input);
		out = relu1.forward(out);
		out = pool1.forward(out);
		// Второй свёрточный слой
		out = conv2.forward(out);
		out = relu2.forward(out);
		out = pool2.forward(out);
		// Третий свёрточный слой
		out = conv3.forward(out);            
		out = relu3.forward(out);
		out = pool3.forward(out);
		// Результат в вектор flatten
		vector<float>& flat_data = out.getData();
		int flat_size = flat_data.size();
		Tensor flat = out.reshape({ 1, flat_size });
		// Полносвязная сеть
		out = fc1.forward(flat);
		out = fc_relu.forward(out);
		Tensor output = fc2.forward(out);

		return output;
	}
	void backward(Tensor& grad_output) {
		Tensor grad = grad_output;
		grad = fc2.backward(grad);
		grad = fc_relu.backward(grad);
		grad = fc1.backward(grad);

		Tensor grad_flat = grad.reshape({ 1, 64, 28, 28 });

	

	}
};
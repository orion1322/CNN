#include <stdio.h>
#include "Tensor.h"

using namespace std;

class FullyConnected {
private:
	Tensor weights;
	Tensor bias;
	// Параметры градиента
	Tensor grad_weights;
	Tensor grad_bias;
	Tensor cache;

	int in_features;
	int out_features;

public:
	FullyConnected(){}
	FullyConnected(int in_feature, int out_feature) {
		in_features = in_feature;
		out_features = out_feature;
		grad_weights = Tensor({ out_features, in_features });
		grad_bias = Tensor({ out_features });
		weights = Tensor({out_feature, in_feature});
		vector<float>& w_data = weights.getData();

		float scale = sqrt(2 / in_features); // Xavier
		for (int i = 0; i < w_data.size(); i++) {
			w_data[i] = ((float)rand() / RAND_MAX) * scale*2 - scale;
		}

		bias = Tensor({ out_features });
		vector<float>& b_data = bias.getData();
		for (int i = 0; i < b_data.size(); i++) {
			b_data[i] = 0;
		}
	}

	Tensor forward(Tensor& tensor) {
		cache = tensor;
		vector<float>& flat_rows = tensor.getData();
		int count_rows = flat_rows.size();
		Tensor flat = tensor.reshape({ 1, count_rows });
		Tensor flat_T = flat.transpose();
		Tensor raw = matMul(weights, flat_T);
		Tensor output = raw.transpose();
		
		// Добавляем смещение bias
		vector<float>& out_data = output.getData();
		vector<float>& b_data = bias.getData();
		for (int i = 0; i < out_data.size(); i++) {
			out_data[i] += b_data[i];
		}
		return output;
	}

	Tensor backward(Tensor& grad_output) {
		vector<float>& grad_data = grad_output.getData();
		vector<float>& grad_bias_data = grad_bias.getData();
		for (int i = 0; i < out_features; i++) { // Градиент по смещению bias
			grad_bias_data[i] = grad_data[i]; // += накапливаем градиент | = перезаписываем
		}
		// Градиент для весов dw = grad_output^T * input
		vector<float> cache_shape = cache.getData();
		int lenght = cache_shape.size();
		Tensor flat_input = cache.reshape({ 1, lenght });
		Tensor grad_out_T = grad_output.transpose();
		Tensor grad_w = matMul(grad_out_T, flat_input);
		vector<float>& grad_w_data = grad_w.getData();
		vector<float>& grad_weights_data = grad_weights.getData();
		for (int i = 0; i < grad_w_data.size(); i++) {
			grad_weights_data[i] = grad_w_data[i]; // += накапливаем градиент | = перезаписываем
		}
		// Градиаент для входа dInput = weights^T * grad_output^T
		Tensor weights_T = weights.transpose();
		Tensor grad_input = matMul(weights_T, grad_out_T);
		grad_input = grad_input.transpose();

		return grad_input.reshape(cache.getShape());// Возвращаем градиент предыдущего слоя
	}
	void update(float rate) {
		// Обновление весов w = w - rate * dw
		vector<float>& w_data = weights.getData();
		vector<float>& grad_w_data = grad_weights.getData();
		for (int i = 0; i < w_data.size(); i++) {
			w_data[i] -= rate * grad_w_data[i];
		}
		// Обновление смещения bias
		vector<float>& b_data = bias.getData();
		vector<float>& grad_b_data = grad_bias.getData();
		for (int i = 0; i < b_data.size(); i++) {
			b_data[i] -= rate * grad_b_data[i];
		}
	}
	void zeroGrad() {
		vector<float>& grad_w_data = grad_weights.getData();
		for (int i = 0; i < grad_w_data.size(); i++) {
			grad_w_data[i] = 0;
		}

		vector<float>& grad_b_data = grad_bias.getData();
		for (int i = 0; i < grad_b_data.size(); i++) {
			grad_b_data[i] = 0;
		}
	}
};
#include <stdio.h>
#include "Tensor.h"


using namespace std;

class MaxPooling {
private:
	int pool_size = 2;
	int stride = 2;
	// Параметры для градиента
	Tensor cache;
	vector<int> max_idx;

public:
	MaxPooling(){}
	Tensor forward(Tensor& tensor) {
		cache = tensor;
		max_idx.clear();

		vector<int> shape_tensor = tensor.getShape();
		int batch = shape_tensor[0];
		int channels = shape_tensor[1];
		int height = shape_tensor[2];
		int width = shape_tensor[3];

		int out_h = height / stride;
		int out_w = width / stride;

		Tensor output({batch, channels, out_h, out_w});
		vector<float>& out_data = output.getData();
		vector<float>& in_data = tensor.getData();
		for (int b = 0; b < batch; b++) { // Пробегаем по тензору пулинговой сеткой
			for (int c = 0; c < channels; c++) {
				for (int h = 0; h < out_h; h++) {
					for (int w = 0; w < out_w; w++) {
						float max_value = -100; // В пулинговой сетке находим максимальное
						int max_h = 0;
						int max_w = 0;
						for (int ph = 0; ph < pool_size; ph++) {
							for (int pw = 0; pw < pool_size; pw++) {
								int h_in = h * stride + ph;
								int w_in = w * stride + pw;
								int index = ((b * channels + c) * height + h_in) * width + w_in;
								float value = in_data[index];
								if (value > max_value) {
									max_value = value;
									max_h = ph;
									max_w = pw;
								}
							}
						}
						int out_index = ((b * channels + c) * out_h + h) * out_w + w;
						out_data[out_index] = max_value;
						max_idx.push_back(max_h);
						max_idx.push_back(max_w);
					}
				}
			}
		}
		return output;
	}
	Tensor backward(Tensor& grad_output) {
		vector<int>& cache_shape = cache.getShape();
		Tensor grad_input = Tensor(cache_shape);
		vector<int> grad_shape = grad_output.getShape();
		int batch = grad_shape[0];
		int channels = grad_shape[1];
		int out_h = grad_shape[2];
		int out_w = grad_shape[3];
		int height = cache_shape[2];
		int width = cache_shape[3];

		vector<float>& grad_in_data = grad_input.getData();
		vector<float>& grad_out_data = grad_output.getData();
		int idx = 0;
		for (int b = 0; b < batch; b++) {
			for (int c = 0; c < channels; c++) {
				for (int h = 0; h < out_h; h++) {
					for (int w = 0; w < out_w; w++) {
						int max_h = max_idx[idx * 2];
						int max_w = max_idx[idx * 2 + 1];

						int h_in = h * stride + max_h;
						int w_in = w * stride + max_w;

						grad_in_data[((b * channels + c) * height + h_in) * width + w_in] = grad_out_data[((b * channels + c) * out_h + h) * out_w + w];
						idx++;
					}
				}
			}
		}
		return grad_input;
	}
};
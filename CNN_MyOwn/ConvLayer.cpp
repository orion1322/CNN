#include <stdio.h>
#include "Tensor.h"

using namespace std;

class ConvolutionalLayer {
private:
	Tensor weights_kernel;
	Tensor bias;
	int size_kernel;
	int stride;
	int padding;

	// Параметры градиента
	Tensor grad_weights;
	Tensor grad_bias;
	Tensor cache;
	Tensor col_cache;


public:
	ConvolutionalLayer(){}
	ConvolutionalLayer(int RGB_channels, int counts_filters, int kernel_size, int in_stride, int in_padding) {
		size_kernel = kernel_size;
		stride = in_stride;
		padding = in_padding;
		weights_kernel = Tensor({ counts_filters, RGB_channels, kernel_size, kernel_size });
		vector<float>& weights_constr = weights_kernel.getData();
		float scale = sqrt(2 / (RGB_channels * kernel_size * kernel_size)); //Xavier
		for (int i = 0; i < weights_constr.size(); i++) {
			weights_constr[i] = ((float)rand() / RAND_MAX) * scale * 2 - scale;
		}

		bias = Tensor({ counts_filters });
		vector<float>& b_data = bias.getData();
		for (int i = 0; i < b_data.size(); i++) {
			b_data[i] = 0;
		}

		grad_weights = Tensor({ counts_filters, RGB_channels, kernel_size, kernel_size });
		grad_bias = Tensor({ counts_filters });
	}
	Tensor forward(Tensor& tensor) {
		cache = tensor;
		Tensor mat = imgToCol(tensor, size_kernel, stride, padding); // Преобразуем изображение в матрицу
		col_cache = mat;

		vector<int> w_shape = weights_kernel.getShape();
		Tensor weights_2d = weights_kernel.reshape({ w_shape[0], w_shape[1] * w_shape[2] * w_shape[3] });
		Tensor output_mat = matMul(weights_2d, mat); // Умножаем фильтр на матрицу изображения

		// Добавляем смещение bias
		vector<int>& shape_mat = output_mat.getShape();
		vector<float>& data_mat = output_mat.getData();
		vector<float>& bias_data = bias.getData();
		for (int i = 0; i < shape_mat[1]; i++) {
			for (int c = 0; c < shape_mat[0]; c++) {
				data_mat[c * shape_mat[1] + i] += bias_data[c];
			}
		}

		// Получаем размеры входа
		vector<int> tensorShape = tensor.getShape();
		int batch = tensorShape[0];
		int counts_filters = w_shape[0];
		int out_h = (tensorShape[2] + 2 * padding - size_kernel) / stride + 1;
		int out_w = (tensorShape[3] + 2 * padding - size_kernel) / stride + 1;

		// Получаем выходные тензоры
		Tensor output({batch, counts_filters, out_h, out_w});
		vector<float>& out_data = output.getData();
		for (int b = 0; b < batch; b++) {
			for (int c = 0; c < counts_filters; c++) {
				for (int h = 0; h < out_h; h++) {
					for (int w = 0; w < out_w; w++) {
						int col_index = (b * out_h + h) * out_w + w;
						int out_index = (b * counts_filters + c) * out_h * out_w + out_w * h + w;
						out_data[out_index] = data_mat[c * (out_h * out_w * batch) + col_index];
					}
				}
			}
		}
		return output;
	}
	Tensor backward(Tensor& grad_output) {
		vector<int>& grad_shape = grad_output.getShape();
		int batch = grad_shape[0];
		int out_channels = grad_shape[1];
		int out_h = grad_shape[2];
		int out_w = grad_shape[3];
		vector<float>& grad_bias_data = grad_bias.getData();
		vector<float>& grad_data = grad_output.getData();
		for (int c = 0; c < out_channels; c++) {
			float sum = 0;
			for (int b = 0; b < batch; b++) {
				for (int h = 0; h < out_h; h++) {
					for (int w = 0; w < out_w; w++) {
						sum += grad_data[((b * out_channels + c) * out_h + h) * out_w + w];
					}
				}
			}
			grad_bias_data[c] += sum;
		}
		Tensor grad_2d = grad_output.reshape({ out_channels, out_h * out_w * batch });
		// Градиент для весов dw = grad_output * col^T
		Tensor col_T = col_cache.transpose();
		Tensor grad_w = matMul(grad_2d, col_T);
		vector<int>& w_kernel_shape = weights_kernel.getShape();
		grad_w = grad_w.reshape({ out_channels, w_kernel_shape[1], size_kernel, size_kernel });

		vector<float>& grad_w_data = grad_w.getData();
		vector<float>& grad_weights_data = grad_weights.getData();
		for (int i = 0; i < grad_w_data.size(); i++) {
			grad_weights_data[i] += grad_w_data[i]; // += накопление градиентов | = перезаписывание градиентов
		}
		// Градиент для входа dInput = weights^T * grad_output
		Tensor weights_2d = weights_kernel.reshape({ out_channels, w_kernel_shape[1] * size_kernel * size_kernel });
		Tensor weights_T = weights_2d.transpose();
		Tensor grad_input_mat = matMul(weights_T, grad_2d);
		Tensor grad_input = colToImg(grad_input_mat, cache.shape, size_kernel, stride, padding);

		return grad_input;

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
	void update(float rate) {
		// Обновление весов w = w - rate * dw
		vector<float>& w_data = weights_kernel.getData();
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
};
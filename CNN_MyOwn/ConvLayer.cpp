#include <stdio.h>
#include "Tensor.cpp"

using namespace std;

class ConvolutionalLayer {
private:
	Tensor weights_kernel;
	int size_kernel;
	int stride;
	int padding;

public:
	ConvolutionalLayer(int RGB_channels, int counts_filters, int kernel_size, int in_stride, int in_padding) {
		size_kernel = kernel_size;
		stride = in_stride;
		padding = in_padding;
		weights_kernel = Tensor({ counts_filters, RGB_channels, kernel_size, kernel_size });
		vector<float>& weights_constr = weights_kernel.getData();
		for (int i = 0; i < weights_constr.size(); i++) {
			weights_constr[i] = rand() % 10;
		}
	}
	Tensor forward(Tensor& tensor) {
		Tensor mat = imgToCol(tensor, size_kernel, stride, padding); // Преобразуем изображение в матрицу

		vector<int> w_shape = weights_kernel.getShape();
		Tensor weights_2d = weights_kernel.reshape({ w_shape[0], w_shape[1] * w_shape[2] * w_shape[3] });
		Tensor output_mat = matMul(weights_2d, mat); // Умножаем фильтр на матрицу изображения

		// Получаем размеры входа
		vector<int> tensorShape = tensor.getShape();
		int batch = tensorShape[0];
		int counts_filters = w_shape[0];
		int out_h = (tensorShape[2] + 2 * padding - size_kernel) / stride + 1;
		int out_w = (tensorShape[3] + 2 * padding - size_kernel) / stride + 1;

		// Получаем выходные тензоры
		Tensor output({batch, counts_filters, out_h, out_w});
		vector<float>& outData = output.getData();
		vector<float>& outMatData = output_mat.getData();
		for (int b = 0; b < batch; b++) {
			for (int c = 0; c < counts_filters; c++) {
				for (int h = 0; h < out_h; h++) {
					for (int w = 0; w < out_w; w++) {
						int col_index = (b * out_h + h) * out_w + w;
						int out_index = (b * counts_filters + c) * out_h * out_w + out_w * h + w;
						outData[out_index] = outMatData[c * (out_h * out_w * batch) + col_index];
					}
				}
			}
		}
		return output;
	}
};
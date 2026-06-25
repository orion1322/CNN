#include <stdio.h>
#include "Tensor.cpp"

using namespace std;

class MaxPooling {
private:
	int pool_size = 2;
	int stride = 2;

public:
	Tensor forward(Tensor& tensor) {
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
						for (int ph = 0; ph < pool_size; ph++) {
							for (int wh = 0; wh < pool_size; wh++) {
								int h_in = h * stride + ph;
								int w_in = w * stride + wh;
								int index = ((b * channels + c) * height + h_in) * width + w_in;
								float value = in_data[index];
								if (value > max_value) {
									max_value = value;
								}
							}
						}
						int out_index = ((b * channels + c) * out_h + h) * out_w + w;
						out_data[out_index] = max_value;
					}
				}
			}
		}
		return output;
	}

};
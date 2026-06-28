#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include "Tensor.h"

using namespace std;

Tensor::Tensor() {
		shape = {0, 0, 0, 0};
		data = {};
	}
Tensor::Tensor(const vector<int>& new_shape) {
		shape = new_shape;
		int size = 1;
		for (int i = 0; i < shape.size(); i++) {
			size *= shape[i];
		}
		data.clear();
		for (int i = 0; i < size; i++) {
			data.push_back(rand()%10);
		}
	}

void Tensor::fillTestData() {
		float test_data[] = { 5, -2, 3, -1, 0, 8, -4, 6, -7, 2, -3, 9 };
		for (int i = 0; i < data.size() && i < 12; i++) {
			data[i] = test_data[i];
		}
	}
void Tensor::print() {
		cout << "\nРазмерность: [";
		for (int i = 0; i < shape.size(); i++) {
			cout << shape[i];
			if (i < shape.size() - 1) {
				cout << ", ";
			}
		}
		cout << "]";

		cout << "\nДанные: {";
		for (int i = 0; i < data.size(); i++) {
			cout << data[i] ;
			if (i < data.size() - 1) {
				cout << ", ";
			}
		}
		cout << "}\n";
	}
float Tensor::getValue(int batch, int channel, int height, int width) const {
		int index = ((batch * shape[1] + channel) * shape[2] + height) * shape[3] + width;
		if (data.size() > index && index >= 0) {
			//cout << "Индекс: " << index << "\n";
			return data[index];
		}
		else {
			throw runtime_error("Такого индекса нет!");
		}
	}
Tensor Tensor::reshape(const vector<int>& new_shape) {
		int old_size = 1;
		for (int i = 0; i < shape.size(); i++) {
			old_size *= shape[i];
		}
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) {
			new_size *= new_shape[i];
		}
		if (old_size != new_size) {
			throw runtime_error("Размерность не совпадает!");
		}
		return Tensor({new_shape});
	}
vector<int>& Tensor::getShape() {
		return shape;
	}
vector<float>& Tensor::getData() {
		return data;
	}
Tensor Tensor::transpose() {
	if (shape.size() != 2) {
		throw runtime_error("Транспонирование только для 2-мерных тензоров!");
	}
	else {
		int cols = shape[1];
		int rows = shape[0];
		Tensor result({ cols,rows });

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[j * rows + i] = data[i * cols + j];
			}
		}
		return result;
	}
}

Tensor imgToCol(const Tensor& tensor, int size_kernel, int stride, int padding) { // Доработать для разных размерностей тензора
	int batch = tensor.shape[0];
	int channel = tensor.shape[1];
	int height = tensor.shape[2];
	int width = tensor.shape[3];

	int count_vert = (height + 2 * padding - size_kernel) / stride + 1; // Кол-во проходов фильтра по вертикали изображения
	int count_hor = (width + 2 * padding - size_kernel) / stride + 1;   // Кол-во проходов фильтра по горизонтали изображения

	int rows = channel * size_kernel * size_kernel;
	int cols = count_vert * count_hor * batch;
	Tensor mat({ rows, cols });

	for (int b = 0; b < batch; b++) {
		for (int h = 0; h < count_vert; h++) {
			for (int w = 0; w < count_hor; w++) {
				int col_index = (b * count_vert + h) * count_hor + w;  // Номер столбца от 0 до 8
				for (int c = 0; c < channel; c++) {
					for (int kh = 0; kh < size_kernel; kh++) {
						for (int kw = 0; kw < size_kernel; kw++) {
							int h_in = h * stride + kh - padding;
							int w_in = w * stride + kw - padding;
							float value = 0;
							if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
								value = tensor.getValue(b, c, h_in, w_in);
							}
							int row_index = kh * size_kernel + kw;     // Номер строки от 0 до 3
							mat.data[row_index * cols + col_index] = value;
						}
					}
				}
			}
		}
	}
	return mat;
}
Tensor matMul(Tensor& tensorA, Tensor& tensorB) {
	vector<int> tensorA_2d = tensorA.getShape();
	vector<int> tensorB_2d = tensorB.getShape();
	if (tensorA_2d.size() == 2 && tensorB_2d.size() == 2 && tensorA_2d[1] == tensorB_2d[0]) {
		int I = tensorA_2d[0];
		int J = tensorA_2d[1];
		int K = tensorB_2d[1];
		Tensor result({ I, K });
		vector<float>& dataRes = result.getData();
		for (int i = 0; i < dataRes.size(); i++) {
			dataRes[i] = 0;
		}
		vector<float>& dataA = tensorA.getData();
		vector<float>& dataB = tensorB.getData();

		for (int i = 0; i < I; i++) {
			for (int j = 0; j < J; j++) {
				for (int k = 0; k < K; k++) {
					dataRes[i * K + k] += dataA[i * J + j] * dataB[j * K + k];
				}
			}
		}
		return result;
	}
	else {
		throw runtime_error("Несоответствие размеров тензоров!");
	}
}
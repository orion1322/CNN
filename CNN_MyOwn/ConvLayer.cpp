#include <stdio.h>
#include "Tensor.cpp"

using namespace std;

class ConvolutionalLayer {
private:
	Tensor weights;
	Tensor bias;
	int size_kernel;
	int stride;
	int padding;

public:
	Tensor forward(const Tensor& tensor) {
		Tensor mat = imgToCol(tensor, size_kernel, stride, padding);

		vector<int> w_shape = weights.getShape();
		Tensor weights_2d;
		weights_2d.reshape({w_shape[0], w_shape[1] * w_shape[2] * w_shape[3]});
	}
	



};
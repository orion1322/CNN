#include <stdio.h>
#include <math.h>
#include "Tensor.h"

using namespace std;

class ReLU {
private:
	Tensor cache;
public:
	Tensor forward(Tensor& tensor) {
		cache = tensor;

		Tensor output(tensor.getShape());
		vector<float>& out_data = output.getData();
		vector<float>& in_data = tensor.getData();

		for (int i = 0; i < in_data.size(); i++) {
			out_data[i] = max(0.0f, in_data[i]);
		}
		return output;
	}
	Tensor backward(Tensor& grad_output) {
		Tensor grad_input(cache.getShape());
		vector<float>& grad_in_data = grad_input.getData();
		vector<float>& grad_out_data = grad_output.getData();
		vector<float>& cache_data = cache.getData();
		for (int i = 0; i < grad_in_data.size(); i++) {
			grad_in_data[i] = (cache_data[i] > 0) ? grad_out_data[i] : 0;
		}
		return grad_input;
	}
};
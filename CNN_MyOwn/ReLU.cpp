#include <stdio.h>
#include <math.h>
#include "Tensor.cpp"

using namespace std;

class ReLU {
public:
	Tensor forward(Tensor& tensor) {
		Tensor output(tensor.getShape());
		vector<float>& out_data = output.getData();
		vector<float>& in_data = tensor.getData();

		for (int i = 0; i < in_data.size(); i++) {
			out_data[i] = max(0.0f, in_data[i]);
		}
		return output;
	}
};
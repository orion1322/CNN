#include <stdio.h>
#include "Tensor.h"

using namespace std;

class FullyConnected {
private:
	Tensor weights;

public:
	FullyConnected(int in_features, int out_features) {
		weights = Tensor({out_features, in_features});
		vector<float>& w_data = weights.getData();

		for (int i = 0; i < w_data.size(); i++) {
			w_data[i] = rand() % 10;
		}
	}

	Tensor forward(Tensor& tensor) {
		vector<float>& flat_rows = tensor.getData();
		int count_rows = flat_rows.size();
		Tensor flat = tensor.reshape({ 1, count_rows });
		Tensor flat_T = flat.transpose();
		Tensor raw = matMul(weights, flat_T);
		Tensor output = raw.transpose();

		return output;
	}
};
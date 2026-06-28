#include <stdio.h>
#include "Tensor.h"

using namespace std;

class FuncLoss {
public:
	float forward(Tensor& predictions, Tensor& target) {
		vector<float>& pred_data = predictions.getData();
		vector<float>& target_data = target.getData();
		float sum = 0;
		for (int i = 0; i < pred_data.size(); i++) {
			float dif = pred_data[i] - target_data[i];
			sum += dif * dif;
		}
		return sum / pred_data.size();
	}
};
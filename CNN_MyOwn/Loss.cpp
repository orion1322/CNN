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
	Tensor backward(Tensor& prediction, Tensor& target) {
		vector<float>& pred_data = prediction.getData();
		vector<float>& target_data = target.getData();
		Tensor grad(prediction.getShape());
		vector<float>& grad_data = grad.getData();

		// Градиент dLoss/dPred = 2*(pred - target) / n
		float n = pred_data.size();
		for (int i = 0; i < pred_data.size(); i++) {
			grad_data[i] = 2 * (pred_data[i] - target_data[i]) / n;
		}
		return grad;
	}
};
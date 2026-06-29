#pragma once
#include <vector>
#include <iostream>
#include <stdio.h>

using namespace std;

class Tensor {
private:
	vector<int> shape;
	vector<float> data;

public:
	Tensor();
	Tensor(const vector<int>& shape);

	void fillTestData();
	void print();
	vector<int>& getShape();
	vector<float>& getData();
	float getValue(int batch, int channels, int height, int width) const;
	Tensor reshape(const vector<int>& new_shape);
	Tensor transpose();

	friend Tensor matMul(Tensor& tensorA, Tensor& tensorB);
	friend Tensor imgToCol(const Tensor& tensor, int size_kernel, int stride, int padding);
	friend Tensor colToImg(const Tensor& col, const vector<int> input_shape, int size_kernel, int stride, int padding);
};
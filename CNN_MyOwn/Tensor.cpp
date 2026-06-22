#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std;


class Tensor {
private:
	vector<int> shape;
	vector<float> data;

public:
	Tensor() {
		shape = {0, 0, 0, 0};
		data = {};
	}
	Tensor(int a, int b, int c, int d) {
		shape = { a, b, c, d };
		int size = a * b * c * d;
		for (int i = 0; i < size; i++) {
			data.push_back(rand() % 10);
		}
	}

	void print() {
		cout << "Размерность: [";
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

	float& getValue(int a, int b, int c, int d) {
		int index = ((a * shape[1] + b) * shape[2] + c) * shape[3] + d;
		if (data.size() > index && index >= 0) {
			cout << "Индекс: " << index << "\n";
			return data[index];
		}
		else {
			throw runtime_error("Такого индекса нет!");
		}
	}
};
#include <stdio.h>
#include "Tensor.cpp"
#include <locale.h>

using namespace std;


int main() {
	setlocale(LC_ALL, "rus");

	Tensor t(1,3,2,2);
	t.print();

	float& value = t.getValue(4, 0, 1, 1);
	cout << "Значение:" << value;

	return 0;
}
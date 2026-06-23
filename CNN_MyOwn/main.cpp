#include <stdio.h>
#include "Tensor.cpp"
#include <locale.h>

using namespace std;


int main() {
	setlocale(LC_ALL, "rus");

	Tensor t({1,3,2,2});
	t.print();

	float value = t.getValue(0, 1, 2, 1);
	cout << "Значение:" << value << endl;

	//t.reshape({1,3,4});
	//t.print();

    // Тест imgToCol
	Tensor mat;
	mat = imgToCol(t, 3, 1, 1);
    cout << "\nМатрица col:";
    mat.print();

	Tensor A({ 2, 3 });
	A.print();
	Tensor B({ 3, 2 });
	B.print();
	Tensor C({0});
	C.print();
	C = matMul(A, B);
	C.print();

	return 0;
}
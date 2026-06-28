#include <stdio.h>
#include "ConvLayer.cpp"
#include "ReLU.cpp"
#include "Tensor.h"
#include "MaxPooling.cpp"
#include "FullyConnected.cpp"
#include <locale.h>
#include <chrono>

using namespace std;

int main() {
	setlocale(LC_ALL, "rus");

	// ============= ТЕСТЫ ============
	
	// Работа конструктора и вывода
	//Tensor t({1,3,2,2});
	//t.print();

	// Вытягивание значения их тензора по индексу
	//float value = t.getValue(0, 1, 2, 1);
	//cout << "Значение:" << value << endl;

	// Преобразование размеров тензора
	//t.reshape({1,3,4});
	//t.print();

    // ========== ТЕСТ алгоритма imgToCol ==========
	//Tensor mat;
	//mat = imgToCol(t, 3, 1, 1);
	//cout << "\nМатрица col:";
	//mat.print();

	// =========== ТЕСТ умножения матриц ===========
	//Tensor A({ 2, 3 });
	//A.print();
	//Tensor B({ 3, 2 });
	//B.print();
	//Tensor C({0});
	//C.print();
	//C = matMul(A, B);
	//C.print();

	// ========== ТЕСТ слоя свёртки ==========
	//auto start = chrono::high_resolution_clock::now();
	//cout << "Создание слоя" << endl;
	//ConvolutionalLayer conv(3, 16, 3, 1, 1);
	//cout << "Создание входа" << endl;
	//Tensor input({ 1, 3, 1024, 1024 });
	//cout << "Вход создан" << endl;
	//cout << "Запуск forward" << endl;
	//Tensor output = conv.forward(input);
	//auto end = chrono::high_resolution_clock::now();
	//auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
	//cout << "Время выполнения: " << duration.count() << " мс" << endl;

	// ========== ТЕСТ ReLU ==========
	//Tensor input({ 1, 3, 2, 2 });
	//input.fillTestData();
	//input.print();
	//ReLU ReLU;
	//Tensor output = ReLU.forward(input);
	//cout << "ReLU:";
	//output.print();

	// ========== ТЕСТ MaxPooling ==========
	//Tensor input({ 1, 1, 4, 4 });
	//input.fillTestData();
	//input.print();

	//MaxPooling pool;
	//Tensor output = pool.forward(input);
	//cout << "После MaxPooling:";
	//output.print();

	// ========= ТЕСТ полносвязного слоя =========
	//Tensor input({1,4});
	//input.fillTestData();
	//input.print();

	//FullyConnected fc(4, 3);
	//Tensor output = fc.forward(input);
	//output.print();

	// ======== ПОЛНЫЙ ТЕСТ ========
	ConvolutionalLayer conv(3, 16, 3, 1, 1);
	ReLU relu;
	MaxPooling pool;
	FullyConnected fc(16 * 8 * 8, 10);

	Tensor input({ 1,3,16,16 });
	cout << "Вход";
	input.print();

	Tensor output = conv.forward(input);
	cout << "После свёртки";
	output.print();

	output = relu.forward(output);
	cout << "Функия активации";
	output.print();

	output = pool.forward(output);
	cout << "Пулинг";
	output.print();

	output = fc.forward(output);
	cout << "После полносвязной сети";
	output.print();


	return 0;
}
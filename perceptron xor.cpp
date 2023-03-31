#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{
    return x * (1 - x);
}

int main()
{
    int datos[4][3] = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}}; // Datos para el XOR
    srand(time(NULL));
    double pesosCapaOculta[] = {(double)rand() / RAND_MAX, (double)rand() / RAND_MAX};
    double pesosCapaSalida[] = {(double)rand() / RAND_MAX};

    double biasCapaOculta = (double)rand() / RAND_MAX;
    double biasCapaSalida = (double)rand() / RAND_MAX;

    double tasaAprendizaje = 0.5;
    int epocas = 100000;

    for (int i = 0; i < epocas; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // Forward propagation
            double entradaCapaOculta = datos[j][0] * pesosCapaOculta[0] + datos[j][1] * pesosCapaOculta[1] + biasCapaOculta;
            double salidaCapaOculta = sigmoid(entradaCapaOculta);

            double entradaCapaSalida = salidaCapaOculta * pesosCapaSalida[0] + biasCapaSalida;
            double salidaRedNeuronal = sigmoid(entradaCapaSalida);

            // Backpropagation
            double error = datos[j][2] - salidaRedNeuronal;
            double derivadaSalida = sigmoidDerivative(salidaRedNeuronal);
            double deltaSalida = error * derivadaSalida;

            double derivadaCapaOculta = sigmoidDerivative(salidaCapaOculta);
            double deltaCapaOculta = deltaSalida * pesosCapaSalida[0] * derivadaCapaOculta;

            // Actualización de pesos y bias
            pesosCapaSalida[0] += salidaCapaOculta * deltaSalida * tasaAprendizaje;
            biasCapaSalida += deltaSalida * tasaAprendizaje;

            pesosCapaOculta[0] += datos[j][0] * deltaCapaOculta * tasaAprendizaje;
            pesosCapaOculta[1] += datos[j][1] * deltaCapaOculta * tasaAprendizaje;
            biasCapaOculta += deltaCapaOculta * tasaAprendizaje;
        }
    }

    // Verificación de las salidas
    for (int i = 0; i < 4; i++)
    {
        double entradaCapaOculta = datos[i][0] * pesosCapaOculta[0] + datos[i][1] * pesosCapaOculta[1] + biasCapaOculta;
        double salidaCapaOculta = sigmoid(entradaCapaOculta);

        double entradaRedNeuronal = salidaCapaOculta * pesosCapaSalida[0] + biasCapaSalida;
        double salidaRedNeuronal = sigmoid(entradaRedNeuronal);

        cout << "Entradas:" << datos[i][0] << " XOR " << datos[i][1] << " = " << datos[i][2] << " | Perceptron: " << round(salidaRedNeuronal) << endl;
    }

    // Mostrar los pesos útiles del perceptrón
    cout << endl
         << "Pesos capa oculta: " << endl
         << "w1: " << pesosCapaOculta[0]
         << ", w2: " << pesosCapaOculta[1] << endl;
}
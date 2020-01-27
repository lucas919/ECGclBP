#pragma once
#include "Neuron.h"

class Layer {
public:
    Layer(int _nNeurons, int _nInputs);
    ~Layer();

    void setInputs(const double *_inputs, int*** _connections, int jag_length[]); // only for the first layer
    void setInputs(const double* _inputs);
    void convoInputs(const double* _inputs, int noFiltrers, int index, double data[], int i );
    void FullyConvoInputs(const double* _inputs, int noFiltrers, int index, double data[], int i );
    void whatInputs(int noFilters);
    int factorial(int x);
    void fullyConnect(int noFull);
    void initLayer(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
    void calcOutputs();
    void genOutput();
    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    void propInputs(int _index, double _value);
    void propInputs( double _value[], int** _connections, int jag_length[]);
    /*this is for hidden and output layers (not input)*/
    void printLayer();
    void propError(int _neuronIndex, double _nextSum);
    int getnNeurons();
    void setlearningRate(double _learningRate);
    void initLearningAndMomentum(double _learningRate, double _momentum);
    void updateLearningandMomentum();
    double getError(int _neuronIndex);
    double getWeights(int _neuronIndex, int _weightIndex);
    double getInitWeight(int _neuronIndex, int _weightIndex);
    double getWeightChange();
    double getWeightDistance();
    void setError(double _leadError);
    void updateWeights();
    int saveWeights(int _layerIndex, int _neuronCount);
    void snapWeights(int _layerIndex); // This one just saves the final weights
    // i.e. overwrites them

    Neuron *getNeuron(int _neuronIndex);

private:
    int nNeurons = 0;
    int nConvs=0;
    int ii=0;
    const int noAxes = 3;
    int nInputs = 0;
    const double *inputs = 0;
    Neuron **neurons = 0;
    double learningRate = 0;
    double momentum = 0;
    double weightChange=0;
    int** connections;
    int counter = 0;
    int chosen = 0;
    int chosenOnes[];
};

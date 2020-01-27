#include "clbp/Net.h"
#include "clbp/Layer.h"
#include "clbp/Neuron.h"

#include <iostream>

using namespace std;

Net::Net(const int _nLayers, int* _nNeurons, int _nInputs)
{
    nLayers = _nLayers; //no. of layers including inputs and ouputs layers
    layers= new Layer*[nLayers];
    int* nNeuronsp = _nNeurons; //number of neurons in each layer expect input
    nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
    int nInput = 0; //temporary variable to use within the scope of for loop
    for (int i=0; i<nLayers; i++){
        int nNeurons= *nNeuronsp; //no. neurons in this layer
        if (i==0){nInput=nInputs;}
        /* no. inputs to the first layer is equal to no. inputs to the network */
        layers[i]= new Layer(nNeurons, nInput);
        nInput=nNeurons;
        /*no. inputs to the next layer becomes is equal to the number of neurons
         * in the current layer. */
        nNeuronsp++; //point to the no. of neurons in the next layer
    }
    nOutputs=layers[nLayers-1]->getnNeurons();
    //inputs= new double[nInputs];

    nNeurons=0;
    for (int i=0; i<nLayers; i++){
        nNeurons += layers[i]->getnNeurons();
    }
    
    cout << "number of inputs are: " << nInputs << endl;
}

Net::~Net(){
    for (int i=0; i<nLayers; i++){
        delete layers[i];
    }
    delete[] layers;
    //delete[] inputs;
}

void Net::setInputs(const double* _inputs, int*** _connections, int* jag_length[]){
    inputs=_inputs;
    layers[0]->setInputs(inputs, _connections, jag_length[0]); //sets the inputs to the first layer only
}

void Net::convInputs(const double* _inputs, int _noFiltrers, int fullyConnected){
    layers[0]->whatInputs(_noFiltrers);
    inputs = _inputs;
    double data[noAxes];
    int noVars = _noFiltrers*noAxes;
    layers[0]->convoInputs(inputs, noVars, 0, data, 0);
    if(fullyConnected>0) layers[0]->fullyConnect(fullyConnected);
}

void Net::FullyConvInputs(const double* _inputs, int _noFiltrers, int fullyConnected){
    //for the quasi convolved net
    //instead of set inputs
    inputs = _inputs;
    double data[noAxes];
    //buffer for the three elements in combination
    int noVars = _noFiltrers*noAxes;
    //number of inputs to first hidden layer
    layers[0]->FullyConvoInputs(inputs, noVars, 0, data, 0);
    //initiate qcn neurons
    layers[0]->fullyConnect(fullyConnected);
    //initiate specified fully connected neurons
}

void Net::setInputs(const double* _inputs){
    inputs=_inputs;
    layers[0]->setInputs(inputs); //sets the inputs to the first layer only
}

void Net::initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nLayers; i++){
        layers[i]->initLayer(_wim, _bim, _am);
    }
}

void Net::propInputs(){
    for (int i=0; i<nLayers-1; i++){
        layers[i]->calcOutputs();
        for (int j=0; j<layers[i]->getnNeurons(); j++){
            double inputOuput = layers[i]->getOutput(j);
            layers[i+1]->propInputs(j, inputOuput);
        }
    }
    layers[nLayers-1]->calcOutputs();
    /* this calculates the final outoup of the network,
     * i.e. the output of the final layer
     * but this is not fed into any further layer*/
}

void Net::propInputs(int*** _connections, int* jags_length[]){
    //prop inputs funtion overload for PCNN
    //almost the same as regular prop inputs but for 1 line
    for (int i=0; i<nLayers-1; i++){
        layers[i]->calcOutputs();
        double _inputs[layers[i]->getnNeurons()];
        for (int j=0; j<layers[i]->getnNeurons(); j++){
            _inputs[j] = layers[i]->getOutput(j);
        }
        layers[i+1]->propInputs(_inputs, _connections[i+1], jags_length[i+1]);
        //only prop inputs specified in jagged array
    }
    layers[nLayers-1]->calcOutputs();
    /* this calculates the final outoup of the network,
     * i.e. the output of the final layer
     * but this is not fed into any further layer*/
}

double Net::getOutput(int _neuronIndex){
    return (layers[nLayers-1]->getOutput(_neuronIndex));
}

double Net::getSumOutput(int _neuronIndex){
    return (layers[nLayers-1]->getSumOutput(_neuronIndex));
}

int Net::getnLayers(){
    return (nLayers);
}

int Net::getnInputs(){
    return (nInputs);
}

Layer* Net::getLayer(int _layerIndex){
    assert(_layerIndex<nLayers);
    return (layers[_layerIndex]);
}

void Net::propError(){
    double tempError=0;
    double tempWeight=0;
    for (int i=nLayers-1; i>0 ; i--){
        for (int k=0; k<layers[i-1]->getnNeurons();k++){
            double sum=0;
            for (int j=0; j<layers[i]->getnNeurons(); j++){
                tempError=layers[i]->getError(j);
                tempWeight=layers[i]->getWeights(j,k);
                sum+=tempError * tempWeight;
            }
            layers[i-1]->propError(k, sum);
        }
    }
}

void Net::setError(double _leadError){
    /* this is only for the final layer */
    layers[nLayers-1]->setError(_leadError);
    /* if the leadError was diff. for each output neuron
     * then it would be implemented in a for-loop */
}

void Net::updateWeights(){
    for (int i=nLayers-1; i>=0; i--){
        layers[i]->updateWeights();
    }
}

double Net::getWeightDistance(){
    double weightChange = 0 ;
    double weightDistance =0;
    for (int i=0; i<nLayers; i++){
        weightChange += layers[i]->getWeightChange();
    }
    weightDistance=sqrt(weightChange);
    // cout<< "Net: WeightDistance is: " << weightDistance << endl;

    return (weightDistance);
}

double Net::getLayerWeightDistance(int _layerIndex){
    return sqrt(layers[_layerIndex]->getWeightChange());
}

double Net::getWeights(int _layerIndex, int _neuronIndex, int _weightIndex){
    double weight=layers[_layerIndex]->getWeights(_neuronIndex, _weightIndex);
    return (weight);
}

void Net::setLearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nLayers; i++){
        layers[i]->setlearningRate(learningRate);
    }
}

void Net::InitLearningAndMomentum(double _learningRate, double _momentum){
    learningRate=_learningRate;
    momentum= _momentum;
    for (int i=0; i<nLayers; i++){
        layers[i]->initLearningAndMomentum(learningRate, momentum);
    }
}

void Net::updateLearningandMomentum(){
    for (int i=0; i<nLayers; i++){
        layers[i]->updateLearningandMomentum();
    }
}

int Net::getnNeurons(){
    return (nNeurons);
}

void Net::saveWeights(){
    int neuronCount = 0;
    for (int i=0; i<nLayers; i++){
        neuronCount += layers[i]->saveWeights(i, neuronCount);
        layers[i]->snapWeights(i);
    }
}

void Net::printNetwork(){
    cout<< "This network has " << nLayers << " layers" <<endl;
    for (int i=0; i<nLayers; i++){
        cout<< "Layer number " << i << ":" <<endl;
        layers[i]->printLayer();
    }
    cout<< "The output(s) of the network is(are):";
    for (int i=0; i<nOutputs; i++){
        cout<< " " << this->getOutput(i);
    }
    cout<<endl;
}

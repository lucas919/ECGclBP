#include "clbp/Layer.h"
#include "clbp/Neuron.h"

#include <fstream>

Layer::Layer(int _nNeurons, int _nInputs)
{
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new Neuron*[nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new Neuron(nInputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
}

Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    //delete[] inputs;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

void Layer::setInputs(const double* _inputs){
    /*this is only for the first layer*/
    inputs=_inputs;
    for (int j=0; j<nInputs; j++){
        Neuron** neuronsp = neurons;//point to the 1st neuron
        /* sets a temporarily pointer to neuron-pointers
         * within the scope of this function. this is inside
         * the loop, so that it is set to the first neuron
         * everytime a new value is distributed to neurons */
        double input= *inputs; //take this input value
        for (int i=0; i<nNeurons; i++){
            (*neuronsp)->setInput(j,input);
            //set this input value for this neuron
            neuronsp++; //point to the next neuron
        }
        inputs++; //point to the next input value
    }
}

void Layer::setInputs(const double* _inputs, int*** _connections, int jag_length[]){
    /*this is only for the first layer*/
    inputs=_inputs;
    Neuron** neuronsp = neurons;
    for (int i = 0; i <nNeurons; i++){
        int indx = jag_length[i];
        for (int j = 0;j<indx; j++){
            (*neuronsp)->setInput(j,inputs[_connections[0][i][j]]);
        }
        neuronsp++;
    }
}
void Layer::fullyConnect(int noFull){
    double input= *inputs; //take this input value
    int LastNeuron = nConvs+noFull;
    for (ii; ii<LastNeuron;ii++)
        for (int i = 0;i<nInputs;i++)
            neurons[ii]->setInput(i,input);

}


void Layer::convoInputs(const double* _inputs, int noVars, int index, double data[], int i ){
    inputs = _inputs;
    nConvs = (nInputs*(nInputs-1)*(nInputs-2))/(factorial(noAxes));
    inputs = _inputs;
    if (ii >= nConvs) return;
    if (index == noAxes){
        if (ii == chosenOnes[chosen]) {
            for (int j = 0; j < noAxes; j++) {
                neurons[chosen]->setInput(j, data[j]);
                chosen++;
            }
        }
        ii++;
        return;
    }
    if (i>=noVars)
        return;
    data[index] = inputs[i];
    convoInputs(inputs, noVars, index+1, data, i+1);

    convoInputs(inputs, noVars, index, data, i+1);
}

void Layer::FullyConvoInputs(const double* _inputs, int noVars, int index, double data[], int i ){
    //data is 3 element buffer, index and i are counters
    nConvs = (nInputs*(nInputs-1)*(nInputs-2))/(factorial(noAxes));
    //number of 3 element combinations
    inputs = _inputs;

    if (ii >= nConvs) return;
    //combination function, ii is counter iterates for every combination
    if (index == noAxes) {
        //when index=3, buffer is full
        for (int j = 0; j < noAxes; j++) {
            neurons[ii]->setInput(j, data[j]);
            //give neuron designated inputs
        }

        ii++;
        return;
    }

    if (i>=noVars)
        return;
    data[index] = inputs[i];
    FullyConvoInputs(inputs, noVars, index+1, data, i+1);

    FullyConvoInputs(inputs, noVars, index, data, i+1);
}

void Layer::whatInputs(int noFilters){
    int over = ((noFilters-noAxes)/2)*(noFilters-noAxes+1);
    int first_filter = (noFilters-1) * (nInputs-noAxes) + 1 + noFilters - over;
    chosenOnes[0]=first_filter-1;

    double de_incrementer = nInputs - noAxes + 1;
    double running_total = de_incrementer/2*(de_incrementer+1);
    double de_incremented = running_total;
    int answer;

    for (int i = 1; i<noFilters; i++){
        answer = running_total + first_filter - (noFilters-1)*i;
        chosenOnes[i]=answer-2;
        de_incremented -= de_incrementer;
        running_total += de_incremented;
        de_incrementer--;
    }
    for(int i = 0; i<noFilters;i++){
        cout<<chosenOnes[i]<<endl;
    }
}

int::Layer::factorial(int x){
    //gets factorial required for QCNN
    int y = 1;
    for (int i=1;i<=x;i++)
        y *= i;
    return y;
}

void Layer::propInputs(int _index, double _value){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->propInputs(_index, _value);
    }
}

void Layer::propInputs( double _value[], int** _connections, int jag_length[]){
    connections = _connections;
    Neuron** neuronsp = neurons;
    for (int i = 0; i <nNeurons; i++){
        int indx = jag_length[i];
        //for the number of nuerons nuerons in layer
        for (int j = 0;j<indx; j++){
            //for number of connections specified iin jags_length
            int p = connections[i][j];
            (*neuronsp)->propInputs(j,_value[p]);
        }
        neuronsp++;
    }
}




void Layer::calcOutputs(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->calcOutput();
    }
}

void Layer::genOutput(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->genOutput();
    }
}

void Layer::setError(double _leadError){
    /* this is only for the final layer */
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setError(_leadError);
    }
}

void Layer::propError(int _neuronIndex, double _nextSum){
    neurons[_neuronIndex]->propError(_nextSum);
}

double Layer::getError(int _neuronIndex){
    return (neurons[_neuronIndex]->getError());
}

double Layer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double Layer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double Layer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double Layer::getWeightChange(){
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }

    //cout<< "Layer: WeightChange is: " << weightChange << endl;
    return (weightChange);
}

double Layer::getWeightDistance(){
    double weightDistance=sqrt(weightChange);
    return (weightDistance);
}

double Layer::getOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getOutput());
}


void Layer::initLayer(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initNeuron(_wim, _bim, _am);
    }
}

void Layer::setlearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(learningRate);
    }
}

void Layer::initLearningAndMomentum(double _learningRate, double _momentum){
    momentum = _momentum;
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initLearningAndMomentum(learningRate, momentum);
    }
}



void Layer::updateLearningandMomentum(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateLearningandMomentum();
    }
}

void Layer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}

int Layer::getnNeurons(){
    return (nNeurons);
}

int Layer::saveWeights(int _layerIndex, int _neuronCount){
    char l = '0';
    char n = '0';
    l += _layerIndex + 1;
    char decimal = '0';
    bool skip = true;
    for (int i=0; i<nNeurons; i++){
        if (skip == true){
            n += 1;
            }
            if(skip == false){
                skip = true;
            }
        _neuronCount += 1;
        string name = "w";
        name += 'L';
        name += l;
        name += 'N';
        name += decimal;
        name += n;
        name += ".csv";
        neurons[i]->saveWeights(name);
        if (n == '9'){
            decimal += 1;
            n= '0';
            skip = false;
        }
    }
    return (_neuronCount);
}

void Layer::snapWeights(int _layerIndex){
    std::ofstream wfile;
    char l = '0';
    l += _layerIndex + 1;
    string name = "wL";
    name += l;
    name += ".csv";
    wfile.open(name);
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

Neuron* Layer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons);
    return (neurons[_neuronIndex]);
}

void Layer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}

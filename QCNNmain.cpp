#include <iostream>
#include "Fir1.h"
#include <fstream>
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include "Iir.h"
#include "clbp/Layer.h"
#include "clbp/Net.h"
#include "clbp/Neuron.h"
using namespace std;
#define LEARNING_RATE 1
#define Momentum 1
#define Number_of_Filters 2
double Num_filt = Number_of_Filters;


int L3n1[] = {0,1,2,3,4,5,6,7};
int L3n2[] = {0,1,2,3,4,5,6,7};
int L3n3[] = {0,1,2,3,4,5,6,7};
int* layr3[] = {L3n1,L3n2,L3n3};
int L4n1[] = {0,1,2,3,4};
int L4n2[] = {5};
int L4n3[] = {5,6,7,8,9};
int L4n4[] = {10,11,12,13,14};
int L4n5[] = {14};
int L4n6[] = {15,16,17,18,19};
int* layr4[] ={L4n1,L4n2,L4n3,L4n4,L4n5,L4n6};
int outputlayer[] = {0,1,2,3,4,5};
int* out[]= {outputlayer};
int** connections[] ={layr3, layr4, out};
int jag_length2[9] = {2,4,3,3,2,6,4,4,2};
int jag_length[6] = {5,1,5,5,1,5};
int jag_length1[1] = {6};
int* jags_length[] = {jag_length2,jag_length, jag_length1};

int filter_num = 0;
int fs = 250;
int num_neurons[3] = {20,6,1};
int* Num_Neur_point = num_neurons;
Net* net1 = new Net(3, Num_Neur_point, 6);

Iir::Butterworth::BandPass<4> band[(Number_of_Filters*3)];
//initialise IIR band pass filters
double getFreqs(int j, double baseFreq){
    double freq = ((10-baseFreq)/Num_filt+baseFreq)*j;
    //5 is the top, basefreq is the bottom
    //all bands are divisions between these frequencies
    return freq;
}//functions to define filter frequnecies

double get_filter(double freq1, int j, double baseFreq){
    double freq2 = getFreqs(j, baseFreq);
    band[filter_num].setup(fs, freq1, freq2);
    cout<< freq1<<"    "<<freq2<<endl;
    return freq2;
}//initialise the filters

int main (){
    net1->initNetwork(Neuron::W_RANDOM, Neuron::B_RANDOM, Neuron::Act_Sigmoid);
    net1->setLearningRate(LEARNING_RATE);
    //Initialise network with random values for weights and biases, sigmoid activation funvtion
    //set learning rate for all neurons in network
    for(int i = 0; i<3; i++){
        double baseFreq = 0;
        double freq1 = baseFreq;
         for (int j =0; j<Number_of_Filters;j++){
             freq1 = get_filter(freq1, j+1, baseFreq);
             filter_num++;
         }
    }//call the band filter functions

    ifstream datafile;
    datafile.open("ecg_jogging_00.dat");
    if (!datafile) {
        cout << "Failed to open" << endl;
        exit(1);
    }

    ofstream outFile;
    outFile.open("filtered0.dat");
    double control, ecg2, ecg3, accX, accY, accZ;

    Iir::Butterworth::BandStop<4> bs;
    Iir::Butterworth::LowPass<4> low;
    low.setup(fs, 100);


        bs.setup(fs,45, 55);
        double sumAccX= 0;
        double sumAccY = 0;
        double sumAccZ = 0;

        for(int i=0;;i++)
        {
            if (datafile.eof()) break;
            datafile >> control >> ecg2 >> ecg3 >> accX >> accY >> accZ;
            //read data into variables

            ecg2 = bs.filter(ecg2);
            ecg2 = (low.filter(ecg2));
            //prefiltering Einthoven II ecg

            sumAccX = sumAccX + accX;
            sumAccY = sumAccY + accY;
            sumAccZ = sumAccZ + accZ;
            //running total of acc for normalising

            int gain = 1;
            double accX1 = (accX - (sumAccX/(i+1)))*gain;
            double accY1 = (accY - (sumAccY/(i+1)))*gain;
            double accZ1 = (accZ - (sumAccZ/(i+1)))*gain;
            //normalise about time axis and divide by a gain
            //to suit low values of ecg

            double acc[3] = {accX1, accY1, accZ1};
            double Ref1[6];
            // arrays created for data processing
            // and input array
            int p = 0;
            int q = 0;
            for (int I=0;I<3;I++) {
                    Ref1[p] = (band[q++].filter(acc[I]));
                    Ref1[p++] = (band[q++].filter(acc[I]));
            }//Band passing the acc data


            double* InputX = Ref1;
            //argument to net class must be pointer to array

            net1->FullyConvInputs(InputX, Number_of_Filters, 0);
            //Quasi convolved first layer
            net1->propInputs(connections, jags_length);
            //prop inputs using jagged arrays
            double net1_output = net1->getOutput(0);
            //network setting and propagating inputs , get output
            double output_signal = ecg2 - (net1_output);
            //Network out & error signal

            net1->setError(output_signal);
            net1->propError();
            net1->updateWeights();
            //updating the network
            if (i == 3) net1->printNetwork();
            outFile << output_signal <<"    "<< ecg2 <<"    "<< net1_output<<  endl;
        }
        outFile.close();
}

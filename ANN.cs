using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN {

    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public double alpha;
    List<Layer> layers = new List<Layer>();

    public ANN(int nI, int nO, int nH, int nPH, double a) {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;

        if(numHidden > 0) {
            // setup layer for inputs
            layers.Add(new Layer(numNPerHidden, numInputs));

            for (int i = 0; i < numHidden - 1; i++) {
                //create hidden layers
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }
            // create output layer
            layers.Add(new Layer(numOutputs, numNPerHidden));
        } else {
            // create output layer
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    public List<double> Go (List<double> inputValues, List<double> desiredOutput){
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if(inputValues.Count != numInputs) {
            Debug.Log("ERROR: NUMBER OF INPUTS MUST BE " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);
        // loop through each of the layers
        for (int i = 0; i < numHidden + 1; i++) {

            //if its not the first layer
            if (i >0) {
                //set inputs of this layer to the outputs of the previous layer
                inputs = new List<double>(outputs);
            }
            // clear outputs so we can fill it up again by next layer
            outputs.Clear();

            // loop through the number of neurons
            for (int j = 0; j < layers[i].numNeurons; j++) {
                double N = 0;
                // clear inputs
                layers[i].neurons[j].inputs.Clear();

                // loop through each neuron input
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++) {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    // dot product
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }

                N -= layers[i].neurons[j].bias;
                // set output for neuron
                layers[i].neurons[j].output = ActivationFunction(N);
                // add output to output list
                outputs.Add(layers[i].neurons[j].output);
            }
        }
        UpdateWeights(outputs, desiredOutput);
        return outputs;
    }


    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        // loop through the layers in reverse
        for (int i = numHidden; i >= 0; i--)
        {
            // loop through the neurons in the layers
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                // if we are in the output layer
                if (i == numHidden)
                {
                    // calculate the error
                    error = desiredOutput[j] - outputs[j];
                    // error gradient = outputs * (1- outputs) * error
                    // error gradient is assigning amount of error to that weight
                    // calculates how responsible this neuron is for the error
                    // all of errorGradients together is total error
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                    //errorGradient calculated with Delta Rule: en.wikipedia.org/wiki/Delta_rule
                }
                else
                {
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    //get errorGradientSum of the previous layer (goes from last layer to first layer)
                    double errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].numNeurons; p++)
                    {
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }
                // loop through the neuron inputs and update the weights
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    if (i == numHidden)
                    {
                        error = desiredOutput[j] - outputs[j];
                        // alpha is learning rate
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        // if its not the output layer we multiply by errorgradient
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }
                //update bias for each neuron
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }

    }

    //for full list of activation functions
    //see en.wikipedia.org/wiki/Activation_function
    double ActivationFunction(double value)
    {
        return Sigmoid(value);
    }

    double Step(double value) //(aka binary step)
    {
        if (value < 0) return 0;
        else return 1;
    }

    double Sigmoid(double value) //(aka logistic softstep)
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
    }
}

















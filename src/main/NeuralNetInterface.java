package main;

public interface NeuralNetInterface extends CommonInterface {

    final double bias = 1.0; // The input for each neurons bias weight

    double sigmoid(double x);

    double customSigmoid(double x);

    void initializeWeights();

}

package main;

public interface NeuralNetInterface {

    final double bias = 1.0; // The input for each neurons bias weight

    double sigmoid(double x);

    double customSigmoid(double x);

    void initializeWeights();

    double[][] feedForward();

    void backPropagation(double[] outputsHidden, double[] outputs);

    int train(int epochNum);

}

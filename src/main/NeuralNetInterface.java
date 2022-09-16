package main;

public interface NeuralNetInterface extends CommonInterface{

    final double bias = 1.0; // The input for each neurons bias weight

    public double sigmoid(double x);

    public double customSigmoid(double x);

    public void initializeWeights();

    public void zeroWeights();
}

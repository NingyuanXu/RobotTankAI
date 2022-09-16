package main;
import java.io.File;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface {
    @Override
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double customSigmoid(double x) {
        return 0;
    }

    @Override
    public void initializeWeights() {

    }

    @Override
    public void zeroWeights() {

    }

    @Override
    public double outputFor(double[] X) {
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        return 0;
    }

    @Override
    public void save(File argFile) {

    }

    @Override
    public void load(String argFileName) throws IOException {

    }
}

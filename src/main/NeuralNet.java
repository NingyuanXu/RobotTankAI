package main;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNet implements NeuralNetInterface {

    // Model parameters
    private static int numInput = 2;
    private static int numOutput = 1;

    // a parameter for number of hidden neurons
    private int numHidden;
    // true if it is bipolar representation
    private boolean isBipolar;

    // Other hyper-parameters
    private double learningRate;
    private double momentum;

    // Model input and expected expectedOutput
    private double[][] input;
    private double[][] expectedOutput;

    // Model weights and bias term
    private double[][] inputToHiddenWeights;
    private double[][] hiddenToOuputWeights;
    private double[][] inputToHiddenDelta;
    private double[][] hiddenToOuputDelta;

    // Lower or upper bound for initializing weight
    private static double minWeight = -0.5;
    private static double maxWeight = 0.5;

    // Use that to count the current example
    private static int counterNumExample = 0;

    public NeuralNet(int numHidden, boolean isBipolar, double learningRate, double momentum,
                     double[][] input, double[][] expectedOutput) {
        this.numHidden = numHidden;
        this.isBipolar = isBipolar;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.input = input;
        this.expectedOutput = expectedOutput;

        this.inputToHiddenWeights = new double[numInput][numHidden];
        this.hiddenToOuputWeights = new double[numHidden][numOutput];
        this.inputToHiddenDelta = new double[numInput][numHidden];
        this.hiddenToOuputDelta = new double[numHidden][numOutput];
    }

    @Override
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double customSigmoid(double x) {
        return isBipolar ? -1 + 2 / (1 + Math.exp(-x)) : sigmoid(x);
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 0; j < inputToHiddenWeights[0].length; j++) {
                inputToHiddenWeights[i][j] = minWeight + (Math.random() * (maxWeight - minWeight));
            }
        }
        for (int i = 0; i < hiddenToOuputWeights.length; i++) {
            for (int j = 0; j < hiddenToOuputWeights[0].length; j++) {
                hiddenToOuputWeights[i][j] = minWeight + (Math.random() * (maxWeight - minWeight));
            }
        }
    }

    @Override
    public double[][] feedForward() {
        // forward from input to hidden layer
        double[] inToHidden = new double[numHidden];
        for (int i = 0; i < inToHidden.length; i++) {
            inToHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                inToHidden[i] += input[counterNumExample][j] * inputToHiddenWeights[j][i];
            }
            inToHidden[i] = customSigmoid(inToHidden[i] + bias);
        }
        // forward from hidden to out layer
        double[] hiddenToOut = new double[numOutput];
        for (int i = 0; i < hiddenToOut.length; i++) {
            hiddenToOut[i] = 0;
            for (int j = 0; j < hiddenToOuputWeights.length; j++) {
                hiddenToOut[i] += inToHidden[j] * hiddenToOuputWeights[j][i];
            }
            hiddenToOut[i] = customSigmoid(hiddenToOut[i] + bias);
        }
        return new double[][]{inToHidden, hiddenToOut};
    }

    @Override
    public void backPropagation(double[] inToHidden, double[] hiddenToOut) {

    }

    @Override
    public void train(int epochNum) {
        double trainingError = 0.0;
        List<Double> listOfErrors = new ArrayList<>();
        double[] inToHidden;
        double[] hiddenToOut;

        // Initialize the weight at the beginning
        initializeWeights();

        for (int i = 0; i < epochNum; i++) {
            // Every new epoch, re-initialize
            counterNumExample = 0;
            trainingError = 0.0;

            // Repeat checking each example in the dataset
            while (counterNumExample < input.length) {
                inToHidden = feedForward()[0];
                hiddenToOut = feedForward()[1];
                backPropagation(inToHidden, hiddenToOut);
                // Compute the errors here
                for (int j = 0; j < hiddenToOut.length; j++) {
                    trainingError += Math.pow((hiddenToOut[j] - expectedOutput[counterNumExample][j]), 2);
                }
                counterNumExample ++;
            }
            listOfErrors.add(trainingError);
        }

        // Save the results
        try {
            save(listOfErrors);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void save(List<Double> listOfErrors) throws IOException {
        FileWriter writer = new FileWriter("output.txt");
        ArrayList<Double> arr = new ArrayList<>(listOfErrors);
        for(Double dou: arr) {
            writer.write(dou + System.lineSeparator());
        }
        writer.close();
    }

    public static void main(String[] args) {
        NeuralNet neuralNetBinary = new NeuralNet(4, false, 0.2, 0,
                new double[][]{{0,0}, {1,0}, {0,1}, {1,1}}, new double[][]{{0},{1},{1},{0}});
        NeuralNet neuralNetBipolar = new NeuralNet(4, true, 0.2, 0,
                new double[][]{{-1,-1}, {1,-1}, {-1,1}, {1,1}}, new double[][] {{0}, {1}, {1}, {0}});
        neuralNetBinary.train(100);
        neuralNetBipolar.train(100);
    }
}

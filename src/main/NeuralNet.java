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
    private double[][] hiddenToOutputWeights;
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

        this.inputToHiddenWeights = new double[numInput + 1][numHidden];
        this.hiddenToOutputWeights = new double[numHidden + 1][numOutput];
        this.inputToHiddenDelta = new double[numInput + 1][numHidden]; // prev weight changes
        this.hiddenToOuputDelta = new double[numHidden + 1][numOutput];
    }

    @Override
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double customSigmoid(double x) {
        return isBipolar ? -1 + 2 / (1 + Math.exp(-2 * x)) : sigmoid(x);
    }

    @Override
    public void initializeWeights() {
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 0; j < inputToHiddenWeights[0].length; j++) {
                inputToHiddenWeights[i][j] = minWeight + (Math.random() * (maxWeight - minWeight));
            }
        }
        for (int i = 0; i < hiddenToOutputWeights.length; i++) {
            for (int j = 0; j < hiddenToOutputWeights[0].length; j++) {
                hiddenToOutputWeights[i][j] = minWeight + (Math.random() * (maxWeight - minWeight));
            }
        }
    }

    @Override
    public double[][] feedForward() {
        // forward from input to hidden layer
        double[] inToHidden = new double[numHidden + 1];
        inToHidden[0] = bias;
        for (int i = 1; i < inToHidden.length; i++) {
            inToHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                inToHidden[i] += input[counterNumExample][j] * inputToHiddenWeights[j][i-1];
            }
            inToHidden[i] = customSigmoid(inToHidden[i]);
        }
        // forward from hidden to out layer
        double[] hiddenToOut = new double[numOutput];
        for (int i = 0; i < hiddenToOut.length; i++) {
            hiddenToOut[i] = 0;
            for (int j = 0; j < hiddenToOutputWeights.length; j++) {
                hiddenToOut[i] += inToHidden[j] * hiddenToOutputWeights[j][i];
            }
            hiddenToOut[i] = customSigmoid(hiddenToOut[i]);
        }
        return new double[][]{inToHidden, hiddenToOut};
    }

    @Override
    public void backPropagation(double[] inToHidden, double[] hiddenToOut) {
        double[] errorSignalHidden;
        double[] errorSignalOutput;
        // Compute output error signal
        errorSignalOutput = computeErrorSignalOutput(hiddenToOut);
        // update weights hiddenToOut
        for (int i = 0; i < hiddenToOutputWeights.length; i++) {
            for (int j = 0; j < hiddenToOutputWeights[0].length; j++) {
                hiddenToOuputDelta[i][j] = momentum * hiddenToOuputDelta[i][j] + learningRate * errorSignalOutput[j] * inToHidden[i];
                hiddenToOutputWeights[i][j] += hiddenToOuputDelta[i][j];
            }
        }
        errorSignalHidden = computeErrorSignalHidden(inToHidden, errorSignalOutput);
        // update weights inToHidden
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 1; j < inputToHiddenWeights[0].length; j++) {
                inputToHiddenDelta[i][j-1] = momentum * inputToHiddenDelta[i][j-1] + learningRate * input[counterNumExample][i] * errorSignalHidden[j];
                inputToHiddenWeights[i][j-1] += inputToHiddenDelta[i][j-1];
            }
        }
    }

    private double[] computeErrorSignalHidden(double[] inToHidden, double[] errorSignalOutput) {
        double[] errorSignalHidden = new double[numHidden + 1];
        for (int i = 0; i < errorSignalHidden.length; i++) {
            for (int j = 0; j < numOutput; j++) {
                errorSignalHidden[i] += hiddenToOutputWeights[i][j] * errorSignalOutput[j];
            }
            if (isBipolar) {
                errorSignalHidden[i] *= (1 - inToHidden[i] * inToHidden[i]);
            } else {
                errorSignalHidden[i] *= inToHidden[i] * (1 - inToHidden[i]);
            }
        }
        return errorSignalHidden;
    }

    private double[] computeErrorSignalOutput(double[] hiddenToOut) {
        double[] errorSignalOutput = new double[numOutput];
        if (isBipolar) {
            for (int i = 0; i < hiddenToOut.length; i++) {
                errorSignalOutput[i] = (1 - hiddenToOut[i] * hiddenToOut[i]) * (expectedOutput[counterNumExample][i] - hiddenToOut[i]);
            }
        } else {
            for (int i = 0; i < hiddenToOut.length; i++) {
                errorSignalOutput[i] = hiddenToOut[i] * (1 - hiddenToOut[i]) * (expectedOutput[counterNumExample][i] - hiddenToOut[i]);
            }
        }
        return errorSignalOutput;
    }

    @Override
    public int train(int epochNum) {
        double trainingError;
        List<Double> listOfErrors = new ArrayList<>();
        double[] inToHidden;
        double[] hiddenToOut;
        int endingEpoch = 0;
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
            trainingError /= 2; // 1/2 is added just to make derivative simpler, formula
            listOfErrors.add(trainingError);
            if (trainingError < 0.05) {
                System.out.println("Reach total error less than 0.05 in epoch: " + i);
                endingEpoch = i;
                break;
            }
        }
        // Save the results
        try {
            save(listOfErrors);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return endingEpoch;
    }

    @Override
    public void save(List<Double> listOfErrors) throws IOException {
        String s = isBipolar ? "bipolar_" + String.valueOf(momentum) +  ".csv" : "binary_" + String.valueOf(momentum) +  ".csv";
        FileWriter writer = new FileWriter(s);
        ArrayList<Double> arr = new ArrayList<>(listOfErrors);
        for(Double dou: arr) {
            writer.write(dou + System.lineSeparator());
        }
        writer.close();
    }

    public int repeatTraining(int num) {
        int ret = 0;
        for (int i = 0; i < num; i++) {
            ret += train(30000);
        }
        return ret / num;
    }

    public static void main(String[] args) {
        NeuralNet neuralNetBinaryNoMomentum = new NeuralNet(4, false, 0.2, 0,
                new double[][]{{bias,0,0}, {bias,1,0}, {bias,0,1}, {bias,1,1}}, new double[][]{{0},{1},{1},{0}});
        NeuralNet neuralNetBipolarNoMomentum = new NeuralNet(4, true, 0.2, 0,
                new double[][]{{bias,-1,-1}, {bias,1,-1}, {bias,-1,1}, {bias,1,1}}, new double[][] {{-1}, {1}, {1}, {-1}});
        NeuralNet neuralNetBipolarMomentum = new NeuralNet(4, true, 0.2, 0.9,
                new double[][]{{bias,-1,-1}, {bias,1,-1}, {bias,-1,1}, {bias,1,1}}, new double[][] {{-1}, {1}, {1}, {-1}});
        neuralNetBinaryNoMomentum.train(30000); // give it a large enough number
        neuralNetBipolarNoMomentum.train(10000);
        neuralNetBipolarMomentum.train(10000);
        System.out.println("The average epoch number to reach less than 0.05 error is : " +neuralNetBinaryNoMomentum.repeatTraining(500));
        System.out.println("The average epoch number to reach less than 0.05 error is : " +neuralNetBipolarNoMomentum.repeatTraining(500));
        System.out.println("The average epoch number to reach less than 0.05 error is : " + neuralNetBipolarMomentum.repeatTraining( 500));
    }
}

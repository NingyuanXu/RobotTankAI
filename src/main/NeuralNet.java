package main;

import java.io.*;
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
    private boolean isPreTrained;

    // Other hyper-parameters
    private double learningRate;
    private double momentum;
    private int batchSize;

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
                     double[][] input, double[][] expectedOutput, boolean isPreTrained, int batchSize) {
        this.numHidden = numHidden;
        this.isBipolar = isBipolar;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.input = input;
        this.expectedOutput = expectedOutput;
        this.isPreTrained = isPreTrained;
        this.batchSize = batchSize;
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
        return isBipolar ? -1 + 2 / (1 + Math.exp(-x)) : sigmoid(x);
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
        int counter = counterNumExample;
        inToHidden[0] = bias;
        for (int m = 0; m < batchSize; m++) {
            if (counter >= input.length) break;
            for (int i = 1; i < inToHidden.length; i++) {
                inToHidden[i] = 0;
                for (int j = 0; j < inputToHiddenWeights.length; j++) {
                    inToHidden[i] += input[counter][j] * inputToHiddenWeights[j][i - 1];
                }
                inToHidden[i] = customSigmoid(inToHidden[i]);
            }
            counter++;
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
                errorSignalHidden[i] *= (1 - inToHidden[i] * inToHidden[i]) * 0.5;
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
                errorSignalOutput[i] = (1 - hiddenToOut[i] * hiddenToOut[i]) * (expectedOutput[counterNumExample][i] - hiddenToOut[i]) * 0.5;
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
        // Add bias term
        input = appendBiasTerm(input);
        // Initialize the weight at the beginning
        if (isPreTrained) {
            try {
                load();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            initializeWeights();
        }
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
                counterNumExample += batchSize;
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
            writeErrorToFile(listOfErrors);
        } catch (IOException e) {
            e.printStackTrace();
        }
        save();
        return endingEpoch;
    }

    public double[][] appendBiasTerm(double[][] input) {
        double[][] newInput = new double[input.length][input[0].length + 1];
        for (int i = 0; i < newInput.length; i++) {
            newInput[i][0] = bias;
            for (int j = 1; j < newInput[0].length; j ++) {
                newInput[i][j] = input[i][j-1];
            }
        }
        return newInput;
    }

    public void writeErrorToFile(List<Double> listOfErrors) throws IOException {
        String s = isBipolar ? "bipolar_" + String.valueOf(momentum) +  ".csv" : "binary_" + String.valueOf(momentum) +  ".csv";
        FileWriter writer = new FileWriter(s);
        ArrayList<Double> arr = new ArrayList<>(listOfErrors);
        for(Double dou: arr) {
            writer.write(dou + System.lineSeparator());
        }
        writer.close();
    }

    @Override
    public void save() {
        StringBuilder builder1 = new StringBuilder();
        for(int i = 0; i < inputToHiddenWeights.length; i++)
        {
            for(int j = 0; j < inputToHiddenWeights[0].length; j++)
            {
                builder1.append(inputToHiddenWeights[i][j]+"");
                if(j < inputToHiddenWeights[0].length - 1)
                    builder1.append(",");
            }
            //append new line at the end of the row
            builder1.append("\n");
        }
        BufferedWriter writer1 = null;
        try {
            writer1 = new BufferedWriter(new FileWriter("inputToHiddenWeights.txt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            //save the string representation of the lookup table
            writer1.write(builder1.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            writer1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        StringBuilder builder2 = new StringBuilder();
        for(int i = 0; i < hiddenToOutputWeights.length; i++)
        {
            for(int j = 0; j < hiddenToOutputWeights[0].length; j++)
            {
                builder2.append(hiddenToOutputWeights[i][j]+"");
                if(j < hiddenToOutputWeights[0].length - 1)
                    builder2.append(",");
            }
            //append new line at the end of the row
            builder2.append("\n");
        }
        BufferedWriter writer2 = null;
        try {
            writer2 = new BufferedWriter(new FileWriter("hiddenToOutputWeights.txt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            //save the string representation of the lookup table
            writer2.write(builder2.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            writer2.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void load() throws IOException {
        BufferedReader reader1 = null;
        try {
            reader1 = new BufferedReader(new FileReader("inputToHiddenWeights.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line = "";
        int row = 0;
        while((line = reader1.readLine()) != null)
        {
            //note that if you have used space as separator you have to split on " "
            String[] cols = line.split(",");
            int col = 0;
            for(String  c : cols)
            {
                inputToHiddenWeights[row][col] = Double.parseDouble(c);
                col++;
            }
            row++;
        }
        try {
            reader1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        BufferedReader reader2 = null;
        try {
            reader2 = new BufferedReader(new FileReader("hiddenToOutputWeights.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        int row2 = 0;
        while((line = reader2.readLine()) != null)
        {
            //note that if you have used space as separator you have to split on " "
            String[] cols = line.split(",");
            int col = 0;
            for(String  c : cols)
            {
                hiddenToOutputWeights[row2][col] = Double.parseDouble(c);
                col++;
            }
            row2++;
        }
        try {
            reader2.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double scaleRange( double val, double minIn, double maxIn, double minOut, double maxOut){
        double result;
        result = minOut + ((maxOut - minOut) * (val - minIn) / (maxIn - minIn));
        return result;
    }

    public static void main(String[] args) {
        NeuralNet neuralNetBinaryNoMomentum = new NeuralNet(4, false, 0.2, 0,
                new double[][]{{0,0}, {1,0}, {0,1}, {1,1}}, new double[][]{{0},{1},{1},{0}}, false, 1);
        NeuralNet neuralNetBipolarNoMomentum = new NeuralNet(4, true, 0.2, 0,
                new double[][]{{-1,-1}, {1,-1}, {-1,1}, {1,1}}, new double[][] {{-1}, {1}, {1}, {-1}}, false, 1);
        NeuralNet neuralNetBipolarMomentum = new NeuralNet(4, true, 0.2, 0.9,
                new double[][]{{-1,-1}, {1,-1}, {-1,1}, {1,1}}, new double[][] {{-1}, {1}, {1}, {-1}}, false, 1);
        NeuralNet neuralNetBinaryMomentum = new NeuralNet(4, false, 0.2, 0.9,
                new double[][]{{0,0}, {1,0}, {0,1}, {1,1}}, new double[][] {{0}, {1}, {1}, {0}}, false, 1);
        neuralNetBinaryNoMomentum.train(10000);
        neuralNetBinaryMomentum.train(10000);
        neuralNetBipolarNoMomentum.train(10000);
        neuralNetBipolarMomentum.train(10000);
    }
}

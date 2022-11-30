package main;

import robocode.RobocodeFileWriter;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class RLNeuralNet implements NeuralNetInterface  {

    // Model parameters
    private static int numInput = 5;
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
    static LogFile log = new LogFile();
    private static double discountFactor = 0.1;

    public RLNeuralNet(int numHidden, boolean isBipolar, double learningRate, double momentum,
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

    public RLNeuralNet(int numHidden, boolean isBipolar, double learningRate, double momentum, boolean isPreTrained, int batchSize){
        this.numHidden = numHidden;
        this.isBipolar = isBipolar;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.isPreTrained = isPreTrained;
        this.batchSize = batchSize;
        this.inputToHiddenWeights = new double[numInput + 1][numHidden];
        this.hiddenToOutputWeights = new double[numHidden + 1][numOutput];
        this.inputToHiddenDelta = new double[numInput + 1][numHidden]; // prev weight changes
        this.hiddenToOuputDelta = new double[numHidden + 1][numOutput];
    }

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


    public void writeErrorToFile(List<Double> listOfErrors) throws IOException {
        String s = isBipolar ? "bipolar_" + String.valueOf(momentum) +  ".csv" : "binary_" + String.valueOf(momentum) +  ".csv";
        FileWriter writer = new FileWriter(s);
        ArrayList<Double> arr = new ArrayList<>(listOfErrors);
        for(Double dou: arr) {
            writer.write(dou + System.lineSeparator());
        }
        writer.close();
    }

    public double[][] feedForward(double[] state, int action) {
        // forward from input to hidden layer
        //todo:change state t
        double[] inToHidden = new double[numHidden + 1];
        double[] hiddenToOut = new double[numOutput];
        inToHidden[0] = bias;

        for (int i = 1; i < inToHidden.length; i++) {
            inToHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                inToHidden[i] += adjustInputs(state)[action][j] * inputToHiddenWeights[j][i - 1];
            }
            inToHidden[i] = customSigmoid(inToHidden[i]);
        }

        // forward from hidden to out layer
        for (int i = 0; i < hiddenToOut.length; i++) {
            hiddenToOut[i] = 0;
            for (int j = 0; j < hiddenToOutputWeights.length; j++) {
                hiddenToOut[i] += inToHidden[j] * hiddenToOutputWeights[j][i];
            }
            hiddenToOut[i] = customSigmoid(hiddenToOut[i]);
        }
        return new double[][]{inToHidden, hiddenToOut};
    }

    public void backPropagation(double[] inToHidden, double[] hiddenToOut, double[] state, int action, double Q) {
        double[] errorSignalHidden;
        double[] errorSignalOutput;
        // Compute output error signal
        errorSignalOutput = computeErrorSignalOutput(hiddenToOut, Q);
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
                inputToHiddenDelta[i][j-1] = momentum * inputToHiddenDelta[i][j-1] + learningRate * adjustInputs(state)[action][i] * errorSignalHidden[j];
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

    private double[] computeErrorSignalOutput(double[] hiddenToOut, double Q) {
        double[] errorSignalOutput = new double[numOutput];
        if (isBipolar) {
            for (int i = 0; i < hiddenToOut.length; i++) {
                errorSignalOutput[i] = (1 - hiddenToOut[i] * hiddenToOut[i]) * (Q - hiddenToOut[i]) * 0.5;
            }
        } else {
            for (int i = 0; i < hiddenToOut.length; i++) {
                errorSignalOutput[i] = hiddenToOut[i] * (1 - hiddenToOut[i]) * (Q - hiddenToOut[i]);
            }
        }
        return errorSignalOutput;
    }

    public void train(double[] preState, double[] curState, int preAction, int curAction, boolean onPolicy, double reward) {
        double[] inToHidden;
        double[] hiddenToOut;
        // Add bias term
        double newQ = 0.0, Q;

        inToHidden = feedForward(preState,preAction)[0];
        hiddenToOut = feedForward(preState,preAction)[1];

        Q = hiddenToOut[numOutput-1];
        if(onPolicy){
            newQ = Q + learningRate * (reward + discountFactor * feedForward(preState,preAction)[1][numOutput-1] - Q);
        } else {
            newQ = Q + learningRate * (reward + discountFactor * getMaxQ(curState)[1] - Q);
        }
        backPropagation(inToHidden, hiddenToOut, preState, preAction, newQ);
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

    public double[][] adjustInputs(double[] state) {
        //Normalize actions to range [-1,1]
        double[] normalizedActions = new double[RobotStates.Action.values().length];
        for (int i = 0; i < RobotStates.Action.values().length; i++) {
            normalizedActions[i] = scaleRange(i,0,RobotStates.Action.values().length-1, -1,1);
        }

        double[][] inputs = new double[RobotStates.Action.values().length][numInput + 1];
        for (int j = 0; j < RobotStates.Action.values().length; j++) {
            double[] normalizedStates = normalizeStatesInputs(state,-1,1); //Normalize states inputs
            inputs[j][0] = bias;

            for (int k = 1; k < numInput - 1; k++) {
                inputs[j][k] = normalizedStates[k];
            }
            switch (j % RobotStates.Action.values().length) {
                case 0:
                    inputs[j][numInput-1] = normalizedActions[0];
                    break;
                case 1:
                    inputs[j][numInput-1] = normalizedActions[1];
                    break;
                case 2:
                    inputs[j][numInput-1] = normalizedActions[2];
                    break;
                case 3:
                    inputs[j][numInput-1] = normalizedActions[3];
                    break;
            }
        }
        return inputs;
    }

    public static double[] normalizeStatesInputs(double[] statesInputs, double lowerBound, double upperBound) {
        double[] normalizedStates = new double[4];
        for(int i = 0; i < 4; i++) {
            switch (i) {
                case 0:
                    normalizedStates[0] = scaleRange(statesInputs[0],0,100,lowerBound,upperBound);
                    break;
                case 1:
                    normalizedStates[1] = scaleRange(statesInputs[1],0,100,lowerBound,upperBound);
                    break;
                case 2:
                    normalizedStates[2] = scaleRange(statesInputs[2],0,1000,lowerBound,upperBound);
                    break;
                case 3:
                    normalizedStates[3] = scaleRange(statesInputs[3],0,300,lowerBound,upperBound);
                    break;
            }
        }
        return normalizedStates;
    }

    public static double scaleRange(double val, double minIn, double maxIn, double minOut, double maxOut){
        double result;
        result = minOut + ((maxOut - minOut) * (val - minIn) / (maxIn - minIn));
        return result;
    }

    public double[] getMaxQ(double[] state){
        double maxQ = Double.MIN_VALUE;
        int bestAction = 0;
        double[] result = new double[2];
        for(int i = 0; i < RobotStates.Action.values().length; i++){
            double cur = feedForward(state,i)[1][numOutput-1];
            if(cur >= maxQ){
                maxQ = cur;
                bestAction = i;
            }
        }
        result[0] = (double) bestAction;
        result[1] = maxQ;
        return result;
    }



    public void save(File inToHidden, File hiddToOut) {
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
        RobocodeFileWriter writer1 = null;
        try {
            writer1 = new RobocodeFileWriter(inToHidden);
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
        RobocodeFileWriter writer2 = null;
        try {
            writer2 = new RobocodeFileWriter(hiddToOut);
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

    public void load(File intoHidd, File hiddtoOut) throws IOException {
        BufferedReader reader1 = null;
        try {
            reader1 = new BufferedReader(new FileReader(intoHidd));
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
            reader2 = new BufferedReader(new FileReader(hiddtoOut));
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

}
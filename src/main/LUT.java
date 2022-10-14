package main;


import java.io.*;

public class LUT implements LUTInterface{

    static int numStates = RobotStates.numStates;
    static int numActions = RobotActions.numActions;
    public double[][] lookUpTable;

    public LUT() {
        lookUpTable = new double[numStates][numActions];
        initialiseLUT();
    }

    public double getLookUpTableValue(int state, int action) {
        return lookUpTable[state][action];
    }

    // Given a state, find the maximum look up table value
    // associated with that state as well as action index
    public double[] getMaxLUTActionAndQVal(int state) {
        double maxQvalue = Double.NEGATIVE_INFINITY;
        int bestAction = -1;
        for (int i = 0; i < lookUpTable[state].length; i++) {
            if (lookUpTable[state][i] > maxQvalue) {
                maxQvalue = lookUpTable[state][i];
                bestAction = i;
            }
        }
        return new double[]{bestAction, maxQvalue};
    }

    public void setLookUpTableValue(double value, int state, int action) {
        lookUpTable[state][action] = value;
    }

    @Override
    public void initialiseLUT() {
        // all initialize to zero first
        for (int i = 0; i < lookUpTable.length; i++) {
            for (int j = 0; j < lookUpTable[0].length; j++) {
                lookUpTable[i][j] = 0.0;
            }
        }
    }

    @Override
    public void save() {
        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < lookUpTable.length; i++)
        {
            for(int j = 0; j < lookUpTable[0].length; j++)
            {
                builder.append(lookUpTable[i][j]+"");
                if(j < lookUpTable[0].length - 1)
                    builder.append(",");
            }
            //append new line at the end of the row
            builder.append("\n");
        }
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter("LUT.txt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            //save the string representation of the lookup table
            writer.write(builder.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void load() throws IOException {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader("LUT.txt"));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line = "";
        int row = 0;
        while((line = reader.readLine()) != null)
        {
            //note that if you have used space as separator you have to split on " "
            String[] cols = line.split(",");
            int col = 0;
            for(String  c : cols)
            {
                lookUpTable[row][col] = Double.parseDouble(c);
                col++;
            }
            row++;
        }
        try {
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        LUT lut = new LUT();
        int index = RobotStates.getStateIndex(1,1,1,1,1,1);
        System.out.println(lut.lookUpTable[index][5]);
        System.out.println(lut.getMaxLUTActionAndQVal(1)[0]);
        lut.save();
        try {
            lut.load();
            System.out.println(lut.lookUpTable[0][5]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

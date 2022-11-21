package main;


import robocode.RobocodeFileOutputStream;

import java.io.*;

public class LUT implements LUTInterface{

    static int numStates = RobotStates.numStates;
    static int numActions = RobotActions.numActions;
    static double[][] lookUpTable;

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

    public void save(File argFile) {
        PrintStream saveLUT = null;
        try {
            saveLUT = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int i = 0; i < numStates; i++)
                for (int j = 0; j < numActions; j++)
                    saveLUT.println(lookUpTable[i][j]);

            if (saveLUT.checkError())
                System.out.println("Could not save the data!");
            saveLUT.close();
        }
        catch (IOException e)
        {
            System.out.println("IOException trying to write: " + e);
        }
        finally
        {
            try {
                if (saveLUT != null)
                    saveLUT.close();
            }
            catch (Exception e)
            {
                System.out.println("Exception trying to close witer: " + e);
            }
        }
    }

    public void load(String argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream(argFileName);
        BufferedReader inputReader = null;

        try   {
            inputReader = new BufferedReader(new InputStreamReader(inputFile));
            for (int i = 0; i < RobotStates.numStates; i++)
                for (int j = 0; j < RobotActions.numActions; j++)
                    lookUpTable[i][j] = Double.parseDouble(inputReader.readLine());
        }
        catch (IOException e)   {
            System.out.println("IOException trying to open reader: " + e);
        }
        catch (NumberFormatException e)   {
        }
        finally {
            try {
                if (inputReader != null)
                    inputReader.close();
            }
            catch (IOException e) {
                System.out.println("IOException trying to close reader: " + e);
            }
        }
    }
}

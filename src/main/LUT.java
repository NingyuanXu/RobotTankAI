package main;

import robocode.RobocodeFileOutputStream;

import java.io.*;

public class LUT implements LUTInterface{

    private int myHp, enemyHp , enemDis, wallDis, action;
    public final int hpSize = RobotStates.MyHP.values().length;
    public final int eneDisSize = RobotStates.EneDis.values().length;
    public final int wallDisSize = RobotStates.WallDis.values().length;
    public final int actionSize = RobotStates.Action.values().length;
    public double[][][][][] lut;
    public int[][][][][] visited;

    public LUT() {
        lut = new double[hpSize][hpSize][eneDisSize][wallDisSize][actionSize];
        visited = new int[hpSize][hpSize][eneDisSize][wallDisSize][actionSize];
        initialiseLUT();
    }

    // Given a state, find the maximum look up table value
    // associated with that state as well as action index
    @Override
    public void initialiseLUT() {
        // all initialize to zero first
        for (int i = 0; i < lut.length; i++) {
            for (int j = 0; j < lut[0].length; j++) {
                for (int k = 0; k < lut[0][0].length; k++) {
                    for (int l = 0; l < lut[0][0][0].length; l++) {
                        for (int m = 0; m < lut[0][0][0][0].length; m++) {
                            lut[i][j][k][l][m] = Math.random();
                            visited[i][j][k][l][m] = 0;
                        }
                    };
                }
            }
        }
    }

    public int getRandomAction(){
        return (int) (actionSize * Math.random());
    }
    public int getBestAction(int myHp, int enemyHp ,int enemyDis, int wallDis){
        double max = -1;
        int bestAction = -1;
        for(int i = 0; i < actionSize; i++){
            if(max < lut[myHp][enemyHp][enemyDis][wallDis][i]){
                max = lut[myHp][enemyHp][enemyDis][wallDis][i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    public double getQValue(int myHp, int enemyHp ,int enemyDis, int wallDis, int action){
        return lut[myHp][enemyHp][enemyDis][wallDis][action];
    }

    public void setQValue(int myHp, int enemyHp ,int enemyDis, int wallDis, int action, double value){
        lut[myHp][enemyHp][enemyDis][wallDis][action] = value;
        visited[myHp][enemyHp][enemyDis][wallDis][action] += 1;
    }

    public void save(File argFile) {
        PrintStream saveLUT = null;
        try {
            saveLUT = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int i = 0; i < lut.length; i++) {
                for (int j = 0; j < lut[0].length; j++) {
                    for (int k = 0; k < lut[0][0].length; k++) {
                        for (int l = 0; l < lut[0][0][0].length; l++) {
                            for (int m = 0; m < lut[0][0][0][0].length; m++) {
                                saveLUT.println(lut[i][j][k][l][m]);
                            }
                        };
                    }
                }
            }

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
                System.out.println("Exception trying to close writer: " + e);
            }
        }
    }

    public void load(File argFileName) throws IOException {
        FileInputStream inputFile = new FileInputStream(argFileName);
        BufferedReader inputReader = null;

        try   {
            inputReader = new BufferedReader(new InputStreamReader(inputFile));
            for (int i = 0; i < lut.length; i++) {
                for (int j = 0; j < lut[0].length; j++) {
                    for (int k = 0; k < lut[0][0].length; k++) {
                        for (int l = 0; l < lut[0][0][0].length; l++) {
                            for (int m = 0; m < lut[0][0][0][0].length; m++) {
                                lut[i][j][k][l][m] = Double.parseDouble(inputReader.readLine());
                            }
                        };
                    }
                }
            }
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

    public static void main(String[] args) {
    }
}

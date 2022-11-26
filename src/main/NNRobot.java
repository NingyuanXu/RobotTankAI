package main;

import robocode.*;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NNRobot extends AdvancedRobot {

    // Declare global variables
    private double myHp = 100;
    private double enemyHp = 100;
    private double eneDis = 601;
    private double wallDis = 601;
    private RobotStates.Action action = RobotStates.Action.forward;

    private double preMyHp = 100;
    private double preEnemyHp = 100;
    private double preEneDis = 601;
    private double preWallDis = 601;
    private RobotStates.Action preAction = RobotStates.Action.forward;

    private RobotStates.Operation operation = RobotStates.Operation.scan;
    private int actionSize = RobotStates.Action.values().length;
    public double myX = 0.0, myY = 0.0, myHP = 100, enemyHP = 100, dis = 0.0;
    public static boolean takeImmediate = true, onPolicy = true;
    private double gamma = 0.9, alpha = 0.1, epsilon = 0.0, Q = 0.0, reward = 0.0;
    private final double immediateBonus = 0.5, terminalBonus = 1.0, immediatePenalty = -0.1, terminalPenalty = -0.2;

    public static int curActionIndex;
    public static double enemyBearing;

    public static int totalRound = 0, round = 0, winRound = 0;
    public static double winPercentage = 0.0;
    public static String fileToSaveName = LUTRobot.class.getSimpleName() + "-" + "winningRate" + ".log";
    public static String fileToSaveLUT = "lut0.3.txt";
    static LogFile log = new LogFile();
    static int winRateRound = 100;

    public static LUT lut = new LUT();
    public static boolean isLoad = true, isSave = false;

    public static int batchSize = 10;
    public static ReplayMemory<Experience> replayMem = new ReplayMemory<>(batchSize);

    //todo: change hyper-params
    public static RLNeuralNet nn = new RLNeuralNet(12,true,0.05,0.9, false, 1);

    public RobotStates.MyHP getHP(double hp){
        if(hp < 0){
            return null;
        } else if(hp <= 33){
            return RobotStates.MyHP.low;
        } else if(hp <= 67){
            return RobotStates.MyHP.medium;
        } else {
            return RobotStates.MyHP.high;
        }
    }

    public RobotStates.EneDis getEneDis(double dis){
        if(dis < 0){
            return null;
        } else if(dis < 300){
            return RobotStates.EneDis.close;
        } else if(dis < 600){
            return RobotStates.EneDis.medium;
        } else {
            return RobotStates.EneDis.far;
        }
    }

    public RobotStates.WallDis getWallDis(double x1, double x2) {
        double width = getBattleFieldWidth();
        double height = getBattleFieldHeight();
        double dist1 = width - x1;
        double dist2 = height - x2;
        double closestWall = Math.min(Math.min(x1, dist1), Math.min(x2, dist2));

        if (closestWall < 30) {
            return RobotStates.WallDis.close;
        } else if (closestWall < 80) {
            return RobotStates.WallDis.medium;
        } else {
            return RobotStates.WallDis.far;
        }
    }

    public RobotStates.WallDis getWallDis(double wall_dis){
        if (wall_dis < 30) {
            return RobotStates.WallDis.close;
        } else if (wall_dis < 80) {
            return RobotStates.WallDis.medium;
        } else {
            return RobotStates.WallDis.far;
        }
    }

    public double getWallDDis(double x1, double x2){
        double width = getBattleFieldWidth();
        double height = getBattleFieldHeight();
        double dist1 = width - x1;
        double dist2 = height - x2;
        return Math.min(Math.min(x1, dist1), Math.min(x2, dist2));
    }

    public double getQ(double reward, boolean onPolicy){
        double preQ = lut.getQValue(
                getHP(preMyHp).ordinal(),
                getHP(preEnemyHp).ordinal(),
                getEneDis(preEneDis).ordinal(),
                getWallDis(preWallDis).ordinal(),
                preAction.ordinal());
        double curQ = lut.getQValue(
                getHP(myHp).ordinal(),
                getHP(enemyHp).ordinal(),
                getEneDis(eneDis).ordinal(),
                getWallDis(wallDis).ordinal(),
                action.ordinal()
        );
        int bestAction = lut.getBestAction(
                getHP(myHp).ordinal(),
                getHP(enemyHp).ordinal(),
                getEneDis(eneDis).ordinal(),
                getWallDis(wallDis).ordinal()
        );
        double maxQ = lut.getQValue(
                getHP(myHp).ordinal(),
                getHP(enemyHp).ordinal(),
                getEneDis(eneDis).ordinal(),
                getWallDis(wallDis).ordinal(),
                bestAction
        );
        double res = onPolicy ?
                preQ + alpha * (reward + gamma * curQ - preQ) :
                preQ + alpha * (reward + gamma * maxQ - preQ);
        return res;
    }

    public void run() {
        super.run();
        if(isLoad) {
            try {
                lut.load(getDataFile(fileToSaveLUT));
                isLoad = false;
            } catch (Exception e) {
                System.out.println(e);
            }
        }
        setBulletColor(Color.red);
        setGunColor(Color.darkGray);
        setBodyColor(Color.blue);
        setRadarColor(Color.white);
        myHp = 100;
        while (true) {
            switch(operation) {
                case scan: {
                    reward = 0.0;
                    turnRadarLeft(90);
                    break;
                }
                case action: {
                    RobotStates.WallDis wall_state = getWallDis(myX, myY);
                    double[][] scaled = new double[][]{{myHP,enemyHP,eneDis,wallDis}};
                    scaled = rescaleInput(scaled);
                    curActionIndex = (Math.random() <= epsilon) ?
                            lut.getRandomAction() :
                            nnGetBestAction(scaled[0][0],scaled[0][1],scaled[0][2],scaled[0][3]);
                    action = RobotStates.Action.values()[curActionIndex];
                    switch (action){
                        case fire: {
                            turnGunRight((getHeading() - getGunHeading() + enemyBearing));
                            fire(3);
                            break;
                        }
                        case left: {
                            setTurnLeft(30);
                            execute();
                            break;
                        }
                        case right: {
                            setTurnRight(30);
                            execute();
                        }
                        case forward: {
                            setAhead(100);
                            execute();
                            break;
                        }
                        case backward: {
                            setBack(100);
                            execute();
                            break;
                        }
                    }
                    Q = getQ(reward, onPolicy);
                    lut.setQValue(getHP(preMyHp).ordinal(),
                            getHP(preEnemyHp).ordinal(),
                            getEneDis(preEneDis).ordinal(),
                            getWallDis(preWallDis).ordinal(),
                            preAction.ordinal(),
                            Q
                    );
                    double[] preState = new double[]{preMyHp, preEnemyHp, preEneDis, preWallDis};
                    double[] curState = new double[]{myHP, enemyHP, eneDis, wallDis};
                    replayMem.add(new Experience(preState, preAction.ordinal(), reward, curState));

                    replayTrain();
                    operation = RobotStates.Operation.scan;
                    break;
                }
            }
        }
    }

    public void writeErrorToFile(List<Double> listOfErrors) throws IOException {
        String s = "loss.csv";
        File folderDst1 = getDataFile(s);
        ArrayList<Double> arr = new ArrayList<>(listOfErrors);
        for(Double dou: arr) {
            log.writeToFileLoss(folderDst1, dou);
        }
    }

    public void replayTrain(){
        int train_size = Math.min(replayMem.sizeOf(), batchSize);
        Object[] vector = replayMem.sample(train_size);
        ArrayList<double[]> list = new ArrayList<>();
        ArrayList<double[]> out_list = new ArrayList<>();
        for(Object e: vector){
            Experience exp = (Experience) e;

            double[] preState = exp.getState_t();
            double[] preStateAction = Arrays.copyOf(preState, preState.length + 1);
            preStateAction[preStateAction.length - 1] = exp.getAction_t();
            list.add(preStateAction);
            out_list.add(new double[]{lut.getQValue(
                    getHP(preState[0]).ordinal(),
                    getHP(preState[1]).ordinal(),
                    getEneDis(preState[2]).ordinal(),
                    getWallDis(preState[3]).ordinal(),
                    exp.getAction_t())});
        }

        double[][] input = new double[list.size()][list.get(0).length];
        for(int i = 0; i < list.size(); i++){
            input[i] = list.get(i);
        }
        input = rescaleInput(input);
        double[][] output = new double[out_list.size()][out_list.get(0).length];
        for(int i = 0; i < out_list.size(); i++){
            output[i] = out_list.get(i);
        }
        nn.train(1);
    }

    public double[][] rescaleInput(double[][] in) {
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[0].length; j++) {
                if (j <= 1) {
                    out[i][j] = RLNeuralNet.scaleRange(in[i][j], 0, 100, -1, 1);
                } else if (j == 2) {
                    out[i][j] = RLNeuralNet.scaleRange(in[i][j], 0, 1000, -1, 1);
                } else if (j == 3) {
                    out[i][j] = RLNeuralNet.scaleRange(in[i][j], 0, 300, -1, 1);
                } else {
                    out[i][j] = RLNeuralNet.scaleRange(in[i][j], 0, 4, -1, 1);
                }
            }
        }
        return out;
    }

    public int nnGetBestAction(double myHp, double enemyHp ,double enemyDis, double wallDis){
        double max = -1;
        int bestAction = -1;
        for(int i = 0; i < actionSize; i++){
            double[][] state = new double[][]{{
                    getHP(myHp).ordinal(),
                    getHP(enemyHp).ordinal(),
                    getEneDis(eneDis).ordinal(),
                    getWallDis(wallDis).ordinal(),
                    i
            }};
            double[] qval = nn.outputQVal(state);
            if(max < qval[0]){
                max = qval[0];
                bestAction = i;
            }
        }
        return bestAction;
    }

    // Event handler part
    public void onScannedRobot(ScannedRobotEvent e) {
        super.onScannedRobot(e);
        enemyBearing = e.getBearing();
        myX = getX();
        myY = getY();
        myHP = getEnergy();
        enemyHP = e.getEnergy();
        dis = e.getDistance();
        preMyHp = myHp;
        preEnemyHp = enemyHp;
        preEneDis = eneDis;
        preWallDis = wallDis;
        preAction = action;

        myHp = myHP;
        enemyHp = enemyHP;
        eneDis = dis;
        wallDis = getWallDDis(myX, myY);
        operation = RobotStates.Operation.action;
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e) {
        if(takeImmediate) {
            reward += immediatePenalty;
        }
    }

    @Override
    public void onBulletHit(BulletHitEvent e) {
        if(takeImmediate) {
            reward += immediateBonus;
        }
    }

    @Override
    public void onBulletMissed(BulletMissedEvent e) {
        if(takeImmediate){
            reward += immediatePenalty;
        }
    }

    @Override
    public void onHitWall(HitWallEvent e){
        if(takeImmediate) {
            reward += immediatePenalty;
        }
        avoidObstacle();
    }

    public void avoidObstacle() {
        setBack(200);
        setTurnRight(60);
        execute();
    }

    @Override
    public void onHitRobot(HitRobotEvent e) {
        if(takeImmediate) {
            reward += immediatePenalty;
        }
    }

    @Override
    public void onWin(WinEvent e){
        reward = terminalBonus;
        Q = getQ(reward, onPolicy);
        lut.setQValue(getHP(preMyHp).ordinal(),
                getHP(preEnemyHp).ordinal(),
                getEneDis(preEneDis).ordinal(),
                getWallDis(preWallDis).ordinal(),
                preAction.ordinal(),
                Q
        );
        winRound++;
        totalRound++;

        if((totalRound % winRateRound == 0) && (totalRound != 0)) {
            winPercentage = (double) winRound / winRateRound;
            System.out.println(String.format("%d, %.3f", ++round, winPercentage));
            File folderDst1 = getDataFile(fileToSaveName);
            log.writeToFile(folderDst1, winPercentage, round);
            winRound = 0;


            if(totalRound == 4000 && isSave){
                File argFile = getDataFile(fileToSaveLUT);
                lut.save(argFile);
            }

        }
    }

    @Override
    public void onDeath(DeathEvent e){
        reward = terminalPenalty;
        Q = getQ(reward, onPolicy);
        lut.setQValue(getHP(preMyHp).ordinal(),
                getHP(preEnemyHp).ordinal(),
                getEneDis(preEneDis).ordinal(),
                getWallDis(preWallDis).ordinal(),
                preAction.ordinal(),
                Q
        );
        totalRound++;

        if((totalRound % winRateRound == 0) && (totalRound != 0)) {
            winPercentage = (double) winRound / winRateRound;
            System.out.println(String.format("%d, %.3f", ++round, winPercentage));
            File folderDst1 = getDataFile(fileToSaveName);
            log.writeToFile(folderDst1, winPercentage, round);
            winRound = 0;
            if(totalRound == 4000 && isSave == true) {
                File argFile = getDataFile(fileToSaveLUT);
                lut.save(argFile);
            }
        }
    }

}


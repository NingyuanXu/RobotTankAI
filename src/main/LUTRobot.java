package main;

import robocode.*;

import java.awt.*;
import java.io.File;

public class LUTRobot extends AdvancedRobot {

    // Declare global variables
    private RobotStates.MyHP myHp = RobotStates.MyHP.high;
    private RobotStates.MyHP enemyHp = RobotStates.MyHP.high;
    private RobotStates.EneDis eneDis = RobotStates.EneDis.close;
    private RobotStates.WallDis wallDis = RobotStates.WallDis.far;
    private RobotStates.Action action = RobotStates.Action.forward;

    private RobotStates.MyHP preMyHp = RobotStates.MyHP.high;
    private RobotStates.MyHP preEnemyHp = RobotStates.MyHP.high;
    private RobotStates.EneDis preEneDis = RobotStates.EneDis.close;
    private RobotStates.WallDis preWallDis = RobotStates.WallDis.far;
    private RobotStates.Action preAction = RobotStates.Action.forward;

    private RobotStates.Operation operation = RobotStates.Operation.scan;

    public double myX = 0.0, myY = 0.0, myHP = 100, enemyHP = 100, dis = 0.0;
    public static boolean takeImmediate = true, onPolicy = true;
    private double gamma = 0.9, alpha = 0.1, epsilon = 0.0, Q = 0.0, reward = 0.0;
    private final double immediateBonus = 0.5, terminalBonus = 1.0, immediatePenalty = -0.1, terminalPenalty = -0.2;

    public static int curActionIndex;
    public static double enemyBearing;

    public static int totalRound = 0, round = 0, winRound = 0;
    public static double winPercentage = 0.0;
    public static String fileToSaveName = LUTRobot.class.getSimpleName() + "-" + "winningRate" + ".log";
    public static String fileToSaveLUT = "lut1.0.txt";
    static LogFile log = new LogFile();
    static int winRateRound = 100;

    public static LUT lut = new LUT();
    public static boolean isLoad = false, isSave = false;

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

    public double getQ(double reward, boolean onPolicy){
        double preQ = lut.getQValue(
                preMyHp.ordinal(),
                preEnemyHp.ordinal(),
                preEneDis.ordinal(),
                preWallDis.ordinal(),
                preAction.ordinal());
        double curQ = lut.getQValue(
                myHp.ordinal(),
                enemyHp.ordinal(),
                eneDis.ordinal(),
                wallDis.ordinal(),
                action.ordinal()
        );
        int bestAction = lut.getBestAction(
                myHp.ordinal(),
                enemyHp.ordinal(),
                eneDis.ordinal(),
                wallDis.ordinal()
        );
        double maxQ = lut.getQValue(
                myHp.ordinal(),
                enemyHp.ordinal(),
                eneDis.ordinal(),
                wallDis.ordinal(),
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
        myHp = RobotStates.MyHP.high;
        while (true) {
            switch(operation) {
                case scan: {
                    reward = 0.0;
                    turnRadarLeft(90);
                    break;
                }
                case action: {
                    wallDis = getWallDis(myX, myY);
                    curActionIndex = (Math.random() <= epsilon) ?
                            lut.getRandomAction() :
                            lut.getBestAction(
                                    getHP(myHP).ordinal(),
                                    getHP(enemyHP).ordinal(),
                                    getEneDis(dis).ordinal(),
                                    wallDis.ordinal()
                            );
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
                    lut.setQValue(preMyHp.ordinal(),
                            preEnemyHp.ordinal(),
                            preEneDis.ordinal(),
                            preWallDis.ordinal(),
                            preAction.ordinal(),
                            Q
                    );
                    operation = RobotStates.Operation.scan;
                    break;
                }
            }
        }
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

        myHp = getHP(myHP);
        enemyHp = getHP(enemyHP);
        eneDis = getEneDis(dis);
        wallDis = getWallDis(myX, myY);
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
        lut.setQValue(preMyHp.ordinal(),
                preEnemyHp.ordinal(),
                preEneDis.ordinal(),
                preWallDis.ordinal(),
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
        lut.setQValue(preMyHp.ordinal(),
                preEnemyHp.ordinal(),
                preEneDis.ordinal(),
                preWallDis.ordinal(),
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

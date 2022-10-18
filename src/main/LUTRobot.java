package main;

import robocode.*;
import robocode.util.Utils;

import java.awt.*;
import java.awt.geom.Point2D;
import java.io.File;

public class LUTRobot extends AdvancedRobot {

    // Declare game global variables
    private static double numWinsPerGroupRound = 0.0;
    private static int roundNumber = 1;

    // Reinforcement learning part
    private static LUT lut = new LUT(); // same LUT !!
    private QLearning agent;

    // Robot state/action variable, not static
    // Initialize each time for a new robot round
    private int currentAction;
    private int currentState;
    private int hasHitWall = 0;
    private int isHitByBullet = 0;
    private EnemyRobot enemyTank;

    // Reward policy
    private double currentReward = 0.0;
    private final double goodReward = 5.0;
    private final double badReward = -2.0;
    private final double winReward = 10;
    private final double loseReward = -10;

    public void run() {
        // Initialize robot tank parts
        // Contain code that only needs to be executed once per robot instance
        setBulletColor(Color.green);
        setGunColor(Color.red);
        setBodyColor(Color.pink);
        setRadarColor(Color.yellow);

        // Need to adjust radar based on the distance and direction of the enemy tank
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);

        // IMPORTANT: SET UP HYPER-PARAMETER HERE !!!!!
        agent = new QLearning(lut, 0.2, 0.9, 0.5, false);
        enemyTank = new EnemyRobot();
        RobotStates.initialEnergy = this.getEnergy();

        // Endless while loop that controls the normal behaviour of your robot
        while (true) {
            selectRobotActions();
            moveRadar();
            execute();
        }
    }

    @Override
    public void onScannedRobot (ScannedRobotEvent e){
        double absoluteBearing = (getHeading() + e.getBearing()) % (360) * Math.PI/180;
        enemyTank.bearing = e.getBearingRadians();
        enemyTank.heading = e.getHeadingRadians();
        enemyTank.distance = e.getDistance();
        enemyTank.xCoord = getX() + Math.sin(absoluteBearing) * e.getDistance();
        enemyTank.yCoord = getY() + Math.cos(absoluteBearing) * e.getDistance();
        circularTargeting(e);
    }

    // Circular targeting from Robocode Wiki
    // Reference: https://robowiki.net/wiki/Circular_Targeting
    public void circularTargeting(ScannedRobotEvent e) {
        double bulletPower = Math.min(3.0, getEnergy());
        double myX = getX();
        double myY = getY();
        double absoluteBearing = getHeadingRadians() + e.getBearingRadians();
        double enemyX = getX() + e.getDistance() * Math.sin(absoluteBearing);
        double enemyY = getY() + e.getDistance() * Math.cos(absoluteBearing);
        double oldEnemyHeading = 0.0;
        double enemyHeading = e.getHeadingRadians();
        double enemyHeadingChange = enemyHeading - oldEnemyHeading;
        double enemyVelocity = e.getVelocity();
        oldEnemyHeading = enemyHeading;

        double deltaTime = 0;
        double battleFieldHeight = getBattleFieldHeight(),
                battleFieldWidth = getBattleFieldWidth();
        double predictedX = enemyX, predictedY = enemyY;
        while((++deltaTime) * (20.0 - 3.0 * bulletPower) <
                Point2D.Double.distance(myX, myY, predictedX, predictedY)){
            predictedX += Math.sin(enemyHeading) * enemyVelocity;
            predictedY += Math.cos(enemyHeading) * enemyVelocity;
            enemyHeading += enemyHeadingChange;
            if(	predictedX < 18.0
                    || predictedY < 18.0
                    || predictedX > battleFieldWidth - 18.0
                    || predictedY > battleFieldHeight - 18.0){

                predictedX = Math.min(Math.max(18.0, predictedX),
                        battleFieldWidth - 18.0);
                predictedY = Math.min(Math.max(18.0, predictedY),
                        battleFieldHeight - 18.0);
                break;
            }
        }
        double theta = Utils.normalAbsoluteAngle(Math.atan2(
                predictedX - getX(), predictedY - getY()));

        setTurnRadarRightRadians(Utils.normalRelativeAngle(
                absoluteBearing - getRadarHeadingRadians()));
        setTurnGunRightRadians(Utils.normalRelativeAngle(
                theta - getGunHeadingRadians()));
        fire(3);
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        hasHitWall = 1;
        currentReward += badReward;
        if(euclideanDistance(getX(), getY(),enemyTank.xCoord, enemyTank.yCoord)>200){
            double degToEnemy= getBearingToTarget(enemyTank.xCoord, enemyTank.yCoord, getX(), getY(), getHeadingRadians());
            setTurnRightRadians(degToEnemy);
            setAhead(200);
            execute();
        }
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        currentReward += goodReward;
    }

    @Override
    public void onHitByBullet(HitByBulletEvent e){
        isHitByBullet = 1;
        currentReward -= e.getBullet().getPower();
        double degToEnemy= getBearingToTarget(enemyTank.xCoord, enemyTank.yCoord, getX(), getY(), getHeadingRadians());
        setTurnRightRadians(degToEnemy+2);
        setAhead(100);
        execute();
    }

    @Override
    public void onBulletMissed(BulletMissedEvent event) {
        currentReward += badReward;
    }

    @Override
    public void onHitRobot(HitRobotEvent e) {
        currentReward += badReward;
    }

    @Override
    public void onWin(WinEvent event) {
        currentReward += winReward;
        numWinsPerGroupRound++;
        agent.train(currentState, currentAction, currentReward);
    }

    @Override
    public void onDeath(DeathEvent event) {
        currentReward += loseReward;
        agent.train(currentState, currentAction, currentReward);
    }

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        if (roundNumber % 100 == 0) {
            writeWinRates();
            numWinsPerGroupRound = 0.0;
        }
        roundNumber ++;
        try
        {
            lut.save(getDataFile("LUT.txt"));
        }
        catch (Exception e)
        {
            out.println("Exception trying to write: " + e);
        }
    }

    // Function to decide the next action for the robot to take
    public void selectRobotActions() {
        int state = getRobotState();
        currentAction = agent.getAction(state);

        // IMPORTANT RL learning here
        agent.train(currentState, currentAction, currentReward);
        this.currentReward = 0; // reset reward here

        // reset the state
        this.hasHitWall = 0;
        this.isHitByBullet = 0;

        switch(currentAction) {
            case RobotActions.moveForward:
                setAhead(RobotActions.moveDistance);
                execute();
                break;
            case RobotActions.moveBack:
                setBack(RobotActions.moveDistance);
                execute();
                break;
            case RobotActions.forwardRight:
                setAhead(RobotActions.moveDistance);
                setTurnRight(90.0);
                execute();
                break;
            case RobotActions.forwardLeft:
                setAhead(RobotActions.moveDistance);
                setTurnLeft(90.0);
                execute();
                break;
            case RobotActions.backRight:
                setBack(RobotActions.moveDistance);
                setTurnRight(90.0);
                execute();
                break;
            case RobotActions.backLeft:
                setBack(RobotActions.moveDistance);
                setTurnLeft(90.0);
                execute();
                break;
        }
    }

    public void moveRadar() {
        if (getRadarTurnRemaining() == 0.0)
            setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
    }

    // Get the state index for the current robot, using RobotStates class to helper
    public int getRobotState() {
        int curDistance = RobotStates.getDistanceState(enemyTank.distance);
        int enemyBearing = RobotStates.getEnemyBearingState(Math.toDegrees(enemyTank.bearing));
        int curEnergy = RobotStates.getEnergyState(getEnergy());
        int curDirection = RobotStates.getDirectionState(getHeading());
        currentState = RobotStates.getStateIndex(curDirection, enemyBearing, curEnergy, isHitByBullet, hasHitWall, curDistance);
        return currentState;
    }

    // List of useful helper functions here
    private double euclideanDistance(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }

    private double normalize(double angle) {
        if (angle > Math.PI) angle -= 2*Math.PI;
        if (angle < -Math.PI) angle += 2*Math.PI;
        return angle;
    }

    private double getBearingToTarget(double x, double y, double myX, double myY, double myHeading){
        double deg = Math.PI/2 - Math.atan2(y - myY, x-myX);
        return  normalize(deg - myHeading);
    }

    private void writeWinRates() {
        double winRate = numWinsPerGroupRound / 100.0;
        System.out.println("\n\n" +"win rate"+ " " + winRate + "\n\n");
        File folder = getDataFile("winRate.txt");
        try{
            RobocodeFileWriter fileWriter = new RobocodeFileWriter(folder.getAbsolutePath(), true);
            fileWriter.write(Double.toString(winRate) + "\r\n");
            fileWriter.close();
        }
        catch(Exception e){
            System.out.println(e);
        }
    }
}

package main;

public class RobotStates {

    public static int maxDistance = 1000; // 1000 is the maximum distance in the game
    public static double initialEnergy; // initial energy in the game

    public static final int numDirections = 4;
    public static final int numEnemyBearing = 4;
    public static final int numEnergy = 4; // low, less than half , more than half, high
    public static final int hitWall = 2; // hit or wall
    public static final int hitByBullet = 2; // be shot or not
    public static final int numEnemyDistance = 10; // max is 1000

    public static int numStates = 0;
    public static int stateMap[][][][][][] = new int[numEnemyDistance][numDirections][numEnemyBearing][numEnergy][hitWall][hitByBullet];

    static {
        for (int a = 0; a < numEnemyDistance; a++)
            for (int b = 0; b < numDirections; b++)
                        for (int c = 0; c < numEnemyBearing; c++)
                            for (int d = 0; d < numEnergy; d++)
                                for (int e = 0; e < hitWall; e++)
                                    for (int f = 0; f < hitByBullet; f++)
                                        stateMap[a][b][c][d][e][f] = numStates++;
    }

    public static int getStateIndex(int direction, int bearing, int energy, int bullet, int wall, int distance) {
        return stateMap[distance][direction][bearing][energy][wall][bullet];
    }

    // Input is in degree 0 to 360 degrees
    public static int getDirectionState(double eventHeading) {
        if ((eventHeading >= 0.0 && eventHeading < 45.0) || (eventHeading <= 360.0 && eventHeading > 315.0)) {
            return 0;
        } else if (eventHeading >= 45.0 && eventHeading < 135.0) {
            return 1;
        } else if (eventHeading >= 135.0 && eventHeading < 225.0) {
            return 3;
        } else {
            return 2;
        }
    }

    public static int getEnemyBearingState(double eventEnemyBearing) {
        eventEnemyBearing = eventEnemyBearing < 0.0 ? eventEnemyBearing + 360.0 : eventEnemyBearing;
        double bearing = eventEnemyBearing + 45.0;
        if ((bearing >= 0.0 && bearing < 45.0) || (bearing <= 360.0 && bearing > 315.0)) {
            return 0;
        } else if (bearing >= 45.0 && bearing < 135.0) {
            return 1;
        } else if (bearing >= 135.0 && bearing < 225.0) {
            return 3;
        } else {
            return 2;
        }
    }

    public static int getDistanceState(double eventDistance) {
        int distanceRanges = maxDistance / numEnemyDistance;
        int currentDistance = (int) eventDistance / distanceRanges;
        return (currentDistance < numEnemyDistance) ? currentDistance: numEnemyDistance - 1;
    }

    public static int getEnergyState(double eventRobotEnergy) {
        if (eventRobotEnergy < 25.0) {
            return 0;
        } else if (eventRobotEnergy < 50.0) {
            return 1;
        } else if (eventRobotEnergy < 75.0) {
            return 2;
        } else {
            return 3;
        }
    }
}

package main;

public class RobotStates {

    public static int maxDistance = 1000; // 1000 is the maximum distance in the game
    public static double initialEnergy; // initial energy in the game

    public static final int numDirections = 4; // head, right, back, left
    public static final int numEnemyBearing = 4; // same
    public static final int numEnergy = 4; // low, less than half , more than half, high
    public static final int hitWall = 2; // hit or wall
    public static final int hitByBullet = 2; // be shot or not
    public static final int numEnemyDistance = 10; // max is 1000

    public static final int numStates = numDirections * numEnemyBearing * numEnergy * hitByBullet * hitWall * numEnemyDistance;

    public static int getStateIndex(int direction, int bearing, int energy, int bullet, int wall, int distance) {
        int index = distance + wall * numEnemyDistance + bullet * numEnemyDistance * hitWall +
                    energy * numEnemyDistance * hitWall * hitByBullet + bearing * numEnemyDistance * hitWall * hitByBullet * numEnergy +
                    direction * numEnemyDistance * hitWall * hitByBullet * numEnergy * numEnemyBearing;
        assert(index < numStates);
        return index;
    }

    // Divide 360 degree into four directions, up, down, right, left
    public static int getDirectionState(double eventHeading) {
        double heading = eventHeading + 45.0;
        return (int)(heading > 360.0 ? ((heading - 360.0) / 90.0) : (heading / 90.0));
    }

    // Bearings can be negative degrees
    public static int getEnemyBearingState(double eventEnemyBearing) {
        eventEnemyBearing = eventEnemyBearing < 0.0 ? eventEnemyBearing + 360.0 : eventEnemyBearing;
        double bearing = eventEnemyBearing + 45.0;
        return (int)(bearing > 360.0 ? ((bearing - 360.0) / 90.0) : (bearing / 90.0));
    }

    public static int getDistanceState(double eventDistance) {
        int distance = (int)eventDistance / (maxDistance / numEnemyDistance);
        return distance == numEnemyDistance ? numEnemyDistance - 1 : distance;
    }

    public static int getEnergyState(double eventRobotEnergy) {
        int energy = (int)(eventRobotEnergy / (initialEnergy / (double)numEnergy));
        return energy == numEnergy ? numEnergy - 1 : energy;
    }
}

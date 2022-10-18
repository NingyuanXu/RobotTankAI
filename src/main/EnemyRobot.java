package main;

public class EnemyRobot {
    public double heading;
    public double bearing;
    public double distance;
    public double xCoord;
    public double yCoord;

    public EnemyRobot() {}

    public EnemyRobot(double heading, double bearing, double distance, double xCoord, double yCoord) {
        this.heading = heading;
        this.bearing = bearing;
        this.distance = distance;
        this.xCoord = xCoord;
        this.yCoord = yCoord;
    }
}

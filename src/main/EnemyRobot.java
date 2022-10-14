package main;

public class EnemyRobot {
    public double heading;
    public double velocity;
    public double bearing;
    public double distance;
    public double xCoord;
    public double yCoord;
    public double energy;
    public double time;

    public EnemyRobot() {}

    public EnemyRobot(double heading, double velocity, double bearing, double distance, double xCoord, double yCoord, double energy, double time) {
        this.heading = heading;
        this.velocity = velocity;
        this.bearing = bearing;
        this.distance = distance;
        this.xCoord = xCoord;
        this.yCoord = yCoord;
        this.energy = energy;
        this.time = time;
    }
}

package main;

import robocode.*;

public class LUTRobot extends AdvancedRobot {

    // Declare global variables

    public void run() {
        // Initialize robot tank parts
        // Contain code that only needs to be executed once per robot instance
        while (true) {
            // Endless while loop that controls the normal behaviour of your robot
            ahead(100);
            turnGunRight(360);
            back(100);
            turnGunRight(360);
        }
    }

    // Event handler part
    public void onScannedRobot(ScannedRobotEvent e) {
        fire(1);
        e.getBearing();
    }

    public void onHitByBullet(HitByBulletEvent e) {
        turnLeft(90 - e.getBearing());
    }
}

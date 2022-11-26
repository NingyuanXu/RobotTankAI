package main;

public class RobotStates {
    public enum MyHP {low,medium,high};
    public enum EneDis {close, medium, far};
    public enum WallDis {close, medium, far};
    public enum Action {fire, forward, backward, left, right};
    public enum Operation {scan, action};

    public static int statesCount = MyHP.values().length * MyHP.values().length * EneDis.values().length * WallDis.values().length;
    public static int totalCount = MyHP.values().length * MyHP.values().length * EneDis.values().length * WallDis.values().length * Action.values().length;

}

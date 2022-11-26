package main;


public class Experience {
    private double[] state_t;
    private int action_t;
    private double reward_t;
    private double[] state_t1;

    public Experience(double[] st, int at, double rt, double[] st1){
        this.state_t = st;
        this.action_t = at;
        this.reward_t = rt;
        this.state_t1 = st1;
    }

    public double[] getState_t(){
        return this.state_t;
    }

    public int getAction_t(){
        return this.action_t;
    }

    public double getReward_t(){
        return this.reward_t;
    }

    public double[] getState_t1(){
        return this.state_t1;
    }
}
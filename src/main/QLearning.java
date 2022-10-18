package main;

public class QLearning {

    private int prevState = -1; // initialize them
    private int prevAction = -1;
    private boolean isOnPolicy; // Q-learn is OFF-policy; Sarsa is ON-policy
    LUT lookUpTable;

    // Hyper-parameter
    private double learningRate;
    private double discountRate;
    private double epsilon;

    public QLearning(LUT lookUpTable, double learningRate, double discountRate, double epsilon, boolean isOnPolicy) {
        this.lookUpTable = lookUpTable;
        this.learningRate = learningRate;
        this.discountRate = discountRate;
        this.epsilon = epsilon;
        this.isOnPolicy = isOnPolicy;
    }

    public void train(int state, int action, double reward) {
        // if state and action not initialize yet, initialize first
        if (prevState != -1 && prevAction != -1) {
            double QVal = this.lookUpTable.getLookUpTableValue(prevState, prevAction);
            if (isOnPolicy) {
                // Sarsa on-policy
                QVal += this.learningRate * (reward + this.discountRate * this.lookUpTable.getLookUpTableValue(state, action) - QVal) ;
            } else {
                // Q-Learning off-policy
                QVal += this.learningRate * (reward + this.discountRate * this.lookUpTable.getMaxLUTActionAndQVal(state)[1] - QVal);
            }
            this.lookUpTable.setLookUpTableValue(QVal, prevState, prevAction);
        }
        prevState = state;
        prevAction = action;
    }

    public int getAction(int state) {
        // when e is 0.5, 1/2 probability to randomly pick an action
        // feel free to comment out code if not want to test randomness
        // selected randomly with probability e and greedily with probability 1 - e
        if (Math.random() > this.epsilon) {
            return (int)this.lookUpTable.getMaxLUTActionAndQVal(state)[0];
        } else {
            return (int)(Math.random() * LUT.numActions);
        }
    }

}

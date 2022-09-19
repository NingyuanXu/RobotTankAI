package main;

import java.io.IOException;
import java.util.List;

public interface CommonInterface {
    void train(int epochNum);

    void save(List<Double> listOfErrors) throws IOException;
}

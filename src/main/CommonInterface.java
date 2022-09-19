package main;

import java.io.IOException;
import java.util.List;

public interface CommonInterface {
    int train(int epochNum);

    void save(List<Double> listOfErrors) throws IOException;
}

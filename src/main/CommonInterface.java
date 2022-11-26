package main;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    public void save(File argFile);
    public void load(File argFileName) throws IOException;
}

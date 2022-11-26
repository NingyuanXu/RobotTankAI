package main;

import robocode.RobocodeFileWriter;

import java.io.File;

public class LogFile {
    public void writeToFile(File fileToWrite, double winRate, int roundCount){
        try{
            RobocodeFileWriter fileWriter = new RobocodeFileWriter(fileToWrite.getAbsolutePath(), true);
            fileWriter.write(roundCount + " " + Double.toString(winRate) + "\r\n");
            fileWriter.close();
        }
        catch(Exception e){
            System.out.println(e);
        }
    }

    public void writeToFileLoss(File fileToWrite, double loss){
        try{
            RobocodeFileWriter fileWriter = new RobocodeFileWriter(fileToWrite.getAbsolutePath(), true);
            fileWriter.write(loss + System.lineSeparator());
            fileWriter.close();
        }
        catch(Exception e){
            System.out.println(e);
        }
    }
}

package automation.util;

import java.io.FileWriter;
import java.io.IOException;

public class FileUtils {

    public static void writeToFile(String content, String fileName) {
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


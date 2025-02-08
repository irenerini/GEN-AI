package automation.pipeline;

import automation.generator.LLMTestGenerator;
import automation.generator.TestCodeGenerator;
import automation.util.FileUtils;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TestAutomationPipeline {
	
	private static String scode = "\\\"WebDriverdriver=newChromeDriver();driver.get(\\\"https://login.salesforce.com/?locale=in\\\");Wait<WebDriver>wait=newWebDriverWait(driver,Duration.ofSeconds(2));WebElementusernameInput=wait.until(ExpectedConditions.elementToBeClickable(By.xpath(\\\"//input[@name='username']\\\")));usernameInput.sendKeys(\\\"your_username\\\");WebElementpasswordInput=wait.until(ExpectedConditions.elementToBeClickable(By.xpath(\\\"//input[@type='password']\\\")));passwordInput.sendKeys(\\\"your_password\\\");WebElementrememberMeCheckbox=wait.until(ExpectedConditions.elementToBeClickable(By.xpath(\\\"//input[@type='checkbox']\\\")));rememberMeCheckbox.click();WebElementloginButton=wait.until(ExpectedConditions.elementToBeClickable(By.xpath(\\\"//input[@type='submit']\\\")));loginButton.click();\\\"";


    public static void main(String[] args) {

        // 2. Generate test case outline using LLM
        LLMTestGenerator llmGen = new LLMTestGenerator();
        String llmRawOutput = llmGen.generateTestCases(scode);

        // 3. Extract only the Java code from the LLM response
        TestCodeGenerator codeGen = new TestCodeGenerator();
        String finalTestCode = codeGen.extractJavaCode(llmRawOutput);

        // 4. Extract the class name from the generated Java code
        String className = extractClassName(finalTestCode);
        String outputFileName = (className != null && !className.isEmpty()) 
                                ? className + ".java" 
                                : "GeneratedPlaywrightTest.java";

        // 5. Write generated code to a Java file with a name matching the class name
        FileUtils.writeToFile(finalTestCode, "src/main/java/automation/tests/"+outputFileName);

        System.out.println("Generated test code written to " + outputFileName);
    }

    /**
     * Extracts the Java class name from the given code.
     * It searches for a pattern matching "public class <ClassName>".
     */
    private static String extractClassName(String javaCode) {
        Pattern pattern = Pattern.compile("public\\s+class\\s+(\\w+)");
        Matcher matcher = pattern.matcher(javaCode);
        if (matcher.find()) {
            return matcher.group(1);
        }
        return null;
    }
}

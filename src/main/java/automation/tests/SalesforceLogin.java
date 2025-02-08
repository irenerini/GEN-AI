package automation.tests;

import com.microsoft.playwright.*;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class SalesforceLogin {
    public static void main(String[] args) {
        // Launch browser in non-headless mode
        Playwright playwright = Playwright.create();
        BrowserType chromium = playwright.chromium();
        Browser browser = chromium.launch(new BrowserType.LaunchOptions().setHeadless(false));
        BrowserContext context = browser.newContext();
        Page page = context.newPage();

        // Navigate to Salesforce login page
        page.navigate("https://login.salesforce.com/?locale=in");

        // Wait for username input to be clickable
        page.waitForSelector("//input[@name='username']");

        // Enter username
        page.fill("//input[@name='username']", "your_username");

        // Wait for password input to be clickable
        page.waitForSelector("//input[@type='password']");

        // Enter password
        page.fill("//input[@type='password']", "your_password");

        // Wait for remember me checkbox to be clickable
        page.waitForSelector("//input[@type='checkbox']");

        // Click remember me checkbox
        page.click("//input[@type='checkbox']");

        // Wait for login button to be clickable
        page.waitForSelector("//input[@type='submit']");

        // Click login button
        page.click("//input[@type='submit']");

        // Take screenshot before quitting browser
        page.screenshot(new Page.ScreenshotOptions().setPath(Paths.get("salesforce_login.png")));

        // Quit browser
        browser.close();
        playwright.close();
    }
}
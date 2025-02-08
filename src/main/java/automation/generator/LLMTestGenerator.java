package automation.generator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import com.fasterxml.jackson.databind.ObjectMapper;

public class LLMTestGenerator {

	private static final String LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions";
	private static final String API_KEY = "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";

	public static String generateTestCases(String seleniumCode) {

		if (seleniumCode == null || seleniumCode.isEmpty()) {
			return "No valid selenium code details to convert to playwright";
		}

		String userPrompt = "Convert selenium code " + seleniumCode
				+ " to playwright code using latest microsoft playwright version";

		try {
			List<Map<String, String>> messages = new ArrayList<>();
			Map<String, String> systemMessage = new HashMap<>();
			systemMessage.put("role", "system");
			systemMessage.put("content",
					"You are a helpful assistant that converts Selenium code to Microsoft Playwright code with latest playwright version. "
							+ "- Launch browser in non-headless mode"
							+ "- Your response must contain only Playwright code enclosed in a single code block using triple backticks (```java ... ```). "
							+ "- Use only standard and correct imports (e.g.,import com.microsoft.playwright.*, import com.microsoft.playwright.BrowserType.LaunchOptions;). "
							+ "- Add package as automation.tests" + "- Write comments on the code"
							+ "- take screenshot before quitting browser, create png file, store image and import file path from java"
							+ "- Quit Browser");

			messages.add(systemMessage);
			Map<String, String> userMessage = new HashMap<>();
			userMessage.put("role", "user");
			userMessage.put("content", userPrompt);
			messages.add(userMessage);
			Map<String, Object> payload = new HashMap<>();
			payload.put("model", "llama-3.3-70b-specdec");
			payload.put("messages", messages);
			payload.put("temperature", 0.02);
			payload.put("max_tokens", 1000);
			payload.put("top_p", 0.1);
			ObjectMapper mapper = new ObjectMapper();
			String requestBody = mapper.writeValueAsString(payload);
			return callLLMApi(requestBody);
		} catch (Exception e) {
			e.printStackTrace();
			return "Error building JSON payload: " + e.getMessage();
		}
	}

	private static String callLLMApi(String requestBody) {
		try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
			HttpPost request = new HttpPost(LLM_API_URL);
			request.setHeader("Content-Type", "application/json");
			request.setHeader("Authorization", "Bearer " + API_KEY);
			request.setEntity(new StringEntity(requestBody));
			try (CloseableHttpResponse response = httpClient.execute(request)) {
				return EntityUtils.toString(response.getEntity());
			}
		} catch (Exception e) {
			e.printStackTrace();
			return "Error calling LLM API: " + e.getMessage();
		}
	}
}

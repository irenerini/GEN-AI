package automation.generator;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class TestCodeGenerator {

	public String extractJavaCode(String llmResponse) {
		try {
			ObjectMapper mapper = new ObjectMapper();
			JsonNode rootNode = mapper.readTree(llmResponse);
			JsonNode choicesNode = rootNode.path("choices");
			if (choicesNode.isArray() && choicesNode.size() > 0) {
				JsonNode messageNode = choicesNode.get(0).path("message");
				String content = messageNode.path("content").asText().trim();
				if (content.contains("```java")) {
					int start = content.indexOf("```java");
					int end = content.lastIndexOf("```");
					if (start != -1 && end != -1 && end > start) {
						content = content.substring(start + "```java".length(), end).trim();
					}
				} else if (content.contains("package")) {
					int index = content.indexOf("package");
					content = content.substring(index).trim();
				}
				return addMissingImports(content);
			}
		} catch (

		Exception e) {
			e.printStackTrace();
		}
		return convertToJavaCode(llmResponse);
	}

	public String convertToJavaCode(String extractedCode) {
		return addMissingImports(extractedCode);
	}

	/**
	 * Ensures all required Rest Assured and Hamcrest imports are included in the
	 * generated code.
	 */
	private String addMissingImports(String javaCode) {
		String requiredImports = """
								import com.microsoft.playwright.*;
								import com.microsoft.playwright.BrowserType.LaunchOptions;
								import com.microsoft.playwright.options.LoadState;
								import com.microsoft.playwright.Page;
								import com.microsoft.playwright.options.WaitForSelectorState;
								import com.microsoft.playwright.Browser;
								  """;

		if (!javaCode.contains("import com.microsoft.playwright.*")) {
			javaCode = requiredImports + "\n" + javaCode;
		}
		return javaCode;
	}
}

using OpenAI;
using OpenAI.Chat;
using System.Diagnostics;
using System.ClientModel;
using System.Text.Json;

if (args.Length < 2 || args[0] != "-p")
{
    throw new Exception("Usage: program -p <prompt>");
}

var prompt = args[1];

if (string.IsNullOrEmpty(prompt))
{
    throw new Exception("Prompt must not be empty");
}

var apiKey = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY");
var baseUrl = Environment.GetEnvironmentVariable("OPENROUTER_BASE_URL") ?? "https://openrouter.ai/api/v1";

if (string.IsNullOrEmpty(apiKey))
{
    throw new Exception("OPENROUTER_API_KEY is not set");
}

var client = new ChatClient(
    model: "anthropic/claude-haiku-4.5",
    credential: new ApiKeyCredential(apiKey),
    options: new OpenAIClientOptions { Endpoint = new Uri(baseUrl) }
);

var readToolSchema = JsonSerializer.Serialize(new
{
    type = "object",
    properties = new
    {
        file_path = new
        {
            type = "string",
            description = "The path to the file to read"
        }
    },
    required = new[] { "file_path" }
});

var writeToolSchema = JsonSerializer.Serialize(new
{
    type = "object",
    required = new[] { "file_path", "content" },
    properties = new
    {
        file_path = new
        {
            type = "string",
            description = "The path of the file to write to"
        },
        content = new
        {
            type = "string",
            description = "The content to write to the file"
        }
    }
});

var bashToolSchema = JsonSerializer.Serialize(new
{
    type = "object",
    required = new[] { "command" },
    properties = new
    {
        command = new
        {
            type = "string",
            description = "The command to execute"
        }
    }
});

var completionOptions = new ChatCompletionOptions();
completionOptions.Tools.Add(
    ChatTool.CreateFunctionTool(
        functionName: "Read",
        functionDescription: "Read and return the contents of a file",
        functionParameters: BinaryData.FromString(readToolSchema)
    )
);
completionOptions.Tools.Add(
    ChatTool.CreateFunctionTool(
        functionName: "Write",
        functionDescription: "Write content to a file",
        functionParameters: BinaryData.FromString(writeToolSchema)
    )
);
completionOptions.Tools.Add(
    ChatTool.CreateFunctionTool(
        functionName: "Bash",
        functionDescription: "Execute a shell command",
        functionParameters: BinaryData.FromString(bashToolSchema)
    )
);

var messages = new List<ChatMessage>
{
    new UserChatMessage(prompt)
};

const int maxAgentIterations = 20;

for (var iteration = 0; iteration < maxAgentIterations; iteration++)
{
    ChatCompletion response = client.CompleteChat(messages, completionOptions);

    // Preserve the assistant turn (including any tool calls) in conversation history.
    messages.Add(new AssistantChatMessage(response));

    if (response.ToolCalls == null || response.ToolCalls.Count == 0)
    {
        if (response.Content == null || response.Content.Count == 0)
        {
            throw new Exception("No choices in response");
        }

        Console.Write(response.Content[0].Text);
        return;
    }

    foreach (var toolCall in response.ToolCalls)
    {
        var toolResult = ExecuteToolCall(toolCall);
        messages.Add(new ToolChatMessage(toolCall.Id, toolResult));
    }
}

throw new Exception($"Agent loop exceeded maximum iterations ({maxAgentIterations})");

static string ExecuteToolCall(ChatToolCall toolCall)
{
    if (string.Equals(toolCall.FunctionName, "Read", StringComparison.OrdinalIgnoreCase))
    {
        return ExecuteRead(toolCall);
    }

    if (string.Equals(toolCall.FunctionName, "Write", StringComparison.OrdinalIgnoreCase))
    {
        return ExecuteWrite(toolCall);
    }

    if (string.Equals(toolCall.FunctionName, "Bash", StringComparison.OrdinalIgnoreCase))
    {
        return ExecuteBash(toolCall);
    }

    throw new Exception($"Unsupported tool call: {toolCall.FunctionName}");
}

static string ExecuteRead(ChatToolCall toolCall)
{
    using var argumentsJson = JsonDocument.Parse(toolCall.FunctionArguments.ToString());
    if (!argumentsJson.RootElement.TryGetProperty("file_path", out var filePathElement) ||
        string.IsNullOrWhiteSpace(filePathElement.GetString()))
    {
        throw new Exception("Read tool call must include a non-empty file_path");
    }

    var filePath = filePathElement.GetString()!;
    return File.ReadAllText(filePath);
}

static string ExecuteWrite(ChatToolCall toolCall)
{
    using var argumentsJson = JsonDocument.Parse(toolCall.FunctionArguments.ToString());

    if (!argumentsJson.RootElement.TryGetProperty("file_path", out var filePathElement) ||
        string.IsNullOrWhiteSpace(filePathElement.GetString()))
    {
        throw new Exception("Write tool call must include a non-empty file_path");
    }

    if (!argumentsJson.RootElement.TryGetProperty("content", out var contentElement) ||
        contentElement.ValueKind != JsonValueKind.String)
    {
        throw new Exception("Write tool call must include string content");
    }

    var filePath = filePathElement.GetString()!;
    var content = contentElement.GetString()!;

    var directoryPath = Path.GetDirectoryName(filePath);
    if (!string.IsNullOrWhiteSpace(directoryPath))
    {
        Directory.CreateDirectory(directoryPath);
    }

    File.WriteAllText(filePath, content);
    return $"Wrote content to {filePath}";
}

static string ExecuteBash(ChatToolCall toolCall)
{
    using var argumentsJson = JsonDocument.Parse(toolCall.FunctionArguments.ToString());

    if (!argumentsJson.RootElement.TryGetProperty("command", out var commandElement) ||
        commandElement.ValueKind != JsonValueKind.String ||
        string.IsNullOrWhiteSpace(commandElement.GetString()))
    {
        throw new Exception("Bash tool call must include a non-empty command");
    }

    var command = commandElement.GetString()!;
    var processStartInfo = new ProcessStartInfo
    {
        WorkingDirectory = Environment.CurrentDirectory,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
    };

    if (OperatingSystem.IsWindows())
    {
        processStartInfo.FileName = "powershell.exe";
        processStartInfo.ArgumentList.Add("-NoProfile");
        processStartInfo.ArgumentList.Add("-Command");
        processStartInfo.ArgumentList.Add(command);
    }
    else
    {
        processStartInfo.FileName = File.Exists("/bin/bash") ? "/bin/bash" : "/bin/sh";
        processStartInfo.ArgumentList.Add("-c");
        processStartInfo.ArgumentList.Add(command);
    }

    using var process = new Process();
    process.StartInfo = processStartInfo;
    process.Start();

    var standardOutputTask = process.StandardOutput.ReadToEndAsync();
    var standardErrorTask = process.StandardError.ReadToEndAsync();

    const int timeoutMilliseconds = 30_000;
    if (!process.WaitForExit(timeoutMilliseconds))
    {
        try
        {
            process.Kill(entireProcessTree: true);
        }
        catch
        {
            // Ignore kill failures and return timeout context.
        }

        return "Command timed out after 30 seconds.";
    }

    Task.WaitAll([standardOutputTask, standardErrorTask]);
    var standardOutput = standardOutputTask.Result;
    var standardError = standardErrorTask.Result;

    if (process.ExitCode != 0)
    {
        var errorOutput = string.IsNullOrWhiteSpace(standardError)
            ? standardOutput
            : standardError;
        return $"Command failed with exit code {process.ExitCode}.\n{errorOutput}".TrimEnd();
    }

    return (standardOutput + standardError).TrimEnd('\r', '\n');
}


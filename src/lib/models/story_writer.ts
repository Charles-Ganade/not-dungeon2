import { ProviderRegistry, ChatParams, StreamedChunk } from "../ai/provider";
import { STORY_WRITER_PROMPT } from "../prompts";

export class StoryWriter {
    private providerRegistry: ProviderRegistry;

    constructor(providerRegistry: ProviderRegistry) {
        this.providerRegistry = providerRegistry;
    }

    public async *writeNextTurnStream(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): AsyncGenerator<StreamedChunk> {
        const systemPrompt = STORY_WRITER_PROMPT;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context },
            ],
            options: options,
            tool_choice: "none",
        };

        const result = this.providerRegistry.chatStream(params);

        for await (const chunk of result) {
            yield chunk;
        }
    }
}

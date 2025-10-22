import { ProviderRegistry, ChatParams, StreamedChunk } from "../ai/provider";

/**
 * The StoryWriter component, responsible for generating the narrative text.
 */
export class StoryWriter {
    private providerRegistry: ProviderRegistry;

    /**
     * @param providerRegistry The central registry for AI provider access.
     */
    constructor(providerRegistry: ProviderRegistry) {
        this.providerRegistry = providerRegistry;
    }

    /**
     * Generates the next turn of the story using the active AI provider.
     * @param context The formatted context string (history, state, memories, director notes).
     * @param model The specific model name to use for story writing.
     * @param options Optional parameters like temperature for the LLM call.
     * @returns The generated story text content.
     */
    public async writeNextTurn(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): Promise<string> {
        const systemPrompt = `You are a creative story writer. Continue the narrative based on the provided context, focusing on engaging prose and character actions/dialogue. Describe the scene and what happens next.`;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context }, // The user role holds the context bundle
            ],
            options: options,
            // StoryWriter typically does *not* use tools or forced JSON format.
            tool_choice: "none",
        };

        const response = await this.providerRegistry.chat(params); //
        return response.message.content;
    }

    /**
     * Generates the next turn of the story as a stream using the active AI provider.
     * @param context The formatted context string.
     * @param model The specific model name to use.
     * @param options Optional parameters for the LLM call.
     * @returns An async generator yielding stream chunks.
     */
    public async *writeNextTurnStream(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): AsyncGenerator<StreamedChunk> {
        const systemPrompt = `You are a creative story writer. Continue the narrative based on the provided context, focusing on engaging prose and character actions/dialogue. Describe the scene and what happens next.`;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context },
            ],
            options: options,
            tool_choice: "none",
        };

        // Yield each chunk as it comes from the provider registry's stream method
        for await (const chunk of this.providerRegistry.chatStream(params)) {
            //
            yield chunk;
        }
    }
}

import { ProviderRegistry, ChatParams, StreamedChunk } from "../ai/provider";

export class StoryWriter {
    private providerRegistry: ProviderRegistry;

    constructor(providerRegistry: ProviderRegistry) {
        this.providerRegistry = providerRegistry;
    }

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
                { role: "user", content: context },
            ],
            options: options,
            tool_choice: "none",
        };

        const response = await this.providerRegistry.chat(params); //
        return response.message.content;
    }

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

        for await (const chunk of this.providerRegistry.chatStream(params)) {
            yield chunk;
        }
    }
}

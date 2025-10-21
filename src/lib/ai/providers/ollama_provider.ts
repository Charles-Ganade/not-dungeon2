import { Message, Ollama } from "ollama";
import {
    AIProvider,
    ChatMessage,
    ChatParams,
    ChatResponse,
    ChatRoles,
    EmbedParams,
    EmbedResponse,
    StreamedChunk,
} from "../provider";

export class OllamaProvider extends AIProvider {
    name = "ollama";
    client: Ollama;

    constructor() {
        super();
        this.client = new Ollama({ host: "http://localhost:11434" });
    }

    private standardMessageToOllama(message: ChatMessage): Message {
        return {
            ...message,
            tool_calls: message.tool_calls?.map((tool) => ({
                type: "function",
                function: {
                    name: tool.function.name,
                    arguments: JSON.parse(tool.function.arguments),
                },
            })),
        };
    }

    private ollamaMessageToStandard(message: Message): ChatMessage {
        return {
            ...message,
            role: message.role as ChatRoles,
            tool_calls: message.tool_calls?.map((tool) => ({
                type: "function",
                function: {
                    name: tool.function.name,
                    arguments: JSON.stringify(tool.function.arguments),
                },
            })),
        };
    }

    async chat(params: ChatParams): Promise<ChatResponse> {
        const response = await this.client.chat({
            model: params.model,
            messages: params.messages.map(this.standardMessageToOllama),
            options: {
                ...params.options,
                num_ctx: params.options?.max_completion_tokens,
            },
        });

        const message = response.message;

        return {
            message: this.ollamaMessageToStandard(message),
            model_name: response.model,
            provider_name: this.name,
            raw: response,
        };
    }

    async *chatStream(params: ChatParams): AsyncGenerator<StreamedChunk> {
        const stream = await this.client.chat({
            model: params.model,
            messages: params.messages.map((msg) => ({
                ...msg,
                tool_calls: msg.tool_calls?.map((func) => ({
                    type: "function",
                    function: {
                        name: func.function.name,
                        arguments: JSON.parse(func.function.arguments),
                    },
                })),
            })),
            options: {
                ...params.options,
                num_ctx: params.options?.max_completion_tokens,
            },
            stream: true,
        });

        for await (const chunk of stream) {
            yield {
                delta: this.ollamaMessageToStandard(chunk.message),
                done: chunk.done,
                provider_name: this.name,
                model_name: chunk.model,
                raw: chunk,
            };
        }
    }

    async embed(params: EmbedParams): Promise<EmbedResponse> {
        const response = await this.client.embed({ ...params, keep_alive: -1 });
        return {
            embeddings: response.embeddings,
            model_name: response.model,
            provider_name: this.name,
            raw: response,
        };
    }
}

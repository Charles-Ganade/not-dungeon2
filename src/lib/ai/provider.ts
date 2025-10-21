import { JSONSchema } from "json-schema-typed";
export interface ToolDefinition {
    type: "function";
    function: {
        name: string;
        description: string;
        parameters: JSONSchema;
        strict: boolean;
    };
}

const tool: ToolDefinition = {
    type: "function",
    function: {
        name: "get_weather",
        description: "Get the current weather for a location.",
        parameters: {
            type: "object",
            properties: {
                location: {
                    type: "string",
                    description: "The location to get the weather from.",
                },
            },
            required: ["location"],
        },
        strict: false,
    },
};

export type FunctionCall = {
    type: "function";
    function: {
        name: string;
    };
};

export type ChatRoles = "system" | "user" | "assistant" | "tool";

export interface ChatMessage {
    role: ChatRoles;
    content: string;
    thinking?: string;
    tool_calls?: {
        type: "function";
        function: {
            name: string;
            arguments: string; // JSON stringified
        };
    }[];
}

export interface ChatParams {
    model: string;
    messages: ChatMessage[];
    think?: boolean;
    tools?: ToolDefinition[];
    tool_choice?:
        | "none"
        | "auto"
        | "required"
        | FunctionCall
        | {
              type: "allowed_tools";
              allowed_tools: {
                  mode: "auto" | "required";
                  tools: FunctionCall[];
              };
          };
    options?: {
        frequency_penalty?: number;
        max_completion_tokens?: number;
        presence_penalty?: number;
        repeat_penalty?: number;
        seed?: number;
        temperature?: number;
        top_k?: number;
        top_p?: number;
    };
    format?: "json" | JSONSchema;
}

export interface EmbedParams {
    model: string;
    input: string | string[];
}

export interface StandardResponse {
    provider_name: string;
    model_name: string;
    raw: any;
}

export interface ChatResponse extends StandardResponse {
    message: ChatMessage;
}

export interface StreamedChunk extends StandardResponse {
    delta: ChatMessage;
    done: boolean;
}

export interface EmbedResponse extends StandardResponse {
    embeddings: number[][];
}

export abstract class AIProvider {
    abstract readonly name: string;

    abstract chat(params: ChatParams): Promise<ChatResponse>;

    abstract chatStream(params: ChatParams): AsyncGenerator<StreamedChunk>;

    abstract embed(params: EmbedParams): Promise<EmbedResponse>;
}

export class ProviderRegistry {
    private providers = new Map<string, AIProvider>();
    private active?: string | null = null;

    public register(provider: AIProvider) {
        if (this.providers.has(provider.name)) {
            console.warn(
                `Provider with key ${provider.name} is being overwritten`
            );
        }
        this.providers.set(provider.name, provider);
        if (this.active == null) {
            this.active = provider.name;
        }
    }

    public unregister(provider: AIProvider) {
        if (!this.providers.has(provider.name)) {
            console.error(
                `No provider registered with key ${provider.name} exists.`
            );
        } else {
            this.providers.delete(provider.name);
            if (this.active == provider.name) {
                this.active = this.providers.keys().next().value;
            }
        }
    }
    public setActive(provider: string | AIProvider) {
        if (typeof provider === "string") {
            if (this.active === provider) {
                console.warn(`The provider '${provider}' is already active`);
            } else if (this.providers.has(provider)) {
                this.active = provider;
            } else {
                console.error(
                    `No provider with the key '${provider} is currently registered'`
                );
            }
        } else {
            if (this.active === provider.name) {
                console.warn(`The provider '${provider}' is already active`);
            } else {
                this.providers.set(provider.name, provider);
                this.active = provider.name;
            }
        }
    }

    private get activeProvider(): AIProvider {
        if (!this.active) throw new Error("No active provider set");
        const provider = this.providers.get(this.active);
        if (!provider)
            throw new Error(`Active provider '${this.active}' not found`);
        return provider;
    }

    public chat(params: ChatParams) {
        return this.activeProvider.chat(params);
    }

    public chatStream(params: ChatParams) {
        return this.activeProvider.chatStream(params);
    }

    public embed(params: EmbedParams) {
        return this.activeProvider.embed(params);
    }
}

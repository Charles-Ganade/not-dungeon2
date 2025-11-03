import { JSONSchema } from "json-schema-typed";
import {
    ChatParams,
    FunctionCall,
    ProviderRegistry,
    ToolDefinition,
} from "../ai/provider";
import { GAME_DIRECTOR_PROMPTS } from "../prompts";

export class GameDirector {
    private providerRegistry: ProviderRegistry;
    private worldStateTools: ToolDefinition[];

    constructor(providerRegistry: ProviderRegistry) {
        this.providerRegistry = providerRegistry;
        this.worldStateTools = this.generateWorldStateTools();
    }

    private generateWorldStateTools(): ToolDefinition[] {
        const tools: ToolDefinition[] = [
            {
                type: "function",
                function: {
                    name: "patchState",
                    description:
                        "Deeply merges a partial state object into the current world state. Use this for general updates like character locations, item possessions, environment changes, etc.",
                    parameters: {
                        type: "object",
                        properties: {
                            partialState: {
                                type: "object",
                                description:
                                    "An partial of the world state containing only the changes to merge into the world state.",
                                additionalProperties: true,
                            },
                        },
                        required: ["partialState"],
                    } as JSONSchema,
                    strict: false,
                },
            },
            {
                type: "function",
                function: {
                    name: "addPlot",
                    description: "Adds a new plotline to the world state.",
                    parameters: {
                        type: "object",
                        properties: {
                            title: {
                                type: "string",
                                description: "Title of the new plot.",
                            },
                            description: {
                                type: "string",
                                description: "Description of the new plot.",
                            },
                            player_alignment: {
                                type: "number",
                                description:
                                    "Initial player alignment score (0.0 to 1.0).",
                            },
                        },
                        required: ["title", "description", "player_alignment"],
                    } as JSONSchema,
                    strict: false,
                },
            },
            {
                type: "function",
                function: {
                    name: "updatePlot",
                    description: "Updates an existing plotline by its ID.",
                    parameters: {
                        type: "object",
                        properties: {
                            plotId: {
                                type: "string",
                                description:
                                    "The unique ID of the plot to update.",
                            },
                            updates: {
                                type: "object",
                                properties: {
                                    title: {
                                        type: "string",
                                        description: "New title (optional).",
                                    },
                                    description: {
                                        type: "string",
                                        description:
                                            "New description (optional).",
                                    },
                                    player_alignment: {
                                        type: "number",
                                        description:
                                            "New alignment score (optional).",
                                    },
                                },
                                additionalProperties: false,
                            },
                        },
                        required: ["plotId", "updates"],
                    } as JSONSchema,
                    strict: false,
                },
            },
            {
                type: "function",
                function: {
                    name: "removePlot",
                    description: "Removes a plotline by its ID.",
                    parameters: {
                        type: "object",
                        properties: {
                            plotId: {
                                type: "string",
                                description:
                                    "The unique ID of the plot to remove.",
                            },
                        },
                        required: ["plotId"],
                    } as JSONSchema,
                    strict: false,
                },
            },
            {
                type: "function",
                function: {
                    name: "determineActionResult",
                    description:
                        "Explicitly determine if the player's primary action succeeds or fails, and provide a brief note on the outcome for the StoryWriter.",
                    parameters: {
                        type: "object",
                        properties: {
                            actionDescription: {
                                type: "string",
                                description:
                                    "A brief summary of the player's intended action.",
                            },
                            success: {
                                type: "boolean",
                                description: "Whether the action succeeded.",
                            },
                            outcomeNote: {
                                type: "string",
                                description:
                                    "A brief note describing the result (e.g., 'Guard is convinced', 'The lock breaks', 'Attack misses').",
                            },
                        },
                        required: [
                            "actionDescription",
                            "success",
                            "outcomeNote",
                        ],
                    } as JSONSchema,
                    strict: false,
                },
            },
        ];
        return tools;
    }

    public async *initializeWorldStream(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): AsyncGenerator<FunctionCall[] | undefined> {
        const systemPrompt = GAME_DIRECTOR_PROMPTS.initializeWorld;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context },
            ],
            tools: this.worldStateTools.filter((tool) =>
                ["patchState", "addPlot"].includes(tool.function.name)
            ),
            tool_choice: "auto",
            options: options,
            think: false,
        };

        const response = this.providerRegistry.chatStream(params);

        for await (const chunk of response) {
            yield chunk.delta.tool_calls;
        }
    }

    public async *assessPlayerTurnStream(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): AsyncGenerator<{
        tool_calls: FunctionCall[] | undefined;
        thinking: string | undefined;
    }> {
        const systemPrompt = GAME_DIRECTOR_PROMPTS.assessPlayerTurn;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context },
            ],
            tools: this.worldStateTools,
            tool_choice: "auto",
            options: options,
            think: true,
        };

        const response = this.providerRegistry.chatStream(params);

        for await (const chunk of response) {
            yield {
                thinking: chunk.delta.thinking,
                tool_calls: chunk.delta.tool_calls,
            };
        }
    }

    public async *assessWriterTurnStream(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): AsyncGenerator<{
        tool_calls: FunctionCall[] | undefined;
        thinking: string | undefined;
    }> {
        const systemPrompt = GAME_DIRECTOR_PROMPTS.assessWriterTurn;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context },
            ],
            tools: this.worldStateTools.filter((t) =>
                ["patchState", "updatePlot", "addPlot", "removePlot"].includes(
                    t.function.name
                )
            ),
            tool_choice: "auto",
            options: options,
            think: true,
        };

        const response = this.providerRegistry.chatStream(params);

        for await (const chunk of response) {
            yield {
                thinking: chunk.delta.thinking,
                tool_calls: chunk.delta.tool_calls,
            };
        }
    }
}

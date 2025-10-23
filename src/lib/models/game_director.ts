import { JSONSchema } from "json-schema-typed";
import {
    ChatParams,
    FunctionCall,
    ProviderRegistry,
    ToolDefinition,
} from "../ai/provider";
import { WorldState } from "./world_state";

export class GameDirector {
    private providerRegistry: ProviderRegistry;
    private worldStateTools: ToolDefinition[];

    constructor(providerRegistry: ProviderRegistry, worldState?: WorldState) {
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
                                    "An object containing only the changes to merge into the world state.",
                                additionalProperties: true, // Allows flexible state structure
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
            // NEW Tool: To handle background simulation results
            {
                type: "function",
                function: {
                    name: "recordBackgroundEvent",
                    description:
                        "Records a significant event that happened in the background simulation, outside the player's view. This note might be used by the StoryWriter later.",
                    parameters: {
                        type: "object",
                        properties: {
                            eventNote: {
                                type: "string",
                                description:
                                    "A brief note describing the background event (e.g., 'Merchant caravan arrived', 'Rival faction captured the McGuffin').",
                            },
                        },
                        required: ["eventNote"],
                    } as JSONSchema,
                    strict: false,
                },
            },
        ];
        return tools;
    }

    public async initializeWorld(
        context: string,
        model: string,
        options?: ChatParams["options"]
    ): Promise<FunctionCall[] | undefined> {
        const systemPrompt = `You are the Game Director initializing a new story world. Based on the context, set up the initial state. Use the available tools ('patchState', 'addPlot') to define starting characters, locations, items, and initial plotlines. Focus only on calling tools.`;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context },
            ],
            tools: this.worldStateTools.filter((t) =>
                ["patchState", "addPlot"].includes(t.function.name)
            ), // Limit tools for init
            tool_choice: "auto", // Allow multiple calls
            options: options,
            format: "json", // Helpful for tool use consistency
        };

        const response = await this.providerRegistry.chat(params); //
        return response.message.tool_calls as FunctionCall[] | undefined;
    }

    public async assessPlayerTurn(
        context: string, // From ContextBuilder.buildDirectorContext
        model: string,
        options?: ChatParams["options"]
    ): Promise<{
        tool_calls: FunctionCall[] | undefined;
        thinking: string | undefined;
    }> {
        const systemPrompt = `You are the Game Director. Analyze the player's action within the game context.
1. Determine Success/Failure: Call 'determineActionResult' for the player's main action.
2. Update Plots: Assess player alignment with active plots. Call 'updatePlot' if alignment changes significantly, or 'addPlot'/'removePlot' if necessary based on player actions deviating from or resolving plots.
3. Simulate Background (Optional): If enough time seems to have passed or consequences ripple outward, briefly simulate events outside the player's view. Use 'patchState' to reflect resulting changes (e.g., NPC movement, item location changes) and 'recordBackgroundEvent' to note significant occurrences.
4. Direct State Changes: Use 'patchState' for any other direct consequences of the player's action (e.g., item usage, immediate NPC reaction state).
Focus **only** on calling the necessary tools based on your assessment. Make multiple tool calls if needed.`;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context }, // Context includes player input
            ],
            tools: this.worldStateTools, // Provide all tools
            tool_choice: "auto", // Crucial: Allow the LLM to choose which tools to call, potentially multiple
            options: options,
            think: true,
        };

        const response = await this.providerRegistry.chat(params); //
        return {
            tool_calls: response.message.tool_calls as
                | FunctionCall[]
                | undefined,
            thinking: response.message.thinking,
        };
    }

    public async assessWriterTurn(
        context: string, // From ContextBuilder.buildDirectorPostWriterContext
        model: string,
        options?: ChatParams["options"]
    ): Promise<{
        tool_calls: FunctionCall[] | undefined;
        thinking: string | undefined;
    }> {
        const systemPrompt = `You are the Game Director. Review the story text that was just written. Identify if the narrative implies any changes to the world state (e.g., a character picked up an item, moved location, changed disposition; a plot point was advanced). If changes are implied, use the 'patchState' or 'updatePlot' tools to update the world state accordingly. If no state changes are implied by the text, do not call any tools. Focus **only** on calling tools for implied state updates.`;

        const params: ChatParams = {
            model: model,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: context }, // Context includes writer output
            ],
            tools: this.worldStateTools.filter((t) =>
                ["patchState", "updatePlot"].includes(t.function.name)
            ),
            tool_choice: "auto",
            options: options,
            format: "json", // Enforce JSON for better tool call generation
        };

        const response = await this.providerRegistry.chat(params); //
        return {
            tool_calls: response.message.tool_calls as
                | FunctionCall[]
                | undefined,
            thinking: response.message.thinking,
        };
    }
}

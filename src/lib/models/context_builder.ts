import { MemoryBank } from "./memory_bank";
import { PlotCardManager } from "./plot_card_manager";
import { StoryHistory } from "./story_history";
import { WorldState } from "./world_state";

export class ContextBuilder {
    private worldState: WorldState;
    private storyHistory: StoryHistory;
    private memoryBank: MemoryBank;
    private plotCardManager: PlotCardManager;

    constructor(
        worldState: WorldState,
        storyHistory: StoryHistory,
        memoryBank: MemoryBank,
        plotCardManager: PlotCardManager
    ) {
        this.worldState = worldState;
        this.storyHistory = storyHistory;
        this.memoryBank = memoryBank;
        this.plotCardManager = plotCardManager;
    }

    public async buildDirectorContext(
        playerInput: string,
        currentTurn: number,
        historyTurns: number = 10,
        memoryLimit: number = 5,
        plotCardLimit: number = 3
    ): Promise<string> {
        const recentTurns = this.storyHistory.getLastNTurns(historyTurns); //
        const historyText = recentTurns
            .map((t) => `${t.actor}: ${t.text}`)
            .join("\n");
        const worldStateText = this.worldState.toText(); //
        const searchQuery = `${historyText}\nPlayer: ${playerInput}`;
        const [relevantMemories, relevantPlotCards] = await Promise.all([
            this.memoryBank.search(searchQuery, currentTurn, memoryLimit),
            this.plotCardManager.search(searchQuery, plotCardLimit),
        ]);
        const memoryText =
            relevantMemories.length > 0
                ? relevantMemories
                      .map((m, i) => `${i + 1}. ${m.text}`)
                      .join("\n")
                : "None.";
        const plotCardText =
            relevantPlotCards.length > 0
                ? relevantPlotCards
                      .map(
                          (p, i) =>
                              `${i + 1}. [${p.category}: ${p.name}] ${
                                  p.content
                              }`
                      )
                      .join("\n")
                : "None.";

        let context = `## Game Context (Turn ${currentTurn})\n\n`;
        context += `### Recent Story History\n${historyText}\n\n`;
        context += `### Player's Current Action\n${playerInput}\n\n`;
        context += `### Relevant Memories\n${memoryText}\n\n`;
        context += `### Relevant Plot Cards (World Info)\n${plotCardText}\n\n`;
        context += `### Current World State & Plots\n${worldStateText}\n`;

        return context;
    }

    public async buildWriterContext(
        directorOutcome: string, // Notes from GameDirector about action results/background events
        currentTurn: number,
        historyTurns: number = 5,
        memoryLimit: number = 3,
        plotCardLimit: number = 2
    ): Promise<string> {
        // --- Fetch Context Components ---
        const recentTurns = this.storyHistory.getLastNTurns(historyTurns); //
        const historyText = recentTurns
            .map((t) => `${t.actor}: ${t.text}`)
            .join("\n");
        const worldStateText = this.worldState.toText(); //

        // Query based on recent events + director outcome
        const searchQuery = `${historyText}\nDirector Outcome: ${directorOutcome}`;
        const [relevantMemories, relevantPlotCards] = await Promise.all([
            this.memoryBank.search(searchQuery, currentTurn, memoryLimit),
            this.plotCardManager.search(searchQuery, plotCardLimit),
        ]);

        const memoryText =
            relevantMemories.length > 0
                ? relevantMemories
                      .map((m, i) => `${i + 1}. ${m.text}`)
                      .join("\n")
                : "None.";
        const plotCardText =
            relevantPlotCards.length > 0
                ? relevantPlotCards
                      .map(
                          (p, i) =>
                              `${i + 1}. [${p.category}: ${p.name}] ${
                                  p.content
                              }`
                      )
                      .join("\n")
                : "None.";

        // --- Format Final Context String ---
        let context = `## Story Writing Context (Turn ${currentTurn})\n\n`;
        context += `### Last Few Turns\n${historyText}\n\n`;
        context += `### What Just Happened / Director's Notes\n${directorOutcome}\n\n`;
        context += `### Relevant Background Information (Memories)\n${memoryText}\n\n`;
        context += `### Relevant Plot Cards (World Info)\n${plotCardText}\n\n`;
        context += `### Current World State Snapshot\n${worldStateText}\n\n`; // Consider simplifying this later
        context += `---`;
        context += `\nContinue the story based on the events and context above. Write the next paragraph or scene.\n`;

        return context;
    }

    /**
     * Builds context for the GameDirector to assess StoryWriter output.
     */
    public async buildDirectorPostWriterContext(
        writerOutput: string,
        currentTurn: number,
        historyTurns: number = 3,
        memoryLimit: number = 3,
        plotCardLimit: number = 2
    ): Promise<string> {
        // --- Fetch Context Components ---
        const recentTurns = this.storyHistory.getLastNTurns(historyTurns); //
        const historyText = recentTurns
            .map((t) => `${t.actor}: ${t.text}`)
            .join("\n"); // Includes writerOutput if historyTurns > 0
        const worldStateText = this.worldState.toText(); //

        // Query based primarily on the writer's new text
        const searchQuery = writerOutput;
        const [relevantMemories, relevantPlotCards] = await Promise.all([
            this.memoryBank.search(searchQuery, currentTurn, memoryLimit),
            this.plotCardManager.search(searchQuery, plotCardLimit),
        ]);

        const memoryText =
            relevantMemories.length > 0
                ? relevantMemories
                      .map((m, i) => `${i + 1}. ${m.text}`)
                      .join("\n")
                : "None.";
        const plotCardText =
            relevantPlotCards.length > 0
                ? relevantPlotCards
                      .map(
                          (p, i) =>
                              `${i + 1}. [${p.category}: ${p.name}] ${
                                  p.content
                              }`
                      )
                      .join("\n")
                : "None.";

        // --- Format Final Context String ---
        let context = `## Post-Narration Assessment (Turn ${currentTurn})\n\n`;
        context += `### Story Text Just Written:\n${writerOutput}\n\n`;
        context += `### Recent History Context:\n${historyText}\n\n`;
        context += `### Relevant Memories:\n${memoryText}\n\n`;
        context += `### Relevant Plot Cards (World Info):\n${plotCardText}\n\n`;
        context += `### Current World State:\n${worldStateText}\n\n`;
        context += `---`;
        context += `\nReview the "Story Text Just Written". If it implies any changes to the world state (e.g., character locations, item possessions, plot progression, character dispositions), call the appropriate tools ('patchState', 'updatePlot') to update the state. If no changes are implied, do not call any tools.`;

        return context;
    }
}

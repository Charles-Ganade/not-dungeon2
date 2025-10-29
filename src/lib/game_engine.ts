import { ChatParams, ProviderRegistry } from "./ai/provider";
import { OllamaProvider } from "./ai/providers/ollama_provider";
import { ContextBuilder } from "./models/context_builder";
import { GameDirector } from "./models/game_director";
import { Memory, MemoryBank } from "./models/memory_bank";
import { PlotCard, PlotCardManager } from "./models/plot_card_manager";
import {
    StoryNode,
    StoryTree,
    StoryActor,
    SerializedStoryTree,
} from "./models/story_tree";
import { StoryWriter } from "./models/story_writer";
import { DeltaPair, Plot, WorldState } from "./models/world_state";
import { FunctionCall } from "./ai/provider";

export interface AIConfig {
    provider: string; // e.g., 'ollama'
    model: string;
    options?: ChatParams["options"];
}

export interface GameAction {
    fromNodeId: string;
    toNodeId: string;
    worldStateDeltas: DeltaPair[];
    storyTreeDeltas: DeltaPair[];
    memoryBankDeltas: DeltaPair[];
}

export interface SerializedGameSession {
    worldState: {
        state: any;
        plots: Plot[];
    };
    storyTree: SerializedStoryTree;
    memories: Memory[];
    plotCards: PlotCard[];

    currentNodeId: string;

    directorConfig: AIConfig;
    writerConfig: AIConfig;
}

export class GameEngine {
    public providerRegistry: ProviderRegistry;
    public worldState: WorldState;
    public storyTree: StoryTree;
    public memoryBank: MemoryBank;
    public plotCardManager: PlotCardManager;
    public gameDirector: GameDirector;
    public storyWriter: StoryWriter;
    public contextBuilder: ContextBuilder;

    private currentNodeId: string | null = null;
    private undoStack: GameAction[] = [];
    private redoStack: GameAction[] = [];

    public directorConfig: AIConfig;
    public writerConfig: AIConfig;

    private readonly DEFAULT_WRITER_CONFIG: AIConfig = {
        provider: "ollama",
        model: "llama3",
        options: { temperature: 0.7 },
    };
    private readonly DEFAULT_DIRECTOR_CONFIG: AIConfig = {
        provider: "ollama",
        model: "llama3",
        options: { temperature: 0.1 },
    };

    constructor() {
        this.providerRegistry = new ProviderRegistry();
        this.providerRegistry.register(new OllamaProvider());
        this.providerRegistry.setActive("ollama");

        this.worldState = new WorldState();
        this.storyTree = new StoryTree();

        const embeddingModel = "nomic-embed-text";
        const summarizerModel = "llama3";

        this.memoryBank = new MemoryBank(
            this.providerRegistry,
            embeddingModel,
            summarizerModel,
            768
        );
        this.plotCardManager = new PlotCardManager(
            this.providerRegistry,
            embeddingModel,
            768
        );

        this.gameDirector = new GameDirector(this.providerRegistry);
        this.storyWriter = new StoryWriter(this.providerRegistry);

        this.contextBuilder = new ContextBuilder(
            this.worldState,
            this.storyTree,
            this.memoryBank,
            this.plotCardManager
        );

        this.directorConfig = { ...this.DEFAULT_DIRECTOR_CONFIG };
        this.writerConfig = { ...this.DEFAULT_WRITER_CONFIG };
    }

    public async init() {
        await this.memoryBank.init();
        await this.plotCardManager.init();
    }

    public async newGame(instructions?: string) {
        // 1. Reset all state
        this.worldState = new WorldState();
        this.storyTree = new StoryTree();

        this.memoryBank;
        // TODO: Clear MemoryBank and PlotCardManager DBs if desired

        this.currentNodeId = null;
        this.undoStack = [];
        this.redoStack = [];

        let firstNodeText = "The story begins.";
        let worldStateDeltas: DeltaPair[] = [];
        let memoryBankDeltas: DeltaPair[] = [];

        // 2. Initialize WorldState with GameDirector if instructions are given
        if (instructions) {
            firstNodeText = "The world is being shaped by your design.";
            const context = `## Initial World Instructions\n${instructions}`;
            const tool_calls = await this.gameDirector.initializeWorld(
                context,
                this.directorConfig.model,
                this.directorConfig.options
            );

            if (tool_calls) {
                const { worldDeltas } = this._applyDirectorTools(tool_calls);
                worldStateDeltas.push(...worldDeltas);
            }
        }

        // 3. Create the "root" node of the story
        const { newId: rootNodeId, deltas: storyTreeDeltas } =
            this._createStoryNode(
                "storywriter",
                firstNodeText,
                null, // No parent
                worldStateDeltas,
                memoryBankDeltas
            );

        this.currentNodeId = rootNodeId;

        // This initial setup is not undo-able.
        this.undoStack = [];
        this.redoStack = [];
    }

    public async loadGame(session: SerializedGameSession) {
        this.worldState = new WorldState(
            session.worldState.state,
            session.worldState.plots
        );

        this.storyTree = StoryTree.deserialize(session.storyTree);

        this.directorConfig = session.directorConfig;
        this.writerConfig = session.writerConfig;

        this.currentNodeId = session.currentNodeId;

        // 5. Restore DBs (simple overwrite)
        // TODO: Implement a more robust clear/bulk-add
        await Promise.all([
            this.memoryBank.init(),
            this.plotCardManager.init(),
        ]);
        for (const mem of session.memories) {
            await this.memoryBank.addMemory(mem.text, mem.created_at_turn);
        }

        for (const card of session.plotCards) {
            await this.plotCardManager.addPlotCard(card);
        }

        // 6. Clear undo/redo stacks, as state is loaded fresh
        this.undoStack = [];
        this.redoStack = [];
    }

    /**
     * Serializes the entire current game session.
     */
    public saveGame(): SerializedGameSession {
        if (!this.currentNodeId) {
            throw new Error("Cannot save an uninitialized game.");
        }

        return {
            worldState: {
                state: this.worldState.getState(),
                plots: this.worldState.getPlots(),
            },
            storyTree: this.storyTree.serialize(),
            // This is a simplification. A real implementation might
            // query the DBs directly.
            memories: (this.memoryBank as any).memories,
            plotCards: this.plotCardManager.getAllPlotCards(),

            currentNodeId: this.currentNodeId,

            directorConfig: this.directorConfig,
            writerConfig: this.writerConfig,
        };
    }

    // === (IN-SESSION ACTIONS) ===

    /**
     * Player takes an action. (P-W flow)
     */
    public async act(playerInput: string) {
        if (!this.currentNodeId) throw new Error("Game not started.");

        const fromNodeId = this.currentNodeId;

        // === 1. (P) Player Turn ===
        const {
            nodeId: playerNodeId,
            worldDeltas: playerWorldDeltas,
            treeDeltas: playerTreeDeltas,
            memDeltas: playerMemDeltas,
            directorOutcome,
        } = await this._executePlayerTurn(playerInput, fromNodeId);

        // === 2. (W) Writer Turn ===
        const {
            nodeId: writerNodeId,
            worldDeltas: writerWorldDeltas,
            treeDeltas: writerTreeDeltas,
            memDeltas: writerMemDeltas,
        } = await this._executeWriterTurn(playerNodeId, directorOutcome);

        // === 3. Bundle & Save Action ===
        this._pushAction({
            fromNodeId: fromNodeId,
            toNodeId: writerNodeId,
            worldStateDeltas: [...playerWorldDeltas, ...writerWorldDeltas],
            storyTreeDeltas: [...playerTreeDeltas, ...writerTreeDeltas],
            memoryBankDeltas: [...playerMemDeltas, ...writerMemDeltas],
        });

        this.currentNodeId = writerNodeId;
        // Notify UI that state has changed...
    }

    /**
     * Player skips their turn, letting the writer continue. (W flow)
     */
    public async continue() {
        if (!this.currentNodeId) throw new Error("Game not started.");

        const fromNodeId = this.currentNodeId;
        const directorOutcome = "The player waits to see what happens next.";

        // === 1. (W) Writer Turn ===
        const {
            nodeId: writerNodeId,
            worldDeltas: writerWorldDeltas,
            treeDeltas: writerTreeDeltas,
            memDeltas: writerMemDeltas,
        } = await this._executeWriterTurn(fromNodeId, directorOutcome);

        // === 2. Bundle & Save Action ===
        this._pushAction({
            fromNodeId: fromNodeId,
            toNodeId: writerNodeId,
            worldStateDeltas: writerWorldDeltas,
            storyTreeDeltas: writerTreeDeltas,
            memoryBankDeltas: writerMemDeltas,
        });

        this.currentNodeId = writerNodeId;
        // Notify UI...
    }

    /**
     * Reruns the last Writer turn from the Player's input, creating a branch.
     * (P-W1) -> P-(W1, >W2)
     */
    public async retry() {
        if (!this.currentNodeId) throw new Error("Game not started.");

        const writerNode = this.storyTree.getNode(this.currentNodeId); // This is W1
        if (!writerNode || writerNode.turn.actor !== "storywriter") {
            console.warn("Retry can only be called after a storywriter turn.");
            return;
        }

        const playerNode = this.storyTree.getNode(writerNode.parentId); // This is P
        if (!playerNode) {
            console.warn("Cannot retry from this node.");
            return;
        }

        // 1. "Time travel" back to the player node (P).
        // This reverts the state changes made by W1.
        await this.switchToNode(playerNode.id);

        // 2. Get the original Director outcome from that player node
        // We need to store this on the node. Let's modify StoryNode...
        // For now, let's just re-run the player turn invisibly to get the outcome.
        // Or, even better, let's assume the directorOutcome is stored on the P node.
        // Let's modify `_executePlayerTurn` to store `directorThinking`
        const directorOutcome =
            playerNode.turn.directorThinking || "No outcome recorded.";

        // 3. Execute *only* a new Writer turn (W2).
        const {
            nodeId: newWriterNodeId,
            worldDeltas,
            treeDeltas,
            memDeltas,
        } = await this._executeWriterTurn(playerNode.id, directorOutcome);

        // 4. Push this new *single* W turn as an action
        this._pushAction({
            fromNodeId: playerNode.id,
            toNodeId: newWriterNodeId,
            worldStateDeltas: worldDeltas,
            storyTreeDeltas: treeDeltas,
            memoryBankDeltas: memDeltas,
        });

        this.currentNodeId = newWriterNodeId;
        // Notify UI...
    }

    /**
     * Deletes the current node and all its children.
     */
    public async deleteCurrentNode() {
        if (!this.currentNodeId) return;

        const nodeToDelete = this.storyTree.getNode(this.currentNodeId);
        if (!nodeToDelete || !nodeToDelete.parentId) {
            console.warn("Cannot delete the root node.");
            return;
        }

        const fromNodeId = this.currentNodeId;
        const parentId = nodeToDelete.parentId;

        // 1. Delete the branch from the story tree
        const result = this.storyTree.deleteBranch(this.currentNodeId);
        if (!result) return;

        const { deletedNodes, delta: storyTreeDelta } = result;

        // 2. Collect all REVERT deltas from the deleted nodes
        let worldReverts: DeltaPair[] = [];
        let memoryReverts: DeltaPair[] = [];

        // We must apply reverts in reverse order (child-first)
        for (const node of deletedNodes) {
            const deltas = (node as any).allDeltas as {
                world: DeltaPair[];
                mem: DeltaPair[];
            };
            if (deltas) {
                worldReverts.push(...deltas.world);
                memoryReverts.push(...deltas.mem);
            }
        }

        // 3. Apply all reverts to the actual state
        for (const delta of worldReverts) {
            this.worldState.applyDelta(delta.revert);
        }
        for (const delta of memoryReverts) {
            await this.memoryBank.applyDelta(delta.revert);
        }

        // 4. Create a GameAction that bundles the *single* storyTree delta
        // with the *many* revert deltas for world/memory.
        const action: GameAction = {
            fromNodeId: fromNodeId,
            toNodeId: parentId,
            storyTreeDeltas: [storyTreeDelta],
            worldStateDeltas: worldReverts,
            memoryBankDeltas: memoryReverts,
        };

        // 5. This is tricky. We need to push a *custom* action
        // where 'apply' does the delete, and 'revert' re-adds everything.
        // This is complex. A simpler `delete` is just a `switchToNode(parentId)`
        // and leaving the branch orphaned.
        //
        // Let's go with the simpler `delete`:
        // Just move up, and the branch becomes orphaned, but still exists.
        // A 'true' delete is much harder to make undo-able.
        console.warn("Simple `delete` (switch to parent) not yet implemented.");
        console.warn("A 'true' undo-able `delete` is complex.");

        // For now, let's implement the "simple" delete (orphaning):
        await this.switchToNode(parentId);
    }

    /**
     * Undoes the last action.
     */
    public async undo() {
        const action = this.undoStack.pop();
        if (!action) return;

        // Apply all revert deltas
        for (const delta of [...action.worldStateDeltas].reverse()) {
            this.worldState.applyDelta(delta.revert);
        }
        for (const delta of [...action.storyTreeDeltas].reverse()) {
            this.storyTree.applyDelta(delta.revert);
        }
        for (const delta of [...action.memoryBankDeltas].reverse()) {
            await this.memoryBank.applyDelta(delta.revert);
        }

        this.redoStack.push(action);
        this.currentNodeId = action.fromNodeId;
        // Notify UI...
    }

    /**
     * Redoes the last undone action.
     */
    public async redo() {
        const action = this.redoStack.pop();
        if (!action) return;

        // Apply all apply deltas
        for (const delta of action.worldStateDeltas) {
            this.worldState.applyDelta(delta.apply);
        }
        for (const delta of action.storyTreeDeltas) {
            this.storyTree.applyDelta(delta.apply);
        }
        for (const delta of action.memoryBankDeltas) {
            await this.memoryBank.applyDelta(delta.apply);
        }

        this.undoStack.push(action);
        this.currentNodeId = action.toNodeId;
        // Notify UI...
    }

    // === (IN-SESSION NAVIGATION / EDITING) ===

    /**
     * Jumps to any node in the tree, applying/reverting state as needed.
     * This is the "time travel" engine.
     */
    public async switchToNode(targetNodeId: string) {
        if (targetNodeId === this.currentNodeId) return;
        if (!this.currentNodeId) {
            // Should only happen if game is empty
            this.currentNodeId = targetNodeId;
            return;
        }

        // 1. Find paths from root
        const currentPath = this.storyTree.getPathToNode(this.currentNodeId);
        const targetPath = this.storyTree.getPathToNode(targetNodeId);
        if (targetPath.length === 0) {
            throw new Error(`Target node ${targetNodeId} not found.`);
        }

        // 2. Find common ancestor
        let ancestorIndex = -1;
        for (
            let i = 0;
            i < Math.min(currentPath.length, targetPath.length);
            i++
        ) {
            if (currentPath[i].id === targetPath[i].id) {
                ancestorIndex = i;
            } else {
                break;
            }
        }

        if (ancestorIndex === -1) {
            throw new Error("No common ancestor found. Story tree is corrupt.");
        }

        // 3. Revert actions from current node up to ancestor
        for (let i = currentPath.length - 1; i > ancestorIndex; i--) {
            const node = currentPath[i];
            const deltas = (node as any).allDeltas as {
                world: DeltaPair[];
                mem: DeltaPair[];
            };
            if (deltas) {
                for (const d of [...deltas.world].reverse())
                    this.worldState.applyDelta(d.revert);
                for (const d of [...deltas.mem].reverse())
                    await this.memoryBank.applyDelta(d.revert);
            }
        }

        // 4. Apply actions from ancestor down to target node
        for (let i = ancestorIndex + 1; i < targetPath.length; i++) {
            const node = targetPath[i];
            const deltas = (node as any).allDeltas as {
                world: DeltaPair[];
                mem: DeltaPair[];
            };
            if (deltas) {
                for (const d of deltas.world)
                    this.worldState.applyDelta(d.apply);
                for (const d of deltas.mem)
                    await this.memoryBank.applyDelta(d.apply);
            }
        }

        // 5. Set new node and clear undo/redo
        this.currentNodeId = targetNodeId;
        this.undoStack = [];
        this.redoStack = [];

        // Notify UI...
    }

    // === (MISC) ===

    public setWriterSettings(config: AIConfig) {
        this.writerConfig = config;
    }

    public setDirectorSettings(config: AIConfig) {
        this.directorConfig = config;
    }

    // Plot Card CRUD (simple pass-through)
    public addPlotCard(cardData: Omit<PlotCard, "id">) {
        return this.plotCardManager.addPlotCard(cardData);
    }

    public editPlotCard(id: number, updates: Partial<Omit<PlotCard, "id">>) {
        return this.plotCardManager.editPlotCard(id, updates);
    }

    public removePlotCard(id: number) {
        return this.plotCardManager.removePlotCard(id);
    }

    // === (PRIVATE HELPERS) ===

    /**
     * Executes the (P) Player-Director half of a turn.
     * @returns The new node ID and all deltas generated.
     */
    private async _executePlayerTurn(
        playerInput: string,
        parentNodeId: string
    ) {
        const currentTurnNum = this.storyTree.getDepth(parentNodeId) + 1;

        // 1. Build Director Context
        const directorContext = await this.contextBuilder.buildDirectorContext(
            playerInput,
            parentNodeId,
            currentTurnNum
        );

        // 2. Call Director
        const { tool_calls, thinking } =
            await this.gameDirector.assessPlayerTurn(
                directorContext,
                this.directorConfig.model,
                this.directorConfig.options
            );

        // 3. Apply Director tool calls
        const { worldDeltas, directorOutcome } =
            this._applyDirectorTools(tool_calls);

        // 4. Create and save the Player node
        const { newId: playerNodeId, deltas: treeDeltas } =
            this._createStoryNode(
                "player",
                playerInput,
                parentNodeId,
                worldDeltas,
                [], // Player turn doesn't generate memories
                thinking // Store director thinking here
            );

        return {
            nodeId: playerNodeId,
            worldDeltas,
            treeDeltas,
            memDeltas: [],
            directorOutcome,
        };
    }

    /**
     * Executes the (W) Writer-Director half of a turn.
     * @returns The new node ID and all deltas generated.
     */
    private async _executeWriterTurn(
        parentNodeId: string,
        directorOutcome: string
    ) {
        const currentTurnNum = this.storyTree.getDepth(parentNodeId) + 1;
        let allMemDeltas: DeltaPair[] = [];

        // 1. Build Writer Context
        const writerContext = await this.contextBuilder.buildWriterContext(
            directorOutcome,
            parentNodeId,
            currentTurnNum
        );

        // 2. Call Story Writer
        const storyOutput = await this.storyWriter.writeNextTurn(
            writerContext,
            this.writerConfig.model,
            this.writerConfig.options
        );

        // 3. Build Post-Writer Director Context
        const postWriterContext =
            await this.contextBuilder.buildDirectorPostWriterContext(
                storyOutput,
                parentNodeId,
                currentTurnNum
            );

        // 4. Call Director (Post-Writer)
        const { tool_calls, thinking } =
            await this.gameDirector.assessWriterTurn(
                postWriterContext,
                this.directorConfig.model,
                this.directorConfig.options
            );

        // 5. Apply Director tool calls
        const { worldDeltas } = this._applyDirectorTools(tool_calls);

        // 6. Generate and save a memory for this (P-W) turn pair
        const parentNode = this.storyTree.getNode(parentNodeId);
        if (parentNode && parentNode.turn.actor === "player") {
            const playerTurn = parentNode.turn;
            const writerTurn = { text: storyOutput, actor: "storywriter" };

            // This call internally creates and applies its own deltas
            const { deltaPair } = await this.memoryBank.generateAndAddMemory(
                [playerTurn, writerTurn] as any, // Cast since StoryTurn lacks id/created_at
                currentTurnNum
            );
            allMemDeltas.push(deltaPair);
        }

        // 7. Create and save the Writer node
        const { newId: writerNodeId, deltas: treeDeltas } =
            this._createStoryNode(
                "storywriter",
                storyOutput,
                parentNodeId,
                worldDeltas,
                allMemDeltas,
                thinking
            );

        return {
            nodeId: writerNodeId,
            worldDeltas: worldDeltas,
            treeDeltas: treeDeltas,
            memDeltas: allMemDeltas,
        };
    }

    /**
     * Applies Director tool calls and returns the resulting deltas.
     */
    private _applyDirectorTools(tool_calls: FunctionCall[] | undefined): {
        worldDeltas: DeltaPair[];
        directorOutcome: string;
    } {
        let worldDeltas: DeltaPair[] = [];
        let outcome = "The story continues."; // Default outcome

        if (!tool_calls) {
            return { worldDeltas, directorOutcome: outcome };
        }

        for (const call of tool_calls) {
            const args = JSON.parse(call.function.arguments);
            let delta: DeltaPair | null = null;

            switch (call.function.name) {
                case "patchState":
                    delta = this.worldState.patchState(args.partialState);
                    break;
                case "addPlot":
                    delta = this.worldState.addPlot(args).delta;
                    break;
                case "updatePlot":
                    delta = this.worldState.updatePlot(
                        args.plotId,
                        args.updates
                    );
                    break;
                case "removePlot":
                    delta = this.worldState.removePlot(args.plotId);
                    break;
                case "determineActionResult":
                    // This is a "meta" tool, it doesn't change state directly
                    outcome = args.outcomeNote;
                    break;
                case "recordBackgroundEvent":
                    // This is also a meta tool
                    outcome += `\n(Background: ${args.eventNote})`;
                    break;
            }
            if (delta) {
                worldDeltas.push(delta);
            }
        }
        return { worldDeltas, directorOutcome: outcome };
    }

    /**
     * Creates a new StoryNode, saves it to the tree, and returns its ID and deltas.
     */
    private _createStoryNode(
        actor: StoryActor,
        text: string,
        parentId: string | null,
        worldDeltas: DeltaPair[],
        memDeltas: DeltaPair[],
        directorThinking?: string
    ): { newId: string; deltas: DeltaPair[] } {
        const node: StoryNode = {
            id: crypto.randomUUID(),
            parentId: parentId || "", // parentId is empty string for root
            childrenIds: [],
            turn: {
                actor: actor,
                text: text,
                directorThinking: directorThinking,
            },
            // This is the CRITICAL part: we attach the deltas that
            // were created *with* this node to the node itself.
            // This allows `switchToNode` to find them.
            deltas: [], // DEPRECATED: We will store deltas on a hidden prop
        };

        // This is a bit of a hack to attach the deltas without
        // serializing them into the DB via storyTree.serialize()
        // `story_tree.ts`'s `StoryNode` interface should be updated.
        (node as any).allDeltas = {
            world: worldDeltas,
            mem: memDeltas,
        };

        const treeDelta = this.storyTree.addNode(node);
        if (!treeDelta) {
            throw new Error(`Failed to add node ${node.id} to tree.`);
        }

        return { newId: node.id, deltas: [treeDelta] };
    }

    /**
     * Pushes a completed action to the undo stack and clears the redo stack.
     */
    private _pushAction(action: GameAction) {
        this.undoStack.push(action);
        this.redoStack = [];

        // Optional: Limit undo stack size
        // if (this.undoStack.length > 50) {
        //     this.undoStack.shift();
        // }
    }
}

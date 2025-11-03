import {
    ChatParams,
    FunctionCall,
    ProviderRegistry,
    StreamedChunk,
} from "./ai/provider";
import { ContextBuilder } from "./models/context_builder";
import { GameDirector } from "./models/game_director";
import { MemoryBank } from "./models/memory_bank";
import { PlotCardManager } from "./models/plot_card_manager";
import { StoryTree, StoryNode } from "./models/story_tree";
import { StoryWriter } from "./models/story_writer";
import { DeltaPair, Plot, WorldState } from "./models/world_state";
import { deepMerge } from "./util/objects";

export interface AIModelConfig {
    model: string;
    options?: ChatParams["options"];
}

export interface GameEngineConfig {
    directorEnabled: boolean;
    memoryGenerationInterval: number;
    models: {
        gameDirector: AIModelConfig;
        storyWriter: AIModelConfig;
        memorySummarizer: AIModelConfig;
        embed: {
            model: string;
            vectorDimension: number;
        };
    };
}

export interface SerializedSession {
    config: GameEngineConfig;
    selectedNodeId: string | null;
    storyTree: ReturnType<StoryTree["serialize"]>;
    worldState: {
        state: any;
        plots: Plot[];
    };
    memoryBank: any;
    plotCards: any;
}

interface EngineAction {
    type: "act" | "continue" | "erase" | "retry" | "edit" | "select" | "init";
    fromNodeId: string | null;
    toNodeId: string | null;
    deltas: {
        tree: DeltaPair | null;
        game: DeltaPair | null;
    };
}

export class GameEngine {
    private providerRegistry: ProviderRegistry;
    private worldState: WorldState;
    private storyTree: StoryTree;
    private memoryBank: MemoryBank;
    private plotCardManager: PlotCardManager;
    private contextBuilder: ContextBuilder;
    private gameDirector: GameDirector;
    private storyWriter: StoryWriter;

    private config: GameEngineConfig;
    private selectedNodeId: string | null = null;
    private isInitialized: boolean = false;
    private turnCounter: number = 0;

    private undoStack: EngineAction[] = [];
    private redoStack: EngineAction[] = [];

    constructor(
        providerRegistry: ProviderRegistry,
        initialConfig: GameEngineConfig
    ) {
        this.providerRegistry = providerRegistry;
        this.config = initialConfig;

        this.worldState = new WorldState();
        this.storyTree = new StoryTree();
        this.gameDirector = new GameDirector(this.providerRegistry);
        this.storyWriter = new StoryWriter(this.providerRegistry);

        this.memoryBank = new MemoryBank(
            this.providerRegistry,
            this.config.models.embed.model,
            this.config.models.memorySummarizer.model,
            this.config.models.embed.vectorDimension
        );

        this.plotCardManager = new PlotCardManager(
            this.providerRegistry,
            this.config.models.embed.model,
            this.config.models.embed.vectorDimension
        );

        this.contextBuilder = new ContextBuilder(
            this.worldState,
            this.storyTree,
            this.memoryBank,
            this.plotCardManager
        );

        this.initDatabases();
    }

    private async initDatabases() {
        await this.memoryBank.init();
        await this.plotCardManager.init();
    }

    public configure(options: Partial<GameEngineConfig>) {
        this.config = deepMerge(this.config, options);
    }

    public async newSession(
        prologue: string,
        isInstruction: boolean,
        onChunk?: (chunk: StreamedChunk) => void
    ) {
        await this.memoryBank.clear();
        await this.plotCardManager.clear();
        this.worldState = new WorldState();
        this.storyTree = new StoryTree();
        this.selectedNodeId = null;
        this.turnCounter = 0;
        this.undoStack = [];
        this.redoStack = [];

        let openingText = prologue;

        if (isInstruction) {
            const gen = this.storyWriter.writeNextTurnStream(
                prologue,
                this.config.models.storyWriter.model,
                this.config.models.storyWriter.options
            );
            openingText = "";
            for await (const chunk of gen) {
                if (chunk.delta.content) {
                    openingText += chunk.delta.content;
                }
                onChunk?.(chunk);
            }
        }

        const allGameDeltas: DeltaPair[] = [];

        if (this.config.directorEnabled) {
            const directorModel = this.config.models.gameDirector;
            const initStream = this.gameDirector.initializeWorldStream(
                openingText,
                directorModel.model,
                directorModel.options
            );

            for await (const toolCalls of initStream) {
                if (toolCalls) {
                    const deltas = (
                        await this._processToolCalls(toolCalls)
                    ).filter((r) => typeof r !== "string");
                    allGameDeltas.push(...deltas);
                }
            }
        }

        const rootNode: StoryNode = {
            id: crypto.randomUUID(),
            parentId: "",
            childrenIds: [],
            turn: { actor: "storywriter", text: openingText },
            deltas: allGameDeltas,
        };

        const treeDelta = this.storyTree.addNode(rootNode);
        this.selectedNodeId = rootNode.id;
        this.isInitialized = true;
        this.turnCounter = 1;

        this._pushToUndo({
            type: "init",
            fromNodeId: null,
            toNodeId: this.selectedNodeId,
            deltas: {
                tree: treeDelta,
                game: this._combineDeltas(allGameDeltas),
            },
        });
    }

    public saveSession(): SerializedSession {
        return {
            config: this.config,
            selectedNodeId: this.selectedNodeId,
            storyTree: this.storyTree.serialize(),
            worldState: {
                state: this.worldState.getState(),
                plots: this.worldState.getPlots(),
            },
            memoryBank: this.memoryBank.export(),
            plotCards: this.plotCardManager.export(),
        };
    }

    public async loadSession(session: SerializedSession) {
        this.config = session.config;
        this.storyTree = StoryTree.deserialize(session.storyTree);
        this.worldState = new WorldState(
            session.worldState.state,
            session.worldState.plots
        );

        this.memoryBank = await MemoryBank.from(
            session.memoryBank,
            this.providerRegistry
        );
        this.plotCardManager = await PlotCardManager.from(
            session.plotCards,
            this.providerRegistry
        );
        this.selectedNodeId = session.selectedNodeId;
        this.turnCounter = this.storyTree.getDepth(this.selectedNodeId!) || 0;
        this.isInitialized = true;
        this.undoStack = [];
        this.redoStack = [];
    }

    public async act(
        playerInput: string,
        onChunk: (chunk: StreamedChunk) => void = () => {},
        onThink: (chunk: string) => void = () => {}
    ) {
        if (!this.isInitialized || !this.selectedNodeId)
            throw new Error("Engine not initialized.");

        const fromNodeId = this.selectedNodeId;
        const parentNodeId = this.selectedNodeId;

        const {
            node: pNode,
            treeDelta: pTreeDelta,
            gameDeltas: pGameDeltas,
            directorOutcome,
        } = await this._generatePlayerTurn(parentNodeId, playerInput, onThink);

        const {
            node: wNode,
            treeDelta: wTreeDelta,
            gameDeltas: wGameDeltas,
        } = await this._generateWriterTurn(pNode.id, directorOutcome, onChunk);

        this.selectedNodeId = wNode.id;
        this.turnCounter++;

        this._pushToUndo({
            type: "act",
            fromNodeId: fromNodeId,
            toNodeId: this.selectedNodeId,
            deltas: {
                tree: this._combineDeltas([pTreeDelta, wTreeDelta]),
                game: this._combineDeltas([...pGameDeltas, ...wGameDeltas]),
            },
        });
    }

    public async continue(onChunk: (chunk: StreamedChunk) => void = () => {}) {
        if (!this.isInitialized || !this.selectedNodeId)
            throw new Error("Engine not initialized.");

        const fromNodeId = this.selectedNodeId;
        const parentNodeId = this.selectedNodeId;

        const {
            node: wNode,
            treeDelta: wTreeDelta,
            gameDeltas: wGameDeltas,
        } = await this._generateWriterTurn(
            parentNodeId,
            "The story continues...",
            onChunk
        );

        this.selectedNodeId = wNode.id;
        this.turnCounter++;

        this._pushToUndo({
            type: "continue",
            fromNodeId: fromNodeId,
            toNodeId: this.selectedNodeId,
            deltas: {
                tree: wTreeDelta,
                game: this._combineDeltas(wGameDeltas),
            },
        });
    }

    public erase() {
        if (!this.isInitialized || !this.selectedNodeId)
            throw new Error("Engine not initialized.");

        const fromNodeId = this.selectedNodeId;
        const node = this.storyTree.getNode(fromNodeId);
        if (!node || !node.parentId) {
            throw new Error("Cannot erase the root node.");
        }

        const toNodeId = node.parentId;

        const gameSyncDelta = this._syncGameState(fromNodeId, toNodeId);

        const deleteResult = this.storyTree.deleteBranch(fromNodeId);
        if (!deleteResult) throw new Error("Failed to erase branch.");

        this.selectedNodeId = toNodeId;
        this.turnCounter = this.storyTree.getDepth(this.selectedNodeId) || 0;

        this._pushToUndo({
            type: "erase",
            fromNodeId: fromNodeId,
            toNodeId: toNodeId,
            deltas: {
                tree: deleteResult.delta,
                game: gameSyncDelta,
            },
        });
    }

    public async retry(
        onChunk: (chunk: StreamedChunk) => void = () => {},
        onThink: (chunk: string) => void = () => {}
    ) {
        if (!this.isInitialized || !this.selectedNodeId)
            throw new Error("Engine not initialized.");

        const fromNodeId = this.selectedNodeId;
        const node = this.storyTree.getNode(fromNodeId);
        if (!node || !node.parentId) {
            throw new Error("Cannot retry the root node.");
        }

        const parentNode = this.storyTree.getNode(node.parentId);
        if (!parentNode)
            throw new Error("Cannot find parent node to retry from.");

        if (node.turn.actor === "player") {
            throw new Error(
                "Cannot retry a Player node. Please retry the Writer node instead."
            );
        }

        const gameSyncDelta = this._syncGameState(fromNodeId, parentNode.id);

        let newNode: StoryNode;
        let newTreeDelta: DeltaPair | null;
        let newGameDeltas: (DeltaPair | null)[];

        const parentActor = parentNode.turn.actor;
        if (parentActor === "player") {
            const { directorOutcome } = await this._generatePlayerTurn(
                parentNode.parentId,
                parentNode.turn.text,
                onThink,
                true
            );

            const regen = await this._generateWriterTurn(
                parentNode.id,
                directorOutcome,
                onChunk
            );
            newNode = regen.node;
            newTreeDelta = regen.treeDelta;
            newGameDeltas = regen.gameDeltas;
        } else {
            const regen = await this._generateWriterTurn(
                parentNode.id,
                "The story continues...",
                onChunk
            );
            newNode = regen.node;
            newTreeDelta = regen.treeDelta;
            newGameDeltas = regen.gameDeltas;
        }

        this.selectedNodeId = newNode.id;

        this._pushToUndo({
            type: "retry",
            fromNodeId: fromNodeId,
            toNodeId: this.selectedNodeId,
            deltas: {
                tree: newTreeDelta,
                game: this._combineDeltas([gameSyncDelta, ...newGameDeltas]),
            },
        });
    }

    public select(nodeId: string) {
        if (nodeId === this.selectedNodeId) return;
        if (!this.storyTree.getNode(nodeId))
            throw new Error("Node does not exist.");

        const fromNodeId = this.selectedNodeId;
        const toNodeId = nodeId;

        const gameSyncDelta = this._syncGameState(fromNodeId, toNodeId);
        this.selectedNodeId = toNodeId;
        this.turnCounter = this.storyTree.getDepth(toNodeId) || 0;

        this._pushToUndo({
            type: "select",
            fromNodeId: fromNodeId,
            toNodeId: toNodeId,
            deltas: {
                tree: null,
                game: gameSyncDelta,
            },
        });
    }

    public switch(direction: "next" | "prev") {
        if (!this.selectedNodeId) return;
        const node = this.storyTree.getNode(this.selectedNodeId);
        if (!node || !node.parentId) return;

        const parent = this.storyTree.getNode(node.parentId);
        if (!parent || parent.childrenIds.length < 2) return;

        const siblings = parent.childrenIds;
        const currentIndex = siblings.indexOf(this.selectedNodeId);
        let newIndex =
            direction === "next" ? currentIndex + 1 : currentIndex - 1;

        if (newIndex < 0) newIndex = siblings.length - 1;
        if (newIndex >= siblings.length) newIndex = 0;

        this.select(siblings[newIndex]);
    }

    public async edit(newText: string) {
        if (!this.isInitialized || !this.selectedNodeId)
            throw new Error("Engine not initialized.");

        const nodeId = this.selectedNodeId;
        const node = this.storyTree.getNode(nodeId);
        if (!node) throw new Error("No node selected.");

        const oldTurn = node.turn;
        const oldGameDeltas = node.deltas;

        const newTurn = { ...oldTurn, text: newText };
        let newGameDeltas: DeltaPair[] = oldGameDeltas;

        if (node.turn.actor === "storywriter" && this.config.directorEnabled) {
            this._applyGameDeltas(oldGameDeltas, "revert");

            const { gameDeltas } = await this._runWriterAssessment(
                newText,
                nodeId
            );
            newGameDeltas = gameDeltas;

            this._applyGameDeltas(newGameDeltas, "apply");
        }

        const treeUpdateDelta = this.storyTree.updateNode(
            nodeId,
            newTurn,
            newGameDeltas
        );

        const gameDelta: DeltaPair = {
            apply: [
                ...newGameDeltas.flatMap((d) => d.apply),
                ...oldGameDeltas.flatMap((d) => d.revert),
            ],
            revert: [
                ...oldGameDeltas.flatMap((d) => d.apply),
                ...newGameDeltas.flatMap((d) => d.revert),
            ],
        };

        this._pushToUndo({
            type: "edit",
            fromNodeId: nodeId,
            toNodeId: nodeId,
            deltas: {
                tree: treeUpdateDelta,
                game: gameDelta,
            },
        });
    }

    public undo() {
        const action = this.undoStack.pop();
        if (!action) return;

        if (action.deltas.game) {
            this._applyGameDeltas([action.deltas.game], "revert");
        }
        if (action.deltas.tree) {
            this.storyTree.applyDelta(action.deltas.tree.revert);
        }

        this.selectedNodeId = action.fromNodeId;
        this.turnCounter = this.storyTree.getDepth(this.selectedNodeId!) || 0;
        this.redoStack.push(action);
    }

    public redo() {
        const action = this.redoStack.pop();
        if (!action) return;

        if (action.deltas.tree) {
            this.storyTree.applyDelta(action.deltas.tree.apply);
        }
        if (action.deltas.game) {
            this._applyGameDeltas([action.deltas.game], "apply");
        }

        this.selectedNodeId = action.toNodeId;
        this.turnCounter = this.storyTree.getDepth(this.selectedNodeId!) || 0;
        this.undoStack.push(action);
    }

    private _syncGameState(
        fromId: string | null,
        toId: string | null
    ): DeltaPair | null {
        if (fromId === toId) return null;

        const fromPath = fromId ? this.storyTree.getPathToNode(fromId) : [];
        const toPath = toId ? this.storyTree.getPathToNode(toId) : [];

        let commonAncestorIndex = -1;
        for (let i = 0; i < Math.min(fromPath.length, toPath.length); i++) {
            if (fromPath[i].id === toPath[i].id) {
                commonAncestorIndex = i;
            } else {
                break;
            }
        }

        const revertPath = fromPath.slice(commonAncestorIndex + 1).reverse();
        const revertDeltas = revertPath.flatMap((node) => node.deltas);

        const applyPath = toPath.slice(commonAncestorIndex + 1);
        const applyDeltas = applyPath.flatMap((node) => node.deltas);

        if (revertDeltas.length === 0 && applyDeltas.length === 0) return null;

        this._applyGameDeltas(revertDeltas, "revert");
        this._applyGameDeltas(applyDeltas, "apply");

        const gameSyncDelta: DeltaPair = {
            apply: [
                ...applyDeltas.flatMap((d) => d.apply),
                ...revertDeltas.flatMap((d) => d.revert),
            ],
            revert: [
                ...revertDeltas.flatMap((d) => d.apply),
                ...applyDeltas.flatMap((d) => d.revert),
            ],
        };

        return gameSyncDelta;
    }

    private _applyGameDeltas(
        deltas: (DeltaPair | null)[],
        direction: "apply" | "revert"
    ) {
        for (const delta of deltas) {
            if (!delta) continue;
            const ops = direction === "apply" ? delta.apply : delta.revert;
            this.worldState.applyDelta(ops);
            this.memoryBank.applyDelta(ops);
        }
    }

    private async _generatePlayerTurn(
        parentNodeId: string,
        playerInput: string,
        onThink: (chunk: string) => void,
        isRetry: boolean = false
    ): Promise<{
        node: StoryNode;
        treeDelta: DeltaPair | null;
        gameDeltas: DeltaPair[];
        directorOutcome: string;
    }> {
        let directorOutcome = "Player action taken.";
        let gameDeltas: DeltaPair[] = [];
        let directorThoughts = "";

        if (this.config.directorEnabled) {
            const directorModel = this.config.models.gameDirector;
            const context = await this.contextBuilder.buildDirectorContext(
                playerInput,
                parentNodeId,
                this.turnCounter
            );
            const directorStream = this.gameDirector.assessPlayerTurnStream(
                context,
                directorModel.model,
                directorModel.options
            );

            for await (const { tool_calls, thinking } of directorStream) {
                if (tool_calls) {
                    const call_results = await this._processToolCalls(
                        tool_calls
                    );
                    const deltas = call_results.filter(
                        (r) => typeof r !== "string"
                    );
                    gameDeltas.push(...deltas);
                    directorOutcome =
                        call_results.findLast((r) => typeof r === "string") ??
                        "";
                }
                if (thinking) {
                    directorThoughts += thinking;
                    onThink(thinking);
                }
            }

            if (!isRetry) {
                this._applyGameDeltas(gameDeltas, "apply");
            }
        }

        const pNode: StoryNode = {
            id: crypto.randomUUID(),
            parentId: parentNodeId,
            childrenIds: [],
            turn: {
                actor: "player",
                text: playerInput,
                directorThinking: directorThoughts,
            },
            deltas: gameDeltas,
        };

        let pTreeDelta: DeltaPair | null = null;
        if (!isRetry) {
            pTreeDelta = this.storyTree.addNode(pNode);
        }

        return {
            node: pNode,
            treeDelta: pTreeDelta,
            gameDeltas: gameDeltas,
            directorOutcome: directorOutcome,
        };
    }

    private async _runWriterAssessment(
        writerText: string,
        contextNodeId: string
    ): Promise<{
        gameDeltas: DeltaPair[];
    }> {
        if (!this.config.directorEnabled) {
            return { gameDeltas: [] };
        }

        const gameDeltas: DeltaPair[] = [];
        const directorModel = this.config.models.gameDirector;
        const context =
            await this.contextBuilder.buildDirectorPostWriterContext(
                writerText,
                contextNodeId,
                this.turnCounter
            );
        const directorStream = this.gameDirector.assessWriterTurnStream(
            context,
            directorModel.model,
            directorModel.options
        );

        for await (const { tool_calls } of directorStream) {
            if (tool_calls) {
                const deltas = (
                    await this._processToolCalls(tool_calls)
                ).filter((r) => typeof r !== "string");
                gameDeltas.push(...deltas);
            }
        }

        return { gameDeltas };
    }

    private async _runMemoryGeneration(
        contextNodeId: string
    ): Promise<DeltaPair | null> {
        if (
            this.config.memoryGenerationInterval <= 0 ||
            this.turnCounter % this.config.memoryGenerationInterval !== 0
        ) {
            return null;
        }

        const nodes = this.storyTree.getRecentTurns(
            contextNodeId,
            this.config.memoryGenerationInterval * 2
        );

        const { deltaPair } = await this.memoryBank.generateAndAddMemory(
            nodes,
            this.turnCounter,
            this.config.models.memorySummarizer.options
        );
        return deltaPair;
    }

    private async _generateWriterTurn(
        parentNodeId: string,
        directorOutcome: string,
        onChunk: (chunk: StreamedChunk) => void
    ): Promise<{
        node: StoryNode;
        treeDelta: DeltaPair | null;
        gameDeltas: (DeltaPair | null)[];
    }> {
        const writerModel = this.config.models.storyWriter;
        const writerContext = await this.contextBuilder.buildWriterContext(
            directorOutcome,
            parentNodeId,
            this.turnCounter
        );
        const writerStream = this.storyWriter.writeNextTurnStream(
            writerContext,
            writerModel.model,
            writerModel.options
        );
        let writerText = "";
        for await (const chunk of writerStream) {
            if (chunk.delta.content) {
                writerText += chunk.delta.content;
            }
            onChunk(chunk);
        }

        const { gameDeltas: writerGameDeltas } =
            await this._runWriterAssessment(writerText, parentNodeId);

        const memDelta = await this._runMemoryGeneration(parentNodeId);

        const allGameDeltas = [...writerGameDeltas, memDelta].filter(
            (d): d is DeltaPair => d !== null
        );

        this._applyGameDeltas(allGameDeltas, "apply");

        const wNode: StoryNode = {
            id: crypto.randomUUID(),
            parentId: parentNodeId,
            childrenIds: [],
            turn: { actor: "storywriter", text: writerText },
            deltas: allGameDeltas,
        };

        const wTreeDelta = this.storyTree.addNode(wNode);

        return {
            node: wNode,
            treeDelta: wTreeDelta,
            gameDeltas: allGameDeltas,
        };
    }

    private _pushToUndo(action: EngineAction) {
        this.undoStack.push(action);
        this.redoStack = [];
    }

    private _combineDeltas(deltas: (DeltaPair | null)[]): DeltaPair | null {
        const validDeltas = deltas.filter((d): d is DeltaPair => d !== null);
        if (validDeltas.length === 0) return null;

        const combinedApply = validDeltas.flatMap((d) => d.apply);
        const combinedRevert = validDeltas.flatMap((d) => d.revert).reverse();

        return {
            apply: combinedApply,
            revert: combinedRevert,
        };
    }

    private async _processToolCalls(
        toolCalls: FunctionCall[]
    ): Promise<(DeltaPair | string)[]> {
        const deltas: (DeltaPair | string)[] = [];

        for (const call of toolCalls) {
            const args = JSON.parse(call.function.arguments);
            switch (call.function.name) {
                case "patchState":
                    deltas.push(this.worldState.patchState(args.partialState));
                    break;
                case "addPlot":
                    deltas.push(
                        this.worldState.addPlot({
                            ...args,
                            created_at_turn: this.turnCounter,
                        }).delta
                    );
                    break;
                case "updatePlot":
                    deltas.push(
                        this.worldState.updatePlot(args.plotId, args.updates)!
                    );
                    break;
                case "removePlot":
                    deltas.push(this.worldState.removePlot(args.plotId)!);
                    break;
                case "determineActionResult":
                    deltas.push(
                        `${args.success ? "Success" : "Failure"}: ${
                            args.outcomeNote
                        }`
                    );
                    break;
            }
        }
        return deltas;
    }
}

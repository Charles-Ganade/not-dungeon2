import { ProviderRegistry } from "./lib/ai/provider";
import { OllamaProvider } from "./lib/ai/providers/ollama_provider";
import { ContextBuilder } from "./lib/models/context_builder";
import { GameDirector } from "./lib/models/game_director";
import { MemoryBank } from "./lib/models/memory_bank";
import { PlotCardManager } from "./lib/models/plot_card_manager";
import { StoryTurn, StoryTree, StoryNode } from "./lib/models/story_tree";
import { WorldState } from "./lib/models/world_state";

async function runPlotCardDemo() {
    console.log("Setting up providers and PlotCardManager...");

    const registry = new ProviderRegistry();
    registry.register(new OllamaProvider());
    const embeddingModel = "nomic-embed-text";

    const plotCardManager = new PlotCardManager(registry, embeddingModel, 768);
    await plotCardManager.init();

    await plotCardManager.clear();

    console.log("PlotCardManager initialized.");

    console.log("\nAdding plot cards...");
    await plotCardManager.addPlotCard({
        category: "Character",
        name: "Gandalf",
        content:
            "An old, wise wizard with a long grey beard and staff. He guides the fellowship.",
        triggerKeyword: "Gandalf",
    });
    await plotCardManager.addPlotCard({
        category: "Location",
        name: "Rivendell",
        content:
            "An ancient Elven sanctuary hidden in a valley. Home of Elrond.",
        triggerKeyword: "Rivendell",
    });
    await plotCardManager.addPlotCard({
        category: "Item",
        name: "The One Ring",
        content:
            "A powerful artifact created by Sauron. It grants invisibility but corrupts its bearer.",
        triggerKeyword: "Ring",
    });
    console.log("Plot cards added:", plotCardManager.getAllPlotCards());

    console.log("\n--- Search 1: Keyword Trigger ---");
    const query1 = "What does Gandalf look like?";
    console.log(`Searching for: "${query1}"`);
    const results1 = await plotCardManager.search(query1, 3);
    console.log("Results:", results1);

    console.log("\n--- Search 2: Semantic Similarity ---");
    const query2 = "Tell me about the hidden elven city.";
    console.log(`Searching for: "${query2}"`);
    const results2 = await plotCardManager.search(query2, 3);
    console.log("Results:", results2);

    console.log("\n--- Search 3: Keyword and Semantic ---");
    const query3 = "Where is the powerful Ring kept?";
    console.log(`Searching for: "${query3}"`);
    const results3 = await plotCardManager.search(query3, 3);
    console.log("Results:", results3);

    console.log("\nRemoving all cards...");
    const cards = plotCardManager.getAllPlotCards().map(async (card) => {
        await plotCardManager.removePlotCard(card.id!);
    });
    await Promise.all(cards);
}

async function runMemoryBankDemo() {
    console.log("Setting up providers and MemoryBank...");

    const registry = new ProviderRegistry();
    registry.register(new OllamaProvider());

    const embeddingModel = "nomic-embed-text";
    const summarizerModel = "qwen3-4b-josiefied";
    const memoryBank = new MemoryBank(
        registry,
        embeddingModel,
        summarizerModel,
        768
    );

    await memoryBank.init();

    await memoryBank.clear();

    console.log("MemoryBank initialized.");
    console.log("Initial memories loaded:", memoryBank.getAll());

    let currentTurn = 1;

    console.log(`\n--- Adding memories at Turn ${currentTurn} ---`);
    await memoryBank.addMemory(
        "The player entered the Prancing Pony inn.",
        currentTurn
    );
    await memoryBank.addMemory(
        "A mysterious ranger named Strider sits in the corner.",
        currentTurn
    );
    await memoryBank.addMemory(
        "Barliman Butterbur is the innkeeper.",
        currentTurn
    );

    console.log("Memories after adding:", memoryBank.getAll());
    currentTurn++;

    console.log(`\n--- Generating memory at Turn ${currentTurn} ---`);
    const dummyTurns: StoryTurn[] = [
        {
            actor: "player",
            text: "I approach Strider and ask about Gandalf.",
        },
        {
            actor: "storywriter",
            text: "Strider looks up, surprised. 'Gandalf? I haven't seen him in weeks. Why do you ask?'",
        },
    ];
    await memoryBank.generateAndAddMemory(dummyTurns, currentTurn);
    console.log("Memories after generating:", memoryBank.getAll());
    currentTurn++;

    console.log(`\n--- Searching at Turn ${currentTurn} ---`);
    const query = "Who is the innkeeper?";
    console.log(`Searching for: "${query}"`);
    const results = await memoryBank.search(query, currentTurn, 5);
    console.log("Search Results:", results);

    console.log(
        "Memories after search (check last_accessed_at_turn):",
        memoryBank.getAll()
    );
    currentTurn++;

    console.log(`\n--- Delta Demo at Turn ${currentTurn} ---`);
    const { newId: tempMemoryId, deltaPair } = await memoryBank.addMemory(
        "A Nazgul screeches outside the inn.",
        currentTurn
    );
    console.log(
        `Added temporary memory (ID: ${tempMemoryId}). Current memories:`,
        memoryBank.getAll()
    );

    console.log("\nUndoing addMemory...");
    await memoryBank.applyDelta(deltaPair.revert);
    console.log("Memories after undo:", memoryBank.getAll());

    console.log("\nRedoing addMemory...");
    await memoryBank.applyDelta(deltaPair.apply);
    console.log(
        "Memories after redo (check if Nazgul memory is back, ID may differ):",
        memoryBank.getAll()
    );

    console.log("\n--- Demo Complete ---");
}

async function runWorldStateDemo() {
    console.log("--- Running WorldState Demo ---");

    console.log("\n--- Test: constructor & getState/getPlots ---");
    const initialState = {
        player: {
            name: "Hero",
            hp: 80,
        },
        location: "Start",
    };
    const initialPlots = [
        {
            id: "plot-001",
            title: "Find the Amulet",
            description: "An old man asked you to find his lost amulet.",
            player_alignment: 0.5,
            created_at_turn: 0,
        },
    ];

    const worldState = new WorldState(initialState, initialPlots);
    console.log("Initial State:", worldState.getState());
    console.log("Initial Plots:", worldState.getPlots());

    console.log("\n--- Test: toText ---");
    console.log(worldState.toText());

    console.log("\n--- Test: deepSet ---");
    const delta1 = worldState.deepSet("player/hp", 100);
    console.log("State after deepSet:", worldState.toText());
    console.log("deepSet apply delta:", delta1.apply);

    console.log("\n--- Test: patchState ---");
    const delta2 = worldState.patchState({
        player: { inventory: ["sword"] },
        location: "Tavern",
    });
    console.log("State after patchState:", worldState.toText());
    console.log("patchState apply delta:", delta2.apply);

    console.log("\n--- Test: addPlot ---");
    const { newId: plotId, delta: delta3 } = worldState.addPlot({
        title: "Main Quest",
        description: "Defeat the dragon.",
        player_alignment: 0.1,
        created_at_turn: 1,
    });
    console.log("Plots after addPlot:", worldState.getPlots());
    console.log("New Plot ID:", plotId);

    console.log("\n--- Test: updatePlot ---");
    const delta4 = worldState.updatePlot(plotId, {
        description: "The dragon lives in the high tower.",
        player_alignment: 0.15,
    });
    console.log(
        "Plots after updatePlot:",
        worldState.getPlots().find((p) => p.id === plotId)
    );

    console.log("\n--- Test: removePlot ---");
    const delta5 = worldState.removePlot(plotId);
    console.log("Plots after removePlot:", worldState.getPlots());

    console.log("\n--- Test: applyDelta (Undo removePlot) ---");
    worldState.applyDelta(delta5!.revert);
    console.log("Plots after undoing remove:", worldState.getPlots());

    console.log("\n--- Test: applyDelta (Undo patchState) ---");
    worldState.applyDelta(delta2.revert);
    console.log("State after undoing patchState:", worldState.toText());

    console.log("\n--- Test: applyDelta (Undo deepSet) ---");
    worldState.applyDelta(delta1.revert);
    console.log(
        "State after undoing deepSet (HP should be 80):",
        worldState.toText()
    );

    console.log("--- WorldState Demo Complete ---");
}

async function runStoryTreeDemo() {
    console.log("--- Running StoryTree Demo ---");

    console.log("\n--- Test: constructor ---");
    const tree = new StoryTree();

    console.log("\n--- Test: addNode (Root) ---");
    const rootNode: StoryNode = {
        id: "root-0",
        parentId: "",
        childrenIds: [],
        turn: { actor: "storywriter", text: "The story begins." },
        deltas: [],
    };
    const delta1 = tree.addNode(rootNode);
    console.log("Root Node:", tree.getRootNode());

    console.log("\n--- Test: addNode (Children) ---");
    const childNode1: StoryNode = {
        id: "child-1",
        parentId: "root-0",
        childrenIds: [],
        turn: { actor: "player", text: "I go left." },
        deltas: [],
    };
    const childNode2: StoryNode = {
        id: "child-2",
        parentId: "root-0",
        childrenIds: [],
        turn: { actor: "player", text: "I go right." },
        deltas: [],
    };
    tree.addNode(childNode1);
    tree.addNode(childNode2);

    const grandChildNode: StoryNode = {
        id: "grandchild-1",
        parentId: "child-1",
        childrenIds: [],
        turn: { actor: "storywriter", text: "You find a cave." },
        deltas: [],
    };
    const deltaGC = tree.addNode(grandChildNode);
    console.log("Root node children:", tree.getRootNode()?.childrenIds);

    console.log("\n--- Test: getNode ---");
    console.log("Get child-1:", tree.getNode("child-1"));

    console.log("\n--- Test: getPathToNode ---");
    const path = tree.getPathToNode("grandchild-1");
    console.log(
        "Path to grandchild-1:",
        path.map((n) => n.id)
    );

    console.log("\n--- Test: getDepth ---");
    console.log("Depth of grandchild-1:", tree.getDepth("grandchild-1")); // Should be 3

    console.log("\n--- Test: getRecentTurns ---");
    console.log(
        "Recent turns for grandchild-1 (n=2):",
        tree.getRecentTurns("grandchild-1", 2)
    );

    console.log("\n--- Test: getNodesAtTurn ---");
    console.log(
        "Nodes at Turn 2 (depth 2):",
        tree.getNodesAtTurn(2).map((n) => n.id)
    );

    console.log("\n--- Test: getDeepestNode ---");
    console.log("Deepest Node:", tree.getDeepestNode()?.id);

    console.log("\n--- Test: editNode ---");
    const deltaEdit = tree.editNode("child-1", {
        actor: "player",
        text: "I go left (Edited).",
    });
    console.log("Edited node turn:", tree.getNode("child-1")?.turn);

    console.log("\n--- Test: applyDelta (Undo editNode) ---");
    tree.applyDelta(deltaEdit!.revert);
    console.log("Reverted node turn:", tree.getNode("child-1")?.turn);

    console.log("\n--- Test: serialize & deserialize ---");
    const serialized = tree.serialize();
    console.log(
        "Serialized tree nodes:",
        serialized.nodes.map((n) => n[0])
    );
    const newTree = StoryTree.deserialize(serialized);
    console.log("Deserialized deepest node:", newTree.getDeepestNode()?.id);

    console.log("\n--- Test: deleteBranch ---");
    const deleted = tree.deleteBranch("child-1")!;
    console.log(
        "Deleted nodes:",
        deleted.deletedNodes.map((n) => n.id)
    ); // Should be [child-1, grandchild-1]
    console.log("Root children after delete:", tree.getRootNode()?.childrenIds); // Should be [child-2]

    console.log("\n--- Test: applyDelta (Undo deleteBranch) ---");
    tree.applyDelta(deleted.delta.revert);
    console.log(
        "Root children after undo delete:",
        tree.getRootNode()?.childrenIds
    ); // Should be [child-2, child-1]
    console.log("Node grandchild-1 re-exists:", !!tree.getNode("grandchild-1"));

    console.log("--- StoryTree Demo Complete ---");
}

async function runContextBuilderDemo() {
    console.log("--- Running ContextBuilder Demo ---");

    console.log("\n--- Setup: Initializing all dependencies ---");
    const registry = new ProviderRegistry();
    registry.register(new OllamaProvider());
    const embeddingModel = "nomic-embed-text";
    const summarizerModel = "llama3";

    const memoryBank = new MemoryBank(
        registry,
        embeddingModel,
        summarizerModel,
        768
    );
    const plotCardManager = new PlotCardManager(registry, embeddingModel, 768);

    await memoryBank.init();
    await plotCardManager.init();
    await memoryBank.clear();
    await plotCardManager.clear();

    const worldState = new WorldState();
    const storyTree = new StoryTree();

    console.log("\n--- Setup: Populating dependencies with data ---");

    // 1. Populate WorldState
    worldState.patchState({
        player: { name: "Hero", location: "Bree" },
        npcs: { Strider: "Tavern Corner" },
    });
    worldState.addPlot({
        title: "Find Gandalf",
        description: "Strider says Gandalf is missing.",
        player_alignment: 0.5,
        created_at_turn: 0,
    });

    // 2. Populate StoryTree
    const rootNode: StoryNode = {
        id: "root",
        parentId: "",
        childrenIds: [],
        turn: { actor: "storywriter", text: "You enter the Prancing Pony." },
        deltas: [],
    };
    const playerNode: StoryNode = {
        id: "player-1",
        parentId: "root",
        childrenIds: [],
        turn: { actor: "player", text: "I look for Strider." },
        deltas: [],
    };
    storyTree.addNode(rootNode);
    storyTree.addNode(playerNode);

    // 3. Populate MemoryBank
    await memoryBank.addMemory("The player is in Bree looking for Gandalf.", 1);

    // 4. Populate PlotCardManager
    await plotCardManager.addPlotCard({
        category: "Character",
        name: "Strider",
        content:
            "A grim ranger, also known as Aragorn. He is waiting for the player.",
        triggerKeyword: "Strider",
    });

    console.log("\n--- Setup: Initializing ContextBuilder ---");
    const contextBuilder = new ContextBuilder(
        worldState,
        storyTree,
        memoryBank,
        plotCardManager
    );

    const currentNodeId = "player-1";
    const currentTurn = 2;

    console.log("\n--- Test: buildDirectorContext ---");
    const directorCtx = await contextBuilder.buildDirectorContext(
        "I ask Strider about the missing wizard.",
        currentNodeId,
        currentTurn
    );
    console.log(
        "=== Director Context Start ===\n",
        directorCtx,
        "\n=== Director Context End ==="
    );

    console.log("\n--- Test: buildWriterContext ---");
    const directorOutcome =
        "The player successfully finds Strider, who seems to recognize them.";
    const writerCtx = await contextBuilder.buildWriterContext(
        directorOutcome,
        currentNodeId,
        currentTurn
    );
    console.log(
        "=== Writer Context Start ===\n",
        writerCtx,
        "\n=== Writer Context End ==="
    );

    console.log("\n--- Test: buildDirectorPostWriterContext ---");
    const writerOutput =
        "'Gandalf has not been seen for weeks,' Strider says, his voice low. 'I fear something is wrong.'";
    const postWriterCtx = await contextBuilder.buildDirectorPostWriterContext(
        writerOutput,
        currentNodeId,
        currentTurn
    );
    console.log(
        "=== Post-Writer Context Start ===\n",
        postWriterCtx,
        "\n=== Post-Writer Context End ==="
    );

    console.log("\n--- Cleanup: Clearing DBs ---");
    await memoryBank.clear();
    await plotCardManager.clear();
    console.log("--- ContextBuilder Demo Complete ---");
}

async function runGameDirectorDemo() {
    console.log("--- Running GameDirector Demo ---");

    console.log("\n--- Setup: Initializing Provider and GameDirector ---");
    const registry = new ProviderRegistry();
    registry.register(new OllamaProvider());
    const gameDirector = new GameDirector(registry);
    const model = "qwen3-4b-josiefied"; // Model for the director

    console.log(`--- Using Model: ${model} ---`);

    console.log("\n\n--- Test 1: initializeWorldStream ---");
    const initContext =
        "A dark and stormy night. The player is a detective named Miles, standing outside a spooky mansion where a murder just occurred. He has a flashlight and a notepad.";
    console.log(`Context: "${initContext}"`);

    try {
        const generator = gameDirector.initializeWorldStream(
            initContext,
            model
        );

        for await (const tools of generator) {
            tools?.forEach((tool) => {
                console.log("Tool Call: ", tool);
            });
        }
    } catch (e) {
        console.error(
            "Failed to run initializeWorldStream. Is Ollama running?",
            e
        );
        return;
    }

    console.log("\n\n--- Test 2: assessPlayerTurnStream ---");
    const playerTurnContext = `
## Game Context (Turn 5)

### Recent Story History
storywriter: You are in the dark library. A large window shows the raging storm outside.
player: I search the desk for a key.
storywriter: You find nothing but an old letter.
player: I read the letter.

### Player's Current Action
I check the window for any clues.

### Relevant Memories
1. The butler, Jeeves, seemed nervous.

### Relevant Plot Cards (World Info)
1. [Location: Library] A large, dusty room with a single desk and a large bay window.

### Current World State & Plots
--- World State ---
{
  "player": {
    "name": "Miles",
    "location": "Library",
    "inventory": ["flashlight", "notepad", "old letter"]
  },
  "world": {
    "time": "10:00 PM"
  }
}

--- Active Plotlines ---
- Find the Murder Weapon: The weapon used to kill Lord Blackwood is missing. (Player Alignment: 0.80)
    `;

    try {
        const playerTurnResultGenerator = gameDirector.assessPlayerTurnStream(
            playerTurnContext,
            model
        );

        let thinking = "";
        let tool_calls = [];

        console.log("Result (Thinking):");
        for await (const result of playerTurnResultGenerator) {
            thinking += result.thinking;
            tool_calls.push(result.tool_calls);
            console.log(`${thinking} \r`);
        }
        tool_calls = tool_calls.filter((v) => v != undefined).flat();
        console.log("Result (Tool Calls): ", tool_calls);
    } catch (e) {
        console.error("Failed to run assessPlayerTurnStream.", e);
    }

    console.log("\n\n--- Test 3: assessWriterTurn ---");
    const writerTurnContext = `
## Post-Narration Assessment (Turn 5)

### Story Text Just Written:
Miles walks to the window, his flashlight beam cutting through the gloom. He notices the latch is broken, and a small, muddy footprint is visible on the windowsill. He pulls out his notepad and jots this down, a new lead forming in his mind.

### Current World State:
{
  "player": {
    "name": "Miles",
    "location": "Library",
    "inventory": ["flashlight", "notepad", "old letter"]
  }
}
    `;

    try {
        const writerTurnResultGenerator = gameDirector.assessWriterTurnStream(
            writerTurnContext,
            model
        );
        let thinking = "";
        let tool_calls = [];

        console.log("Result (Thinking):");
        for await (const result of writerTurnResultGenerator) {
            thinking += result.thinking;
            tool_calls.push(result.tool_calls);
            console.log(`${thinking} \r`);
        }
        tool_calls = tool_calls.filter((v) => v != undefined).flat();
        console.log("Result (Tool Calls): ", tool_calls);
        console.log("--- GameDirector Demo Complete ---");
    } catch (e) {
        console.error("Failed to run assessWriterTurn.", e);
    }
}

export {
    runContextBuilderDemo,
    runMemoryBankDemo,
    runPlotCardDemo,
    runStoryTreeDemo,
    runWorldStateDemo,
    runGameDirectorDemo,
};

import type { Component } from "solid-js";
import { ProviderRegistry } from "./lib/ai/provider";
import { OllamaProvider } from "./lib/ai/providers/ollama_provider";
import { PlotCardManager } from "./lib/models/plot_card_manager";
import { MemoryBank } from "./lib/models/memory_bank";
import { StoryTurn } from "./lib/models/story_tree";

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
    const summarizerModel = "Qwen3-4b";
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

const App: Component = () => {
    // runMemoryBankDemo().catch(console.error);
    runPlotCardDemo().catch(console.error);
    return (
        <p class="text-4xl text-green-700 text-center py-20">Hello tailwind!</p>
    );
};

export default App;

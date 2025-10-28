import { ProviderRegistry } from "../ai/provider";
import { applyPatch, compare, Operation } from "fast-json-patch";
import { deepCopy } from "../util/objects";
import { DeltaPair } from "./world_state";
import { StoryTurn } from "./story_tree";
import { LocalVectorDB } from "../vectordb";

export interface Memory {
    id?: number;
    text: string;
    created_at_turn: number;
    last_accessed_at_turn: number;
}

interface MemoryRecord {
    id: number; // ID assigned by LocalVectorDB
    vector: Float32Array; // Stored as ArrayBuffer in DB, but handled as Float32Array in LocalVectorDB methods
    meta: {
        text: string;
        created_at_turn: number;
        last_accessed_at_turn: number;
    };
    createdAt?: number;
    updatedAt?: number;
}

export class MemoryBank {
    private memories: Memory[] = [];
    private db: LocalVectorDB | null = null;

    private providerRegistry: ProviderRegistry;
    private embeddingModel: string;
    private summarizerModel: string;
    private dbName = "MemoryBankDB"; // Define a specific DB name
    private vectorDimension: number | undefined; // We need to know the dimension

    constructor(
        providerRegistry: ProviderRegistry,
        embeddingModel: string,
        summarizerModel: string,
        vectorDimension: number
    ) {
        this.providerRegistry = providerRegistry;
        this.embeddingModel = embeddingModel;
        this.summarizerModel = summarizerModel;
        this.vectorDimension = vectorDimension; // Store dimension

        // Initialize LocalVectorDB here, but don't call init() yet
        this.db = new LocalVectorDB({
            dbName: this.dbName,
            dimension: this.vectorDimension,
            // Add other LocalVectorDB options if needed, e.g.,
            normalize: true, // Assuming default is okay
            idField: "id", // Assuming default 'id' is okay
        });
    }

    public async init() {
        if (!this.db) throw new Error("LocalVectorDB instance not created.");

        // Init LocalVectorDB (opens connection, runs migrations)
        await this.db.init();

        // Hydrate the in-memory cache after DB init
        await this.hydrateInMemoryCache();
    }

    // Helper to load all data from DB into the in-memory this.memories
    private async hydrateInMemoryCache() {
        if (!this.db) throw new Error("MemoryBank not initialized.");
        const allRecords = await this.db.export(); // Use export to get all data
        this.memories = allRecords.vectors.map((record) => ({
            id: record.id, // Or record[this.db.options.idField] if using custom
            text: record.meta.text,
            created_at_turn: record.meta.created_at_turn,
            last_accessed_at_turn: record.meta.last_accessed_at_turn,
        }));
    }

    private _createDeltaPair(
        mutator: (draft: { memories: Memory[] }) => boolean | void,
        canFail: boolean = false
    ): DeltaPair | null {
        // This function now only operates on the in-memory `this.memories` cache.
        // The actual DB operations happen in addMemory/removeMemory.
        const oldFullObject = { memories: deepCopy(this.memories) }; // Ensure deep copy
        const newFullObject = deepCopy(oldFullObject);

        const result = mutator(newFullObject);
        if (canFail && !result) return null;

        const apply = compare(oldFullObject, newFullObject);
        const revert = compare(newFullObject, oldFullObject);

        // Update the in-memory cache immediately
        this.memories = newFullObject.memories;

        return { apply, revert };
    }

    // Refactored applyDelta: Applies changes primarily to the DB based on
    // analyzing the difference between current in-memory state and patched state.
    public async applyDelta(delta: Operation[]) {
        if (!this.db) throw new Error("MemoryBank not initialized.");

        // 1. Get the current state (from in-memory cache)
        const oldMemories = deepCopy(this.memories);

        // 2. Apply patch to a *copy* to determine the target state
        const patched = applyPatch(
            { memories: deepCopy(oldMemories) }, // Apply to a copy
            delta,
            true, // validate operations
            false // mutate document directly (it's a copy)
        ).newDocument;
        const targetMemories = patched.memories;

        // 3. Determine differences
        const oldIds = new Set(oldMemories.map((m) => m.id));
        const targetIds = new Set(targetMemories.map((m) => m.id));

        const addedMemories = targetMemories.filter((m) => !oldIds.has(m.id));
        const removedIds = [...oldIds].filter((id) => !targetIds.has(id));
        // Note: Updates to existing memories are harder to track this way.
        // This assumes deltas are only simple adds/removes for undo/redo.

        // 4. Apply DB operations
        const removalPromises = removedIds.map((id) => this.db!.delete(id));

        const additionPromises = addedMemories.map(async (memToAdd) => {
            const { embeddings } = await this.providerRegistry.embed({
                model: this.embeddingModel,
                input: memToAdd.text,
            });
            const vector = embeddings[0];
            // ID might already exist if it's a "redo" of an add
            await this.db!.insert({
                id: memToAdd.id, // Pass existing ID if it has one
                vector: vector,
                meta: {
                    text: memToAdd.text,
                    created_at_turn: memToAdd.created_at_turn,
                    last_accessed_at_turn: memToAdd.last_accessed_at_turn,
                },
            });
        });

        await Promise.all([...removalPromises, ...additionPromises]);

        // 5. Update in-memory cache to final target state
        this.memories = targetMemories;
        // Consider re-hydrating from DB to ensure consistency if complex updates occurred
        // await this.hydrateInMemoryCache();
    }

    public async addMemory(text: string, currentTurn: number) {
        if (!this.db) throw new Error("Memory bank is not initialized");

        // 1. Embed the text
        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: text,
        });
        const vector = embeddings[0]; // Assumes single embedding

        // 2. Prepare metadata
        const meta = {
            text,
            created_at_turn: currentTurn,
            last_accessed_at_turn: currentTurn,
        };

        // 3. Insert into LocalVectorDB (ID will be auto-assigned by default)
        const newId = await this.db.insert({
            vector: vector,
            meta: meta,
        });

        // 4. Create the in-memory representation
        const newMemory: Memory = {
            id: newId, // Use the ID returned by the DB
            ...meta,
        };

        // 5. Generate delta pair based *only* on the in-memory change
        const deltaPair = this._createDeltaPair((draft) => {
            draft.memories.push(newMemory);
        });

        if (!deltaPair) {
            // Should not happen if mutator is correct
            console.error("Failed to create delta pair for addMemory");
            // Optionally attempt to delete the record from DB if delta fails?
            await this.db.delete(newId);
            throw new Error("Delta creation failed after DB insert.");
        }

        return { newId, deltaPair: deltaPair };
    }

    public async removeMemory(id: number): Promise<DeltaPair | null> {
        if (!this.db) throw new Error("Memory bank is not initialized");

        const memoryExists = this.memories.some((m) => m.id === id);
        if (!memoryExists) {
            console.warn(`Memory with id ${id} not found in cache.`);
            // Optionally check DB just in case?
            // const dbRecord = await this.db.get(id);
            // if (!dbRecord) return null;
            return null;
        }

        // 1. Perform the database operation FIRST
        try {
            await this.db.delete(id);
        } catch (error) {
            console.error(`Failed to delete memory ${id} from DB:`, error);
            return null; // Don't proceed if DB operation fails
        }

        // 2. Create the delta pair, which updates the in-memory state
        const deltaPair = this._createDeltaPair((draft) => {
            const initialLength = draft.memories.length;
            draft.memories = draft.memories.filter((m: Memory) => m.id !== id);
            // Ensure removal happened for delta generation
            return draft.memories.length < initialLength;
        }, true); // Allow failure if ID wasn't found in draft

        return deltaPair; // Might be null if not found in draft
    }

    public async generateAndAddMemory(
        turns: StoryTurn[],
        current_turn: number
    ) {
        if (!this.db) throw new Error("Memory bank is not initialized");
        const historyText = turns
            .map((t) => `${t.actor}: ${t.text}`)
            .join("\n");

        const prompt =
            "Summarize the following story segment into a single, concise memory that captures the key events, facts, or character developments. Output only the summarized memory and nothing else. Do not include your thinking process.";

        // Assuming providerRegistry.chat works as before
        const response = await this.providerRegistry.chat({
            messages: [
                { role: "system", content: prompt },
                { role: "user", content: historyText },
            ],
            model: this.summarizerModel,
            options: { temperature: 0.5 /* ... other options */ },
        });

        let cleanOutput = response.message.content
            .replace(/<think>[\s\S]*?<\/think>/gi, "")
            .trim();

        // Use the refactored addMemory
        return this.addMemory(cleanOutput, current_turn);
    }

    public async search(
        query: string,
        currentTurn: number,
        limit: number = 10
    ): Promise<Memory[]> {
        if (!this.db) throw new Error("Memory bank is not initialized");

        // 1. Embed query
        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: query,
        });
        const queryVector = embeddings[0];

        // 2. Query LocalVectorDB
        // LocalVectorDB returns { id: any; score: number; meta?: any }[]
        const results = await this.db.query(queryVector, {
            k: limit * 2, // Fetch more initially for potential recency boost
            distance: "cosine", // Or configure as needed
        });

        // 3. Process results & update last_accessed_at_turn *in memory*
        const retrievedMemoryIds = new Set<number>();
        const memoriesFromDb: Memory[] = [];

        for (const result of results) {
            const mem = this.memories.find((m) => m.id === result.id);
            if (mem) {
                // Update in-memory cache
                mem.last_accessed_at_turn = currentTurn;
                retrievedMemoryIds.add(mem.id!);
                memoriesFromDb.push({ ...mem }); // Add a copy
                // **Skipping DB update for last_accessed_at_turn for simplicity**
                // If needed, implement LocalVectorDB.update or do get/insert here
            } else {
                // Result from DB not found in memory cache - indicates inconsistency
                console.warn(
                    `Memory ID ${result.id} found in DB but not in cache.`
                );
                // Add it based on DB data
                memoriesFromDb.push({
                    id: result.id,
                    text: result.meta.text,
                    created_at_turn: result.meta.created_at_turn,
                    last_accessed_at_turn: currentTurn, // Update access turn
                });
                // Potentially add to this.memories cache here too
            }
        }

        // 4. Get most recent memories (not found by vector search) from in-memory cache
        const recentMemories = this.memories // Use the potentially updated cache
            .filter((m) => !retrievedMemoryIds.has(m.id!))
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, 5); // Consider adjusting this number

        // 5. Combine and re-rank (simple combination for now)
        // Combine DB results (already updated) and recent cached memories
        const finalMemoryMap = new Map<number, Memory>();
        memoriesFromDb.forEach((m) => finalMemoryMap.set(m.id!, m));
        recentMemories.forEach((m) => finalMemoryMap.set(m.id!, m)); // Overwrite older versions if needed

        // 6. Sort combined list by last accessed turn and take limit
        return Array.from(finalMemoryMap.values())
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, limit);
    }

    // Refactored clear to use LocalVectorDB.clear
    public async clear() {
        if (!this.db) throw new Error("MemoryBank not initialized.");

        // Clear the database
        await this.db.clear();

        // Also clear the in-memory cache
        this.memories = [];
    }

    public getAll() {
        return this.memories;
    }
}

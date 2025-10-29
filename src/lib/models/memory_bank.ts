import { ChatParams, ProviderRegistry } from "../ai/provider";
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
    id: number;
    vector: Float32Array;
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
    private dbName = "MemoryBankDB";
    private vectorDimension: number | undefined;
    constructor(
        providerRegistry: ProviderRegistry,
        embeddingModel: string,
        summarizerModel: string,
        vectorDimension: number
    ) {
        this.providerRegistry = providerRegistry;
        this.embeddingModel = embeddingModel;
        this.summarizerModel = summarizerModel;
        this.vectorDimension = vectorDimension;
        this.db = new LocalVectorDB({
            dbName: this.dbName,
            dimension: this.vectorDimension,
            normalize: true,
            idField: "id",
        });
    }

    public async init() {
        if (!this.db) throw new Error("LocalVectorDB instance not created.");

        await this.db.init();

        await this.hydrateInMemoryCache();
    }

    private async hydrateInMemoryCache() {
        if (!this.db) throw new Error("MemoryBank not initialized.");
        const allRecords = await this.db.export();
        this.memories = allRecords.vectors.map((record) => ({
            id: record.id,
            text: record.meta.text,
            created_at_turn: record.meta.created_at_turn,
            last_accessed_at_turn: record.meta.last_accessed_at_turn,
        }));
    }

    private _createDeltaPair(
        mutator: (draft: { memories: Memory[] }) => boolean | void,
        canFail: boolean = false
    ): DeltaPair | null {
        const oldFullObject = { memories: deepCopy(this.memories) };
        const newFullObject = deepCopy(oldFullObject);

        const result = mutator(newFullObject);
        if (canFail && !result) return null;

        const apply = compare(oldFullObject, newFullObject);
        const revert = compare(newFullObject, oldFullObject);

        this.memories = newFullObject.memories;

        return { apply, revert };
    }

    public async applyDelta(delta: Operation[]) {
        if (!this.db) throw new Error("MemoryBank not initialized.");

        const oldMemories = deepCopy(this.memories);

        const patched = applyPatch(
            { memories: deepCopy(oldMemories) },
            delta,
            true,
            false
        ).newDocument;
        const targetMemories = patched.memories;

        const oldIds = new Set(oldMemories.map((m) => m.id));
        const targetIds = new Set(targetMemories.map((m) => m.id));

        const addedMemories = targetMemories.filter((m) => !oldIds.has(m.id));
        const removedIds = [...oldIds].filter((id) => !targetIds.has(id));

        const removalPromises = removedIds.map((id) => this.db!.delete(id));

        const additionPromises = addedMemories.map(async (memToAdd) => {
            const { embeddings } = await this.providerRegistry.embed({
                model: this.embeddingModel,
                input: memToAdd.text,
            });
            const vector = embeddings[0];

            await this.db!.insert({
                id: memToAdd.id,
                vector: vector,
                meta: {
                    text: memToAdd.text,
                    created_at_turn: memToAdd.created_at_turn,
                    last_accessed_at_turn: memToAdd.last_accessed_at_turn,
                },
            });
        });

        await Promise.all([...removalPromises, ...additionPromises]);

        this.memories = targetMemories;
    }

    public async addMemory(text: string, currentTurn: number) {
        if (!this.db) throw new Error("Memory bank is not initialized");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: text,
        });
        const vector = embeddings[0];

        const meta = {
            text,
            created_at_turn: currentTurn,
            last_accessed_at_turn: currentTurn,
        };

        const newId = await this.db.insert({
            vector: vector,
            meta: meta,
        });

        const newMemory: Memory = {
            id: newId,
            ...meta,
        };

        const deltaPair = this._createDeltaPair((draft) => {
            draft.memories.push(newMemory);
        });

        if (!deltaPair) {
            console.error("Failed to create delta pair for addMemory");

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

            return null;
        }

        try {
            await this.db.delete(id);
        } catch (error) {
            console.error(`Failed to delete memory ${id} from DB:`, error);
            return null;
        }

        const deltaPair = this._createDeltaPair((draft) => {
            const initialLength = draft.memories.length;
            draft.memories = draft.memories.filter((m: Memory) => m.id !== id);

            return draft.memories.length < initialLength;
        }, true);

        return deltaPair;
    }

    public async generateAndAddMemory(
        turns: StoryTurn[],
        current_turn: number,
        options?: ChatParams["options"]
    ) {
        if (!this.db) throw new Error("Memory bank is not initialized");
        const historyText = turns
            .map((t) => `${t.actor}: ${t.text}`)
            .join("\n");

        const prompt =
            "Summarize the following story segment into a single, concise memory that captures the key events, facts, or character developments. Output only the summarized memory and nothing else. Do not include your thinking process.";

        const response = await this.providerRegistry.chat({
            messages: [
                { role: "system", content: prompt },
                { role: "user", content: historyText },
            ],
            model: this.summarizerModel,
            options: options ?? { temperature: 0.5 /* ... other options */ },
        });

        let cleanOutput = response.message.content
            .replace(/<think>[\s\S]*?<\/think>/gi, "")
            .trim();

        return this.addMemory(cleanOutput, current_turn);
    }

    public async search(
        query: string,
        currentTurn: number,
        limit: number = 10
    ): Promise<Memory[]> {
        if (!this.db) throw new Error("Memory bank is not initialized");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: query,
        });
        const queryVector = embeddings[0];

        const results = await this.db.query(queryVector, {
            k: limit * 2,
            distance: "cosine",
        });

        const retrievedMemoryIds = new Set<number>();
        const memoriesFromDb: Memory[] = [];

        for (const result of results) {
            const mem = this.memories.find((m) => m.id === result.id);
            if (mem) {
                mem.last_accessed_at_turn = currentTurn;
                retrievedMemoryIds.add(mem.id!);
                memoriesFromDb.push({ ...mem });
            } else {
                console.warn(
                    `Memory ID ${result.id} found in DB but not in cache.`
                );

                memoriesFromDb.push({
                    id: result.id,
                    text: result.meta.text,
                    created_at_turn: result.meta.created_at_turn,
                    last_accessed_at_turn: currentTurn,
                });
            }
        }

        const recentMemories = this.memories
            .filter((m) => !retrievedMemoryIds.has(m.id!))
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, 5);

        const finalMemoryMap = new Map<number, Memory>();
        memoriesFromDb.forEach((m) => finalMemoryMap.set(m.id!, m));
        recentMemories.forEach((m) => finalMemoryMap.set(m.id!, m));

        return Array.from(finalMemoryMap.values())
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, limit);
    }

    public async clear() {
        if (!this.db) throw new Error("MemoryBank not initialized.");

        await this.db.clear();

        this.memories = [];
    }

    public getAll() {
        return this.memories;
    }
}

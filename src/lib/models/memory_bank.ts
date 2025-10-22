import { ProviderRegistry } from "../ai/provider";
import { applyPatch, compare, Operation } from "fast-json-patch";
import { deepCopy } from "../util/objects";
import { DeltaPair } from "./world_state";
import { StoryTurn } from "./story_history";
import { EntityDB } from "@babycommando/entity-db";
import { IDBPDatabase, openDB } from "idb";

export interface Memory {
    id: number;
    text: string;
    created_at_turn: number;
    last_accessed_at_turn: number;
    vector?: number[];
}

export class MemoryBank {
    private memories: Memory[] = [];
    private db: EntityDB | null = null;
    private rawDbPromise: Promise<IDBPDatabase> | null = null;
    private vectorPath = "vector";

    private providerRegistry: ProviderRegistry;
    private embeddingModel: string;
    private summarizerModel: string;

    constructor(
        providerRegistry: ProviderRegistry,
        embeddingModel: string,
        summarizerModel: string
    ) {
        this.providerRegistry = providerRegistry;
        this.embeddingModel = embeddingModel;
        this.summarizerModel = summarizerModel;
    }

    public async init() {
        // this.db = await connect(uri);
        this.db = new EntityDB({
            vectorPath: this.vectorPath,
            model: this.embeddingModel,
        });

        this.rawDbPromise = openDB("EntityDB", 1, {
            upgrade(db) {
                if (!db.objectStoreNames.contains("vectors")) {
                    db.createObjectStore("vectors", {
                        keyPath: "id",
                        autoIncrement: true,
                    }); //
                }
            },
        });

        const db = await this.rawDbPromise;
        const allRecords = await db
            .transaction("vectors", "readonly")
            .objectStore("vectors")
            .getAll();

        this.memories = allRecords as Memory[];
    }

    private _createDeltaPair(
        mutator: (draft: { memories: Memory[] }) => boolean | void,
        canFail: boolean = false
    ): DeltaPair | null {
        const oldFullObject = { memories: this.memories };
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
        const newMemories: Memory[] = applyPatch(
            oldMemories,
            delta,
            true,
            false
        ).newDocument;

        const oldIds = new Set(oldMemories.map((m) => m.id));
        const newIds = new Set(newMemories.map((m) => m.id));

        const addedMemories = newMemories.filter((m) => !oldIds.has(m.id));
        const removedIds = [...oldIds].filter((id) => !newIds.has(id));

        await Promise.all([...removedIds.map((id) => this.db!.delete(id))]);
        for (const memToAdd of addedMemories) {
            const { embeddings } = await this.providerRegistry.embed({
                model: this.embeddingModel,
                input: memToAdd.text,
            });
            const { id: oldId, ...data } = memToAdd;
            const dataToInsert = {
                ...data,
                [this.vectorPath]: embeddings[0],
            };
            const newDbId = await this.db.insertManualVectors(dataToInsert); //
            memToAdd.id = newDbId;
        }
        this.memories = newMemories;
    }

    public async addMemory(text: string, currentTurn: number) {
        if (!this.db) throw new Error("Memory bank is not initialized");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: text,
        });

        const vector = embeddings[0];

        // 2. Create memory data (without ID)
        const memoryData = {
            text,
            created_at_turn: currentTurn,
            last_accessed_at_turn: currentTurn,
            [this.vectorPath]: vector, // Add vector under the correct path
        };

        const newId = await this.db.insertManualVectors(memoryData);

        const newMemory: Memory = {
            id: newId,
            text,
            created_at_turn: currentTurn,
            last_accessed_at_turn: currentTurn,
        };

        const deltaPair = this._createDeltaPair((draft) => {
            draft.memories.push(newMemory);
        });

        return { newId, deltaPair: deltaPair! };
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
            "Summarize the following story segment into a single, concise memory that captures the key events, facts, or character developments. Output only the summarized memory and nothing else.";

        const response = await this.providerRegistry.chat({
            messages: [
                {
                    role: "system",
                    content: prompt,
                },
                {
                    role: "user",
                    content: historyText,
                },
            ],
            model: this.summarizerModel,
            options: {
                temperature: 0.5,
                max_completion_tokens: 500,
                top_p: 0.8,
                presence_penalty: 0.2,
                frequency_penalty: 0.5,
            },
        });
        return this.addMemory(response.message.content, current_turn);
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

        const vectorResults = (await this.db.queryManualVectors(queryVector, {
            limit: limit * 2,
        })) as (Memory & { similarity: number })[];

        const retrievedMemoryIds = new Set<number>();
        const updatePromises: Promise<any>[] = [];

        for (const result of vectorResults) {
            const mem = this.memories.find((m) => m.id === result.id);
            if (mem) {
                mem.last_accessed_at_turn = currentTurn;
                retrievedMemoryIds.add(mem.id);

                updatePromises.push(
                    this.db.update(mem.id, {
                        last_accessed_at_turn: currentTurn,
                    })
                );
            }
        }

        const recentMemories = [...this.memories]
            .filter((m) => !retrievedMemoryIds.has(m.id))
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, 5);

        const finalMemoryMap = new Map<number, Memory>();
        vectorResults.forEach((m) => finalMemoryMap.set(m.id, m));
        recentMemories.forEach((m) => finalMemoryMap.set(m.id, m));

        return Array.from(finalMemoryMap.values())
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, limit);
    }
}

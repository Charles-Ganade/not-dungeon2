import { connect, Connection, Table } from "@lancedb/lancedb";
import { ProviderRegistry } from "../ai/provider";
import { applyPatch, compare, Operation } from "fast-json-patch";
import { deepCopy } from "../util/objects";
import { DeltaPair } from "./world_state";
import { StoryTurn } from "./story_history";

export interface Memory {
    id: string;
    text: string;
    created_at_turn: number;
    last_accessed_at_turn: number;
}

export class MemoryBank {
    private memories: Memory[] = [];
    private db: Connection | null = null;
    private table: Table | null = null;

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

    public async init(uri: string) {
        this.db = await connect(uri);

        try {
            this.table = await this.db.openTable("memories");
            const allRecords = await this.table
                .query()
                .select([
                    "id",
                    "text",
                    "created_at_turn",
                    "last_accessed_at_turn",
                ])
                .toArray();

            this.memories = allRecords.map((r: any) => ({
                id: r.id,
                text: r.text,
                created_at_turn: r.created_at_turn,
                last_accessed_at_turn: r.last_accessed_at_turn,
            }));
        } catch (e) {
            const { embeddings } = await this.providerRegistry.embed({
                model: this.embeddingModel,
                input: "test",
            });
            const dim = embeddings[0].length;

            const data = [
                {
                    id: "init_id",
                    text: "initial memory",
                    vector: Array(dim).fill(0.0),
                    created_at_turn: 0,
                    last_accessed_at_turn: 0,
                },
            ];

            this.table = await this.db.createTable("memories", data);
            await this.table.delete("id = 'init_id'");
        }
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
        if (!this.table) throw new Error("MemoryBank not initialized.");

        const oldMemories = deepCopy(this.memories);
        const newMemories: Memory[] = applyPatch(
            oldMemories,
            delta,
            true,
            false
        ).newDocument;

        const oldIds = new Set(oldMemories.map((m) => m.id));
        const newIds = new Set(newMemories.map((m) => m.id));

        const addedIds = [...newIds].filter((id) => !oldIds.has(id));
        const removedIds = [...oldIds].filter((id) => !newIds.has(id));

        await Promise.all([
            // Handle removals from DB
            ...removedIds.map((id) => this.table!.delete(`id = '${id}'`)),
            // Handle additions to DB
            ...addedIds.map(async (id) => {
                const newMem = newMemories.find((m) => m.id === id);
                if (!newMem) return;
                // Re-embed and add the full record
                const { embeddings } = await this.providerRegistry.embed({
                    model: this.embeddingModel,
                    input: newMem.text,
                });
                await this.table!.add([
                    {
                        id: newMem.id,
                        text: newMem.text,
                        vector: embeddings[0],
                        created_at_turn: newMem.created_at_turn,
                        last_accessed_at_turn: newMem.last_accessed_at_turn,
                    },
                ]);
            }),
        ]);
        this.memories = newMemories;
    }

    public async addMemory(text: string, current_turn: number) {
        if (!this.table) throw new Error("Memory bank is not initialized");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: text,
        });

        const vector = embeddings[0];
        const newId = crypto.randomUUID();
        const newMemory: Memory = {
            id: newId,
            text,
            created_at_turn: current_turn,
            last_accessed_at_turn: current_turn,
        };

        await this.table.add([{ ...newMemory, vector }]);

        const deltaPair = this._createDeltaPair((draft) => {
            draft.memories.push(newMemory);
        });

        return { newId, deltaPair: deltaPair! };
    }

    public async generateAndAddMemory(
        turns: StoryTurn[],
        current_turn: number
    ) {
        if (!this.table) throw new Error("Memory bank is not initialized");
        const historyText = turns
            .map((t) => `${t.actor}: ${t.text}`)
            .join("\n");

        const prompt =
            "Summarize the following story segment into a single, concise memory that captures the key events, facts, or character developments. Output only the memory text.";

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
        if (!this.table) throw new Error("Memory bank is not initialized");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: query,
        });
        const queryVector = embeddings[0];

        const results = await this.table
            .search(queryVector)
            .select(["id"])
            .limit(limit)
            .toArray();

        const vectorMemories: Memory[] = [];
        const updatePromises: Promise<any>[] = [];
        const retrievedMemoryIds = new Set<string>();

        for (const r of results as any[]) {
            const mem = this.memories.find((m) => m.id === r.id);

            if (mem) {
                mem.last_accessed_at_turn = currentTurn;
                vectorMemories.push(mem);
                retrievedMemoryIds.add(mem.id);

                updatePromises.push(
                    this.table.update({
                        where: `id = '${mem.id}'`,
                        values: { last_accessed_at_turn: currentTurn },
                    })
                );
            }
        }

        const recentMemories = [...this.memories]
            .filter((m) => !retrievedMemoryIds.has(m.id))
            .sort((a, b) => b.last_accessed_at_turn - a.last_accessed_at_turn)
            .slice(0, 5);

        const finalMemories = [...vectorMemories, ...recentMemories];

        return finalMemories.slice(0, limit);
    }
}

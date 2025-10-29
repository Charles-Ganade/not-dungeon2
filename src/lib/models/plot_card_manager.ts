import { ProviderRegistry } from "../ai/provider";
import { deepCopy } from "../util/objects";
import { LocalVectorDB } from "../vectordb";

export interface PlotCard {
    id?: number;
    category: string;
    name: string;
    content: string;
    triggerKeyword: string;
}

interface PlotCardRecord {
    id: number;
    vector: Float32Array;
    meta: {
        category: string;
        name: string;
        content: string;
        triggerKeyword: string;
    };
    createdAt?: number;
    updatedAt?: number;
}

export class PlotCardManager {
    private plotCards: PlotCard[] = [];
    private db: LocalVectorDB | null = null;

    private providerRegistry: ProviderRegistry;
    private embeddingModel: string;
    private dbName = "PlotCardDB";
    private vectorDimension: number | undefined;

    constructor(
        providerRegistry: ProviderRegistry,
        embeddingModel: string,
        vectorDimension: number
    ) {
        this.providerRegistry = providerRegistry;
        this.embeddingModel = embeddingModel;
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
        if (!this.db) throw new Error("PlotCardManager not initialized.");
        const allRecords = await this.db.export();
        this.plotCards = allRecords.vectors.map((record) => ({
            id: record.id,
            category: record.meta.category,
            name: record.meta.name,
            content: record.meta.content,
            triggerKeyword: record.meta.triggerKeyword,
        }));
    }

    public getAllPlotCards(): PlotCard[] {
        return deepCopy(this.plotCards);
    }

    public async addPlotCard(
        cardData: Omit<PlotCard, "id">
    ): Promise<PlotCard> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: cardData.content,
        });
        const vector = embeddings[0];

        const meta = {
            category: cardData.category,
            name: cardData.name,
            content: cardData.content,
            triggerKeyword: cardData.triggerKeyword,
        };

        const newId = await this.db.insert({
            vector: vector,
            meta: meta,
        });

        const newCard: PlotCard = {
            id: newId,
            ...meta,
        };

        this.plotCards.push(newCard);

        return deepCopy(newCard);
    }

    public async editPlotCard(
        id: number,
        updates: Partial<Omit<PlotCard, "id" | "vector">>
    ): Promise<PlotCard | null> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        const cardIndex = this.plotCards.findIndex((c) => c.id === id);
        if (cardIndex === -1) {
            console.warn(`PlotCard with id ${id} not found in cache.`);
            return null;
        }

        const originalCard = this.plotCards[cardIndex];

        const updatedMeta = {
            category: updates.category ?? originalCard.category,
            name: updates.name ?? originalCard.name,
            content: updates.content ?? originalCard.content,
            triggerKeyword:
                updates.triggerKeyword ?? originalCard.triggerKeyword,
        };

        let vector: number[] | Float32Array;
        let vectorNeedsUpdate = false;

        if (updates.content && updates.content !== originalCard.content) {
            console.log(`Re-embedding PlotCard ${id} due to content change.`);
            const { embeddings } = await this.providerRegistry.embed({
                model: this.embeddingModel,
                input: updatedMeta.content,
            });
            vector = embeddings[0];
            vectorNeedsUpdate = true;
        } else {
            const existingRecord = await this.db.get(id);
            if (existingRecord) {
                vector = new Float32Array(existingRecord.vector);
            } else {
                console.error(
                    `Cannot find existing record with ID ${id} in DB during edit.`
                );
                return null;
            }
        }

        try {
            await this.db.insert({
                id: id,
                vector: vector,
                meta: updatedMeta,
            });
        } catch (error) {
            console.error(`Failed to update PlotCard ${id} in DB:`, error);
            return null;
        }

        const updatedCardInMemory: PlotCard = {
            id: id,
            ...updatedMeta,
        };
        this.plotCards[cardIndex] = updatedCardInMemory;

        return deepCopy(updatedCardInMemory);
    }

    public async removePlotCard(id: number): Promise<boolean> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        const initialLength = this.plotCards.length;
        this.plotCards = this.plotCards.filter((c) => c.id !== id);

        if (this.plotCards.length < initialLength) {
            try {
                await this.db.delete(id);
                return true;
            } catch (e) {
                console.error(`Failed to delete plot card ${id} from DB:`, e);
                return false;
            }
        }
        console.warn(
            `PlotCard with id ${id} not found in cache, nothing to remove.`
        );
        return false;
    }

    public async search(query: string, limit: number = 5): Promise<PlotCard[]> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        const triggeredCards: PlotCard[] = [];
        const lowerCaseQuery = query.toLowerCase();
        this.plotCards.forEach((card) => {
            if (
                card.triggerKeyword &&
                lowerCaseQuery.includes(card.triggerKeyword.toLowerCase())
            ) {
                triggeredCards.push(deepCopy(card));
            }
        });

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: query,
        });
        const queryVector = embeddings[0];

        const vectorResults = await this.db.query(queryVector, {
            k: limit + triggeredCards.length,
            distance: "cosine",
        });

        const finalResultsMap = new Map<
            number,
            { card: PlotCard; score: number }
        >();

        triggeredCards.forEach((card) => {
            finalResultsMap.set(card.id!, { card: card, score: 2.0 });
        });

        vectorResults.forEach((result) => {
            if (!finalResultsMap.has(result.id)) {
                const cardFromResult: PlotCard = {
                    id: result.id,
                    category: result.meta.category,
                    name: result.meta.name,
                    content: result.meta.content,
                    triggerKeyword: result.meta.triggerKeyword,
                };
                finalResultsMap.set(result.id, {
                    card: cardFromResult,
                    score: result.score,
                });
            }
        });

        const sortedResults = Array.from(finalResultsMap.values()).sort(
            (a, b) => {
                return b.score - a.score;
            }
        );

        return sortedResults.slice(0, limit).map((item) => item.card);
    }

    public async clear() {
        if (!this.db) throw new Error("PlotCardManager not initialized.");
        await this.db.clear();
        this.plotCards = [];
    }
}

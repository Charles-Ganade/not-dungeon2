import { ProviderRegistry } from "../ai/provider";
import { deepCopy } from "../util/objects";
// Removed: import { IDBPDatabase, openDB } from "idb";
// Removed: import { EntityDB } from "@babycommando/entity-db";
import { LocalVectorDB } from "../vectordb"; // Import LocalVectorDB

// Interface for the in-memory representation
export interface PlotCard {
    id?: number; // Make id optional for insertion
    category: string;
    name: string;
    content: string;
    triggerKeyword: string;
    // vector is handled by LocalVectorDB
}

// Interface for records stored in LocalVectorDB
interface PlotCardRecord {
    id: number; // ID assigned by LocalVectorDB
    vector: Float32Array; // Stored as ArrayBuffer in DB
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
    // In-memory cache mirroring DB state
    private plotCards: PlotCard[] = [];
    // Changed: Use LocalVectorDB instance
    private db: LocalVectorDB | null = null;
    // Removed: rawDbPromise

    private providerRegistry: ProviderRegistry;
    private embeddingModel: string;
    private dbName = "PlotCardDB"; // Define a specific DB name
    private vectorDimension: number | undefined; // We need to know the dimension
    // Removed: vectorPath
    // Removed: storeName (handled within LocalVectorDB now, defaults to 'vectors')

    constructor(
        providerRegistry: ProviderRegistry,
        embeddingModel: string,
        // Add dimension as a required parameter
        vectorDimension: number
    ) {
        this.providerRegistry = providerRegistry;
        this.embeddingModel = embeddingModel;
        this.vectorDimension = vectorDimension; // Store dimension

        // Initialize LocalVectorDB
        this.db = new LocalVectorDB({
            dbName: this.dbName,
            dimension: this.vectorDimension,
            // Add other options if needed
            normalize: true,
            idField: "id",
        });
    }

    public async init() {
        if (!this.db) throw new Error("LocalVectorDB instance not created.");
        await this.db.init();
        await this.hydrateInMemoryCache();
    }

    // Helper to load all data from DB into the in-memory this.plotCards
    private async hydrateInMemoryCache() {
        if (!this.db) throw new Error("PlotCardManager not initialized.");
        const allRecords = await this.db.export(); // Use export to get all data
        this.plotCards = allRecords.vectors.map((record) => ({
            id: record.id,
            category: record.meta.category,
            name: record.meta.name,
            content: record.meta.content,
            triggerKeyword: record.meta.triggerKeyword,
        }));
    }

    public getAllPlotCards(): PlotCard[] {
        // Return a copy of the in-memory cache
        return deepCopy(this.plotCards);
    }

    public async addPlotCard(
        cardData: Omit<PlotCard, "id">
    ): Promise<PlotCard> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        // 1. Embed content
        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: cardData.content,
        });
        const vector = embeddings[0];

        // 2. Prepare metadata
        const meta = {
            category: cardData.category,
            name: cardData.name,
            content: cardData.content,
            triggerKeyword: cardData.triggerKeyword,
        };

        // 3. Insert into LocalVectorDB
        const newId = await this.db.insert({
            vector: vector,
            meta: meta,
        });

        // 4. Create in-memory representation
        const newCard: PlotCard = {
            id: newId,
            ...meta,
        };

        // 5. Update in-memory cache
        this.plotCards.push(newCard);

        return deepCopy(newCard);
    }

    public async editPlotCard(
        id: number,
        updates: Partial<Omit<PlotCard, "id" | "vector">> // Exclude vector from partial updates
    ): Promise<PlotCard | null> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        const cardIndex = this.plotCards.findIndex((c) => c.id === id);
        if (cardIndex === -1) {
            console.warn(`PlotCard with id ${id} not found in cache.`);
            return null;
        }

        const originalCard = this.plotCards[cardIndex];

        // Create the potential new state (in-memory and meta for DB)
        const updatedMeta = {
            category: updates.category ?? originalCard.category,
            name: updates.name ?? originalCard.name,
            content: updates.content ?? originalCard.content,
            triggerKeyword:
                updates.triggerKeyword ?? originalCard.triggerKeyword,
        };

        let vector: number[] | Float32Array;
        let vectorNeedsUpdate = false;

        // Re-embed only if content changed
        if (updates.content && updates.content !== originalCard.content) {
            console.log(`Re-embedding PlotCard ${id} due to content change.`);
            const { embeddings } = await this.providerRegistry.embed({
                model: this.embeddingModel,
                input: updatedMeta.content,
            });
            vector = embeddings[0];
            vectorNeedsUpdate = true;
        } else {
            // Need the existing vector if content didn't change
            // LocalVectorDB's insert acts as upsert but needs the vector. Fetch it.
            const existingRecord = await this.db.get(id);
            if (existingRecord) {
                // The vector in the DB record is an ArrayBuffer, convert it
                vector = new Float32Array(existingRecord.vector);
            } else {
                console.error(
                    `Cannot find existing record with ID ${id} in DB during edit.`
                );
                return null;
            }
        }

        // Use insert for upsert functionality in LocalVectorDB
        try {
            await this.db.insert({
                id: id, // Provide ID to update existing
                vector: vector,
                meta: updatedMeta,
            });
        } catch (error) {
            console.error(`Failed to update PlotCard ${id} in DB:`, error);
            return null;
        }

        // Update in-memory cache
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
        // Filter in-memory cache first
        this.plotCards = this.plotCards.filter((c) => c.id !== id);

        // If something was removed from cache, remove from DB
        if (this.plotCards.length < initialLength) {
            try {
                await this.db.delete(id);
                return true;
            } catch (e) {
                console.error(`Failed to delete plot card ${id} from DB:`, e);
                // Optionally re-add to in-memory cache if DB delete fails?
                // For simplicity, we assume DB delete succeeds if cache removal did.
                return false;
            }
        }
        console.warn(
            `PlotCard with id ${id} not found in cache, nothing to remove.`
        );
        return false; // Not found in memory
    }

    public async search(query: string, limit: number = 5): Promise<PlotCard[]> {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        // 1. Find keyword-triggered cards from IN-MEMORY cache
        const triggeredCards: PlotCard[] = [];
        const lowerCaseQuery = query.toLowerCase();
        this.plotCards.forEach((card) => {
            if (
                card.triggerKeyword &&
                lowerCaseQuery.includes(card.triggerKeyword.toLowerCase())
            ) {
                // Add a copy to avoid mutation issues
                triggeredCards.push(deepCopy(card));
            }
        });
        const triggeredCardIds = new Set(triggeredCards.map((c) => c.id));

        // 2. Embed the query text
        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: query,
        });
        const queryVector = embeddings[0];

        // 3. Query LocalVectorDB for semantically similar cards
        // Fetch more than limit initially to allow combining/ranking
        const vectorResults = await this.db.query(queryVector, {
            k: limit + triggeredCards.length, // Fetch enough potential candidates
            distance: "cosine", // Assuming cosine
        });

        // 4. Combine and Rank
        const finalResultsMap = new Map<
            number,
            { card: PlotCard; score: number }
        >();

        // Add triggered cards first with a high score boost (e.g., score 2)
        triggeredCards.forEach((card) => {
            finalResultsMap.set(card.id!, { card: card, score: 2.0 }); // Use high score for triggered
        });

        // Add vector results, skipping duplicates already added by keyword trigger
        vectorResults.forEach((result) => {
            if (!finalResultsMap.has(result.id)) {
                // Reconstruct PlotCard from DB result meta
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

        // 5. Sort combined results: Triggered first, then by similarity score
        const sortedResults = Array.from(finalResultsMap.values()).sort(
            (a, b) => {
                // Sort primarily by score (descending), handles triggered boost
                return b.score - a.score;
            }
        );

        // 6. Return top 'limit' results
        return sortedResults.slice(0, limit).map((item) => item.card);
    }

    // Updated clear method
    public async clear() {
        if (!this.db) throw new Error("PlotCardManager not initialized.");

        // Clear the database
        await this.db.clear();

        // Also clear the in-memory cache
        this.plotCards = [];
    }

    // Removed cosineSimilarity helper, as it's now handled by LocalVectorDB
}

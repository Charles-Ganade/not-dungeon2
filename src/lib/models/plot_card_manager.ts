import { ProviderRegistry } from "../ai/provider";
import { deepCopy } from "../util/objects";
import { IDBPDatabase, openDB } from "idb";
import { EntityDB } from "@babycommando/entity-db";

export interface PlotCard {
    id: number; // Changed from string
    category: string;
    name: string;
    content: string;
    triggerKeyword: string;
    vector?: number[]; // Optional in memory
}

export class PlotCardManager {
    private plotCards: PlotCard[] = []; // In-memory cache
    private db: EntityDB | null = null;
    private rawDbPromise: Promise<IDBPDatabase> | null = null;

    private providerRegistry: ProviderRegistry;
    private embeddingModel: string;
    private vectorPath = "vector";

    constructor(providerRegistry: ProviderRegistry, embeddingModel: string) {
        this.providerRegistry = providerRegistry;
        this.embeddingModel = embeddingModel;
    }

    public async init() {
        this.db = new EntityDB({
            vectorPath: this.vectorPath,
            model: this.embeddingModel,
        });

        const storeName = "plot_cards";

        // Initialize raw DB access specifically for plot cards store
        this.rawDbPromise = openDB("PlotCardDB", 1, {
            // Using a separate DB name for clarity
            upgrade(db) {
                if (!db.objectStoreNames.contains(storeName)) {
                    db.createObjectStore(storeName, {
                        keyPath: "id",
                        autoIncrement: true,
                    });
                    // If needed, add indexes for category, name, etc.
                    // store.createIndex('category', 'category');
                }
            },
        });

        // Hydrate the in-memory list
        const db = await this.rawDbPromise;
        const allRecords = await db
            .transaction(storeName, "readonly")
            .objectStore(storeName)
            .getAll();
        this.plotCards = allRecords as PlotCard[];
    }

    public getAllPlotCards(): PlotCard[] {
        return deepCopy(this.plotCards);
    }

    public async addPlotCard(
        cardData: Omit<PlotCard, "id">
    ): Promise<PlotCard> {
        if (!this.rawDbPromise)
            throw new Error("PlotCardManager not initialized.");

        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: cardData.content,
        });
        const vector = embeddings[0];

        const dataToInsert = {
            ...cardData,
            [this.vectorPath]: embeddings[0],
        };

        const db = await this.rawDbPromise;
        const newId = await db
            .transaction("plot_cards", "readwrite")
            .objectStore("plot_cards")
            .add(dataToInsert);

        const newCard: PlotCard = {
            ...cardData,
            id: newId as number,
        };

        // Add to in-memory list
        this.plotCards.push(newCard);

        return deepCopy(newCard);
    }

    public async editPlotCard(
        id: number,
        updates: Partial<Omit<PlotCard, "id">>
    ): Promise<PlotCard | null> {
        if (!this.rawDbPromise)
            throw new Error("PlotCardManager not initialized.");

        const cardIndex = this.plotCards.findIndex((c) => c.id === id);
        if (cardIndex === -1) return null;

        const originalCard = this.plotCards[cardIndex];
        const updatedCardData = { ...originalCard, ...updates };

        let vector: number[];
        // Re-embed only if content changed
        if (updates.content && updates.content !== originalCard.content) {
            const { embeddings } = await this.providerRegistry.embed({
                //
                model: this.embeddingModel,
                input: updatedCardData.content,
            });
            vector = embeddings[0];
        } else {
            // Need the existing vector if content didn't change
            const db = await this.rawDbPromise;
            const existingRecord = await db
                .transaction("plot_cards", "readonly")
                .objectStore("plot_cards")
                .get(id);
            if (existingRecord) {
                vector = existingRecord[this.vectorPath];
            } else {
                console.error(
                    `Cannot find existing record with ID ${id} to get vector during edit.`
                );
                return null; // Or handle error appropriately
            }
        }

        const recordToPut = {
            ...updatedCardData,
            id: id, // Ensure ID is included for put
            [this.vectorPath]: vector,
        };

        // Update DB using put (overwrites record with the same key)
        const db = await this.rawDbPromise;
        await db
            .transaction("plot_cards", "readwrite")
            .objectStore("plot_cards")
            .put(recordToPut);

        // Update in-memory list (remove vector property for consistency if desired)
        const { vector: _, ...cardForMemory } = recordToPut;
        this.plotCards[cardIndex] = cardForMemory;

        return deepCopy(cardForMemory);
    }

    public async removePlotCard(id: number): Promise<boolean> {
        if (!this.rawDbPromise)
            throw new Error("PlotCardManager not initialized.");

        const initialLength = this.plotCards.length;
        this.plotCards = this.plotCards.filter((c) => c.id !== id);

        if (this.plotCards.length < initialLength) {
            // Remove from DB
            const db = await this.rawDbPromise;
            try {
                await db
                    .transaction("plot_cards", "readwrite")
                    .objectStore("plot_cards")
                    .delete(id);
                return true;
            } catch (e) {
                console.error(`Failed to delete plot card ${id} from DB:`, e);
                // Rollback in-memory change?
                return false;
            }
        }
        return false; // Not found in memory
    }

    public async search(query: string, limit: number = 5): Promise<PlotCard[]> {
        if (!this.rawDbPromise || !this.db)
            throw new Error("PlotCardManager not initialized.");

        // 1. Find keyword-triggered cards from in-memory list
        const triggeredCardIds = new Set<number>();
        const lowerCaseQuery = query.toLowerCase();
        this.plotCards.forEach((card) => {
            if (
                card.triggerKeyword &&
                lowerCaseQuery.includes(card.triggerKeyword.toLowerCase())
            ) {
                triggeredCardIds.add(card.id);
            }
        });

        // 2. Embed the query text
        const { embeddings } = await this.providerRegistry.embed({
            model: this.embeddingModel,
            input: query,
        });
        const queryVector = embeddings[0];

        // 3. Get *all* records from the DB to calculate similarity manually
        //    (entity-db's query doesn't work across different object stores easily)
        const db = await this.rawDbPromise;
        const allRecords = await db
            .transaction("plot_cards", "readonly")
            .objectStore("plot_cards")
            .getAll();

        // 4. Calculate similarities manually
        const similarities = allRecords.map((entry: any) => {
            const similarity = this.cosineSimilarity(
                queryVector,
                entry[this.vectorPath]
            );
            // Return only necessary data + similarity
            return {
                id: entry.id,
                category: entry.category,
                name: entry.name,
                content: entry.content,
                triggerKeyword: entry.triggerKeyword,
                similarity,
            };
        });

        // 5. Combine and Rank
        const finalResultsMap = new Map<
            number,
            { card: PlotCard; similarity: number | null }
        >();

        // Add triggered cards first, finding their similarity score
        triggeredCardIds.forEach((id) => {
            const card = this.plotCards.find((c) => c.id === id); // Use in-memory data
            if (card) {
                const vectorMatch = similarities.find((r) => r.id === id);
                finalResultsMap.set(id, {
                    card,
                    similarity: vectorMatch?.similarity ?? null,
                });
            }
        });

        // Add vector results, skipping duplicates
        similarities.forEach((r) => {
            if (!finalResultsMap.has(r.id)) {
                // Construct PlotCard from similarity result (which has all needed fields)
                const { similarity: _, ...cardData } = r;
                finalResultsMap.set(r.id, {
                    card: cardData as PlotCard,
                    similarity: r.similarity,
                });
            }
        });

        // 6. Sort and Limit
        const sortedResults = Array.from(finalResultsMap.values()).sort(
            (a, b) => {
                const aIsTriggered = triggeredCardIds.has(a.card.id);
                const bIsTriggered = triggeredCardIds.has(b.card.id);

                if (aIsTriggered && !bIsTriggered) return -1; // a comes first
                if (!aIsTriggered && bIsTriggered) return 1; // b comes first

                // If both triggered or both not, sort by similarity (higher is better)
                const simA = a.similarity ?? -1; // Treat null as lowest similarity
                const simB = b.similarity ?? -1;
                return simB - simA; // Descending order
            }
        );

        return sortedResults.slice(0, limit).map((item) => item.card);
    }

    // Helper for cosine similarity (copied from entity-db source)
    private cosineSimilarity(vecA: number[], vecB: number[]): number {
        const dotProduct = vecA.reduce(
            (sum, val, index) => sum + val * vecB[index],
            0
        );
        const magnitudeA = Math.sqrt(
            vecA.reduce((sum, val) => sum + val * val, 0)
        );
        const magnitudeB = Math.sqrt(
            vecB.reduce((sum, val) => sum + val * val, 0)
        );
        if (magnitudeA === 0 || magnitudeB === 0) return 0; // Avoid division by zero
        return dotProduct / (magnitudeA * magnitudeB);
    } //
}

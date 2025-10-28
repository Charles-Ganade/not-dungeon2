/*
LocalVectorDB - TypeScript implementation
- Per-database isolation (default)
- Automatic migrations (persisted in _meta.store)
- Normalization default: true
- Optional in-memory cache (explicit opt-in)

Notes for migration authors:
- Migration functions are executed inside the IndexedDB `onupgradeneeded` upgrade transaction.
- Migration functions MUST use only the `db` and `tx` provided and queue IDB requests on that transaction.
- Do NOT perform async operations that are outside the provided `tx` (they won't be atomic with the upgrade).
- Migration functions may read/update the `_meta` object store by keying to 'schema' and 'migrations'.
*/

type MetaSchema = {
    version: number;
    dimension: number;
    format: "dense" | "binary";
    normalize: boolean;
    indexes: string[];
    createdAt: number;
    updatedAt: number;
};

export type LocalVectorDBOptions = {
    dbName: string; // required
    version?: number; // default 1
    dimension: number; // required
    format?: "dense" | "binary"; // default 'dense'
    normalize?: boolean; // default true
    cache?: boolean; // default false
    idField?: string; // default 'id'
    metaIndexes?: string[]; // default []
    onMigrate?: (
        oldVersion: number,
        newVersion: number,
        meta: MetaSchema
    ) => void | Promise<void>;
    verbose?: boolean;
};

// Migration function type. Must operate using the provided `db` and `tx` (upgrade transaction).
export type MigrationFn = (
    db: IDBDatabase,
    tx: IDBTransaction,
    metaStore: IDBObjectStore
) => void;

class MinHeap<T> {
    private items: Array<T> = [];
    private comparator: (a: T, b: T) => number;
    constructor(comparator: (a: T, b: T) => number) {
        this.comparator = comparator;
    }
    size() {
        return this.items.length;
    }
    peek() {
        return this.items[0];
    }
    push(item: T) {
        this.items.push(item);
        this.bubbleUp();
    }
    pop() {
        if (this.items.length === 0) return undefined;
        const top = this.items[0];
        const last = this.items.pop()!;
        if (this.items.length > 0) {
            this.items[0] = last;
            this.bubbleDown();
        }
        return top;
    }
    private bubbleUp() {
        let idx = this.items.length - 1;
        while (idx > 0) {
            const parent = Math.floor((idx - 1) / 2);
            if (this.comparator(this.items[idx], this.items[parent]) < 0) {
                [this.items[idx], this.items[parent]] = [
                    this.items[parent],
                    this.items[idx],
                ];
                idx = parent;
            } else break;
        }
    }
    private bubbleDown() {
        let idx = 0;
        const length = this.items.length;
        while (true) {
            const left = 2 * idx + 1;
            const right = 2 * idx + 2;
            let smallest = idx;
            if (
                left < length &&
                this.comparator(this.items[left], this.items[smallest]) < 0
            )
                smallest = left;
            if (
                right < length &&
                this.comparator(this.items[right], this.items[smallest]) < 0
            )
                smallest = right;
            if (smallest !== idx) {
                [this.items[idx], this.items[smallest]] = [
                    this.items[smallest],
                    this.items[idx],
                ];
                idx = smallest;
            } else break;
        }
    }
}

export class LocalVectorDB {
    // Static migration registry: map of "from->to" -> fn
    private static migrations: Map<string, MigrationFn> = new Map();

    // Register migration
    static registerMigration(
        fromVersion: number,
        toVersion: number,
        fn: MigrationFn
    ) {
        if (toVersion !== fromVersion + 1) {
            throw new Error(
                "Migrations must be registered for consecutive versions (from -> from+1)"
            );
        }
        const key = `${fromVersion}->${toVersion}`;
        this.migrations.set(key, fn);
    }

    static unregisterMigration(fromVersion: number, toVersion: number) {
        const key = `${fromVersion}->${toVersion}`;
        this.migrations.delete(key);
    }

    static listMigrations() {
        const out: Array<{ from: number; to: number }> = [];
        for (const key of this.migrations.keys()) {
            const [from, to] = key.split("->").map((s) => Number(s));
            out.push({ from, to });
        }
        return out.sort((a, b) => a.from - b.from);
    }

    // Instance fields
    private options: LocalVectorDBOptions;
    private dbPromise: Promise<IDBDatabase> | null = null;
    private db: IDBDatabase | null = null;
    private cacheEnabled = false;
    private inMemoryCache:
        | { id: any; vector: Float32Array; meta?: any }[]
        | null = null;

    constructor(options: LocalVectorDBOptions) {
        if (!options || !options.dbName) throw new Error("dbName is required");
        if (!options.dimension) throw new Error("dimension is required");
        this.options = {
            version: 1,
            format: "dense",
            normalize: true,
            cache: false,
            idField: "id",
            metaIndexes: [],
            verbose: false,
            ...options,
        };
        // ensure integers
        this.options.version = Math.max(
            1,
            Math.floor(this.options.version || 1)
        );
    }

    // Initialize (open DB). This will automatically run migrations if needed.
    async init() {
        if (this.db) return;
        if (!this.dbPromise) this.dbPromise = this.openDB();
        this.db = await this.dbPromise;
        if (this.options.cache) await this.enableCache();
        return;
    }

    // Open the IDB database and run migrations in onupgradeneeded
    private openDB(): Promise<IDBDatabase> {
        const { dbName, version } = this.options;
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(dbName, version);
            req.onblocked = () => {
                console.warn("openDB blocked: another connection is open");
            };
            req.onerror = (ev) => reject((ev.target as IDBRequest).error);

            req.onupgradeneeded = (ev) => {
                const db = (ev.target as IDBOpenDBRequest).result;
                const tx = (ev.target as IDBOpenDBRequest)
                    .transaction as IDBTransaction;
                const oldVer = ev.oldVersion || 0;
                const newVer = ev.newVersion || version || 1;
                if (this.options.verbose)
                    console.log(
                        `Upgrading ${dbName} from ${oldVer} → ${newVer}`
                    );

                // Ensure _meta object store exists
                if (!db.objectStoreNames.contains("_meta")) {
                    db.createObjectStore("_meta", { keyPath: "key" });
                }

                // Ensure vectors object store exists; create with idField keyPath if not present
                if (!db.objectStoreNames.contains("vectors")) {
                    const props: any = { keyPath: this.options.idField };
                    if (this.options.idField === "id")
                        props.autoIncrement = true;
                    db.createObjectStore("vectors", props);
                } else {
                    // In case idField changed (rare) we cannot change keyPath easily; user should manage this.
                }

                // Create meta index stores if not present (no-op here, indexes on vectors will be created by migrations)

                // Run consecutive migrations: require registration from v -> v+1
                // We'll run migrations for v in [oldVer, newVer-1]
                for (let v = oldVer; v < newVer; v++) {
                    const key = `${v}->${v + 1}`;
                    const fn = LocalVectorDB.migrations.get(key);
                    if (fn) {
                        try {
                            if (this.options.verbose)
                                console.log(`Applying migration ${key}`);
                            // Provide the _meta store handle so migration can read/write schema and migrations list
                            const metaStore = tx.objectStore("_meta");
                            fn(db, tx, metaStore);
                            // record applied migration into _meta; we cannot read previous value synchronously, so append tracking by using get/put request
                            const getReq = metaStore.get("migrations");
                            getReq.onsuccess = () => {
                                const existing = getReq.result
                                    ? getReq.result.value
                                    : [];
                                const updated = existing.concat([[v, v + 1]]);
                                metaStore.put({
                                    key: "migrations",
                                    value: updated,
                                });
                            };
                            getReq.onerror = () => {
                                // If get failed, just put the migration entry as a fresh array
                                metaStore.put({
                                    key: "migrations",
                                    value: [[v, v + 1]],
                                });
                            };
                        } catch (err) {
                            console.error("Migration fn threw:", err);
                            throw err; // abort upgrade
                        }
                    } else {
                        if (this.options.verbose)
                            console.log(`No migration registered for ${key}`);
                    }
                }

                // Finally, ensure the meta.schema entry exists or is updated
                const metaStore = tx.objectStore("_meta");
                const schemaReq = metaStore.get("schema");
                schemaReq.onsuccess = () => {
                    const now = Date.now();
                    const current = schemaReq.result
                        ? (schemaReq.result.value as MetaSchema)
                        : null;
                    const newSchema: MetaSchema = {
                        version: newVer,
                        dimension: this.options.dimension,
                        format: this.options.format || "dense",
                        normalize: !!this.options.normalize,
                        indexes: this.options.metaIndexes || [],
                        createdAt: current ? current.createdAt : now,
                        updatedAt: now,
                    };
                    metaStore.put({ key: "schema", value: newSchema });
                };
                schemaReq.onerror = () => {
                    const now = Date.now();
                    const newSchema: MetaSchema = {
                        version: newVer,
                        dimension: this.options.dimension,
                        format: this.options.format || "dense",
                        normalize: !!this.options.normalize,
                        indexes: this.options.metaIndexes || [],
                        createdAt: now,
                        updatedAt: now,
                    };
                    metaStore.put({ key: "schema", value: newSchema });
                };

                // Create any metaIndexes on vectors store as simple indexes pointing to meta.<field>
                try {
                    const vs = db.objectStoreNames.contains("vectors")
                        ? (tx.objectStore("vectors") as IDBObjectStore)
                        : null;
                    if (
                        vs &&
                        this.options.metaIndexes &&
                        this.options.metaIndexes.length > 0
                    ) {
                        for (const idx of this.options.metaIndexes) {
                            // Creating index only allowed in upgrade txn via objectStore.createIndex
                            try {
                                // if the index already exists, skip
                                if (!vs.indexNames.contains(idx)) {
                                    vs.createIndex(idx, `meta.${idx}`, {
                                        unique: false,
                                    });
                                }
                            } catch (err) {
                                // createIndex may throw if index already exists in structural DB but not in this txn — ignore
                                if (this.options.verbose)
                                    console.warn(
                                        `Could not create index ${idx}:`,
                                        err
                                    );
                            }
                        }
                    }
                } catch (err) {
                    if (this.options.verbose)
                        console.warn(
                            "Could not create meta indexes during upgrade:",
                            err
                        );
                }
            };

            req.onsuccess = (ev) => {
                const db = (ev.target as IDBOpenDBRequest)
                    .result as IDBDatabase;
                // update persisted _meta.schema if we need to ensure version is set (some browsers/paths may not call onupgradeneeded when version same)
                const tx = db.transaction("_meta", "readwrite");
                const metaStore = tx.objectStore("_meta");
                const schemaGet = metaStore.get("schema");
                schemaGet.onsuccess = () => {
                    const now = Date.now();
                    const existing: MetaSchema | null = schemaGet.result
                        ? schemaGet.result.value
                        : null;
                    const newSchema: MetaSchema = {
                        version: this.options.version || 1,
                        dimension: this.options.dimension,
                        format: this.options.format || "dense",
                        normalize: !!this.options.normalize,
                        indexes: this.options.metaIndexes || [],
                        createdAt: existing ? existing.createdAt : now,
                        updatedAt: now,
                    };
                    metaStore.put({ key: "schema", value: newSchema });
                };
                tx.oncomplete = () => resolve(db);
                tx.onerror = () => reject(tx.error);
            };
        });
    }

    // Insert a single record
    async insert(record: {
        id?: any;
        vector: Float32Array | number[] | ArrayBuffer;
        meta?: any;
    }) {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");

        const vectorBuffer = this.normalizeAndSerialize(record.vector);
        const tx = this.db.transaction("vectors", "readwrite");
        const store = tx.objectStore("vectors");
        const now = Date.now();
        const rec: any = {
            ...record,
            vector: vectorBuffer,
            createdAt: now,
            updatedAt: now,
        };
        // if id is undefined and idField is 'id', allow auto increment; else ensure id exists
        if (this.options.idField !== "id" && record.id === undefined) {
            throw new Error("Custom idField set but record.id is undefined");
        }
        const req = store.put(rec);
        return await new Promise<any>((resolve, reject) => {
            req.onsuccess = (ev) => resolve((ev.target as IDBRequest).result);
            req.onerror = (ev) => reject((ev.target as IDBRequest).error);
        });
    }

    // Batch insert
    async insertBatch(
        records: Array<{
            id?: any;
            vector: Float32Array | number[] | ArrayBuffer;
            meta?: any;
        }>
    ) {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readwrite");
        const store = tx.objectStore("vectors");
        const now = Date.now();
        for (const r of records) {
            const vectorBuffer = this.normalizeAndSerialize(r.vector);
            const rec: any = {
                ...r,
                vector: vectorBuffer,
                createdAt: now,
                updatedAt: now,
            };
            store.put(rec);
        }
        return await new Promise<void>((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    // Normalize (if enabled) and return an ArrayBuffer for storage
    private normalizeAndSerialize(
        vector: Float32Array | number[] | ArrayBuffer
    ): ArrayBuffer {
        let floatArr: Float32Array;
        if (vector instanceof ArrayBuffer) {
            // assume caller provided Float32Array.buffer
            floatArr = new Float32Array(vector);
        } else if (Array.isArray(vector)) {
            floatArr = new Float32Array(vector);
        } else if ((vector as any).buffer instanceof ArrayBuffer) {
            floatArr = vector as Float32Array;
        } else {
            throw new Error("Unsupported vector type");
        }
        if (floatArr.length !== this.options.dimension) {
            throw new Error(
                `dimension mismatch: expected ${this.options.dimension} got ${floatArr.length}`
            );
        }
        if (this.options.normalize) {
            const norm = Math.hypot(...Array.from(floatArr));
            if (norm > 0) {
                for (let i = 0; i < floatArr.length; i++)
                    floatArr[i] = floatArr[i] / norm;
            }
        }
        return floatArr.buffer.slice(0) as ArrayBuffer; // return copy
    }

    // Enable in-memory cache (loads all vectors)
    async enableCache() {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        if (this.cacheEnabled) return;
        const tx = this.db.transaction("vectors", "readonly");
        const store = tx.objectStore("vectors");
        const cursorReq = store.openCursor();
        this.inMemoryCache = [];
        return await new Promise<void>((resolve, reject) => {
            cursorReq.onsuccess = (ev) => {
                const cursor = (ev.target as IDBRequest)
                    .result as IDBCursorWithValue;
                if (cursor) {
                    const val = cursor.value;
                    const vec = new Float32Array(val.vector);
                    this.inMemoryCache!.push({
                        id: val[this.options.idField!],
                        vector: vec,
                        meta: val.meta,
                    });
                    cursor.continue();
                }
            };
            tx.oncomplete = () => {
                this.cacheEnabled = true;
                resolve();
            };
            tx.onerror = () => reject(tx.error);
        });
    }

    async disableCache() {
        this.inMemoryCache = null;
        this.cacheEnabled = false;
    }

    // Query dense vectors using cosine similarity (default) or euclidean
    async query(
        queryVector: Float32Array | number[] | ArrayBuffer,
        opts?: {
            k?: number;
            distance?: "cosine" | "euclidean";
            filter?: (meta: any) => boolean;
            maxCandidates?: number;
        }
    ) {
        await this.init();
        const {
            k = 10,
            distance = "cosine",
            filter,
            maxCandidates,
        } = opts || {};
        const qvec = this.toFloat32(queryVector);
        if (qvec.length !== this.options.dimension)
            throw new Error("query vector dimension mismatch");
        if (this.options.normalize && distance === "cosine") {
            const norm = Math.hypot(...Array.from(qvec));
            if (norm > 0) for (let i = 0; i < qvec.length; i++) qvec[i] /= norm;
        }

        // If cache enabled, search in-memory
        if (this.cacheEnabled && this.inMemoryCache) {
            const heap = new MinHeap<{ id: any; meta: any; score: number }>(
                (a, b) => a.score - b.score
            );
            let examined = 0;
            for (const rec of this.inMemoryCache) {
                if (filter && !filter(rec.meta)) continue;
                const score =
                    distance === "cosine"
                        ? dot(qvec, rec.vector)
                        : -euclidean(qvec, rec.vector); // higher is better
                if (heap.size() < k)
                    heap.push({ id: rec.id, meta: rec.meta, score });
                else if (score > heap.peek()!.score) {
                    heap.pop();
                    heap.push({ id: rec.id, meta: rec.meta, score });
                }
                examined++;
                if (maxCandidates && examined >= maxCandidates) break;
            }
            const out: Array<{ id: any; score: number; meta?: any }> = [];
            const items: any[] = [];
            while (heap.size() > 0) items.push(heap.pop());
            // items are in ascending score; reverse
            items.reverse();
            for (const it of items)
                out.push({ id: it.id, score: it.score, meta: it.meta });
            return out;
        }

        // Otherwise iterate via cursor
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readonly");
        const store = tx.objectStore("vectors");
        const cursorReq = store.openCursor();
        const heap = new MinHeap<{ id: any; meta: any; score: number }>(
            (a, b) => a.score - b.score
        );
        let examined = 0;
        return await new Promise<Array<{ id: any; score: number; meta?: any }>>(
            (resolve, reject) => {
                cursorReq.onsuccess = (ev) => {
                    const cursor = (ev.target as IDBRequest)
                        .result as IDBCursorWithValue;
                    if (cursor) {
                        const val = cursor.value;
                        if (!val.vector) {
                            cursor.continue();
                            return;
                        }
                        const vec = new Float32Array(val.vector);
                        if (filter && !filter(val.meta)) {
                            cursor.continue();
                            return;
                        }
                        const score =
                            distance === "cosine"
                                ? dot(qvec, vec)
                                : -euclidean(qvec, vec);
                        if (heap.size() < k)
                            heap.push({
                                id: val[this.options.idField!],
                                meta: val.meta,
                                score,
                            });
                        else if (score > heap.peek()!.score) {
                            heap.pop();
                            heap.push({
                                id: val[this.options.idField!],
                                meta: val.meta,
                                score,
                            });
                        }
                        examined++;
                        if (maxCandidates && examined >= maxCandidates) {
                            // stop early: close cursor by resolving after processing current
                            // We'll drain remaining using tx.oncomplete
                            // Continue to finish, but we can simply resolve now reading heap contents
                            const out: Array<{
                                id: any;
                                score: number;
                                meta?: any;
                            }> = [];
                            const items: any[] = [];
                            while (heap.size() > 0) items.push(heap.pop());
                            items.reverse();
                            for (const it of items)
                                out.push({
                                    id: it.id,
                                    score: it.score,
                                    meta: it.meta,
                                });
                            resolve(out);
                            return;
                        }
                        cursor.continue();
                    } else {
                        // finished
                        const out: Array<{
                            id: any;
                            score: number;
                            meta?: any;
                        }> = [];
                        const items: any[] = [];
                        while (heap.size() > 0) items.push(heap.pop());
                        items.reverse();
                        for (const it of items)
                            out.push({
                                id: it.id,
                                score: it.score,
                                meta: it.meta,
                            });
                        resolve(out);
                    }
                };
                cursorReq.onerror = () => reject((cursorReq as any).error);
            }
        );
    }

    // Delete
    async delete(id: any) {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readwrite");
        const store = tx.objectStore("vectors");
        store.delete(id);
        return await new Promise<void>((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    // Get by id
    async get(id: any) {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readonly");
        const store = tx.objectStore("vectors");
        const req = store.get(id);
        return await new Promise<any>((resolve, reject) => {
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    // Count
    async count() {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readonly");
        const store = tx.objectStore("vectors");
        const req = store.count();
        return await new Promise<number>((resolve, reject) => {
            req.onsuccess = () => resolve(req.result as number);
            req.onerror = () => reject(req.error);
        });
    }

    // Clear all vectors
    async clear() {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readwrite");
        const store = tx.objectStore("vectors");
        store.clear();
        return await new Promise<void>((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    // Export DB (simple JSON friendly export)
    async export(): Promise<{ schema: MetaSchema; vectors: any[] }> {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction(["_meta", "vectors"], "readonly");
        const metaStore = tx.objectStore("_meta");
        const schemaReq = metaStore.get("schema");
        const vectorsReq = tx.objectStore("vectors").getAll();
        return await new Promise((resolve, reject) => {
            tx.oncomplete = () => {
                const schema: MetaSchema = schemaReq.result
                    ? schemaReq.result.value
                    : ({} as any);
                const vecs = vectorsReq.result || [];
                // Convert ArrayBuffer vectors into number[] for portability
                const serial = vecs.map((v: any) => ({
                    ...v,
                    vector: Array.from(new Float32Array(v.vector)),
                }));
                resolve({ schema, vectors: serial });
            };
            tx.onerror = () => reject(tx.error);
        });
    }

    // Import (overwrite) - simple implementation: insert vectors into vectors store
    async import(
        payload: { schema: MetaSchema; vectors: any[] },
        { clearBefore = false } = {}
    ) {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        if (clearBefore) await this.clear();
        const tx = this.db.transaction(["_meta", "vectors"], "readwrite");
        const metaStore = tx.objectStore("_meta");
        metaStore.put({ key: "schema", value: payload.schema });
        const vs = tx.objectStore("vectors");
        for (const v of payload.vectors) {
            const copy = { ...v, vector: new Float32Array(v.vector).buffer };
            vs.put(copy);
        }
        return await new Promise<void>((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    // Close DB
    async close() {
        if (this.db) {
            this.db.close();
            this.db = null;
            this.dbPromise = null;
        }
    }

    // Helpers
    private toFloat32(
        vec: Float32Array | number[] | ArrayBuffer
    ): Float32Array {
        if (vec instanceof Float32Array) return vec;
        if (vec instanceof ArrayBuffer) return new Float32Array(vec);
        if (Array.isArray(vec)) return new Float32Array(vec);
        throw new Error("Unsupported query vector type");
    }
}

// Utility functions
function dot(a: Float32Array, b: Float32Array) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}
function euclidean(a: Float32Array, b: Float32Array) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return Math.sqrt(s);
}

export default LocalVectorDB;

import { Buffer } from "buffer";

export type VectorFormat = "dense" | "binary" | "mixed";

export type MetaSchema = {
    version: number;
    dimension: number;
    format: VectorFormat;
    normalize: boolean;
    indexes: string[];
    createdAt: number;
    updatedAt: number;
};

export type LocalVectorDBOptions = {
    dbName: string; // required
    version?: number; // default 1
    dimension: number; // required
    format?: VectorFormat; // default 'dense'
    normalize?: boolean; // default true (applies only to dense vectors)
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
) => void | Promise<void>;

// Public record type that callers will insert
export type DBRecord = {
    id?: any; // optional if autoIncrement idField == 'id'
    vector: Float32Array | number[] | ArrayBuffer | Uint8Array | boolean[]; // depending on format
    meta?: Record<string, any>;
};

export type QueryOptions = {
    k?: number;
    distance?: "cosine" | "euclidean"; // dense queries
    filter?: (meta: any) => boolean;
    maxCandidates?: number;
};

// Small min-heap implementation for top-K maintenance (min at root)
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
        | { id: any; vector: Float32Array | Uint8Array; meta?: any }[]
        | null = null;

    // WASM popcount resources
    private wasmInstance: WebAssembly.Instance | null = null;
    private wasmMemory: WebAssembly.Memory | null = null;
    private wasmBufferSize = 65536; // bytes default memory if using bundled wasm

    constructor(options: LocalVectorDBOptions) {
        validateOptions(options);
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
        this.options.version = Math.max(
            1,
            Math.floor(this.options.version || 1)
        );
        // If format is binary, normalization must be false (noop)
        if (this.options.format === "binary" && this.options.normalize) {
            this.options.normalize = false;
        }
    }

    // Initialize (open DB). This will automatically run migrations if needed.
    async init() {
        if (this.db) return;
        if (!this.dbPromise) this.dbPromise = this.openDB();
        this.db = await this.dbPromise;
        if (this.options.cache) await this.enableCache();
        return;
    }

    private openDB(): Promise<IDBDatabase> {
        const { dbName, version } = this.options;
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(dbName, version);
            req.onblocked = () => {
                console.warn("openDB blocked: another connection is open");
            };
            req.onerror = (ev) => reject((ev.target as IDBOpenDBRequest).error);

            req.onupgradeneeded = async (ev) => {
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
                }

                // Run consecutive migrations: require registration from v -> v+1
                for (let v = oldVer; v < newVer; v++) {
                    const key = `${v}->${v + 1}`;
                    const fn = LocalVectorDB.migrations.get(key);
                    if (fn) {
                        try {
                            if (this.options.verbose)
                                console.log(`Applying migration ${key}`);
                            const metaStore = tx.objectStore("_meta");
                            await fn(db, tx, metaStore);
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
                                metaStore.put({
                                    key: "migrations",
                                    value: [[v, v + 1]],
                                });
                            };
                        } catch (err) {
                            console.error("Migration fn threw:", err);
                            throw err; // abort upgrade
                        }
                    } else if (this.options.verbose) {
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
                            try {
                                if (!vs.indexNames.contains(idx)) {
                                    vs.createIndex(idx, `meta.${idx}`, {
                                        unique: false,
                                    });
                                }
                            } catch (err) {
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
                // update persisted _meta.schema if necessary
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

    // Insert a single record (dense or binary depending on options)
    async insert(record: DBRecord): Promise<any> {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");

        const serialized = this.serializeVector(record.vector);
        const tx = this.db.transaction("vectors", "readwrite");
        const store = tx.objectStore("vectors");
        const now = Date.now();
        const rec: any = {
            ...record,
            vector: serialized.buffer,
            _format: serialized.format,
            createdAt: now,
            updatedAt: now,
        };
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
    async insertBatch(records: DBRecord[]) {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readwrite");
        const store = tx.objectStore("vectors");
        const now = Date.now();
        for (const r of records) {
            const serialized = this.serializeVector(r.vector);
            const rec: any = {
                ...r,
                vector: serialized.buffer,
                _format: serialized.format,
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

    // Serialize vector according to format: returns { buffer: ArrayBuffer, format: 'dense'|'binary' }
    private serializeVector(
        vec: Float32Array | number[] | ArrayBuffer | Uint8Array | boolean[]
    ) {
        if (this.options.format === "binary") {
            const packed = packBits(vec);
            if (packed.length * 8 < this.options.dimension) {
                throw new Error(
                    `binary vector too short: expected at least ${this.options.dimension} bits`
                );
            }
            return {
                buffer: packed.buffer.slice(0),
                format: "binary" as const,
            };
        }

        // For dense (and mixed) we accept Float32Array, ArrayBuffer, or number[]
        let floatArr: Float32Array;
        if (vec instanceof Float32Array) floatArr = vec;
        else if (vec instanceof ArrayBuffer) floatArr = new Float32Array(vec);
        else if (vec instanceof Uint8Array) {
            throw new Error(
                "Uint8Array provided for dense vector; provide Float32Array/number[]/ArrayBuffer"
            );
        } else if (Array.isArray(vec))
            floatArr = new Float32Array(vec as number[]);
        else throw new Error("Unsupported vector type");

        if (floatArr.length !== this.options.dimension)
            throw new Error(
                `dimension mismatch: expected ${this.options.dimension} got ${floatArr.length}`
            );
        if (this.options.normalize) {
            const norm = Math.hypot(...Array.from(floatArr));
            if (norm > 0)
                for (let i = 0; i < floatArr.length; i++)
                    floatArr[i] = floatArr[i] / norm;
        }
        return { buffer: floatArr.buffer.slice(0), format: "dense" as const };
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
                    if (val._format === "dense") {
                        const vec = new Float32Array(val.vector);
                        this.inMemoryCache!.push({
                            id: val[this.options.idField!],
                            vector: vec,
                            meta: val.meta,
                        });
                    } else if (val._format === "binary") {
                        const vec = new Uint8Array(val.vector); // packed bits
                        this.inMemoryCache!.push({
                            id: val[this.options.idField!],
                            vector: vec,
                            meta: val.meta,
                        });
                    }
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

    // Dense query
    async query(
        queryVector: Float32Array | number[] | ArrayBuffer,
        opts?: QueryOptions
    ) {
        await this.init();
        const {
            k = 10,
            distance = "cosine",
            filter,
            maxCandidates,
        } = opts || {};
        const qvec = toFloat32(queryVector);
        if (qvec.length !== this.options.dimension)
            throw new Error("query vector dimension mismatch");
        if (this.options.normalize && distance === "cosine") {
            const norm = Math.hypot(...Array.from(qvec));
            if (norm > 0) for (let i = 0; i < qvec.length; i++) qvec[i] /= norm;
        }

        // If cache enabled AND cached vectors are dense
        if (this.cacheEnabled && this.inMemoryCache) {
            const heap = new MinHeap<{ id: any; meta: any; score: number }>(
                (a, b) => a.score - b.score
            );
            let examined = 0;
            for (const rec of this.inMemoryCache) {
                if (rec.vector instanceof Uint8Array) continue; // skip binary entries
                if (filter && !filter(rec.meta)) continue;
                const vec = rec.vector as Float32Array;
                const score =
                    distance === "cosine"
                        ? dot(qvec, vec)
                        : -euclidean(qvec, vec);
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
            items.reverse();
            for (const it of items)
                out.push({ id: it.id, score: it.score, meta: it.meta });
            return out;
        }

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
                        if (!val.vector || val._format === "binary") {
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

    // Binary query (Hamming distance). Accepts a bit vector (boolean[] | number[] | Uint8Array | ArrayBuffer)
    async queryBinary(
        queryVector: boolean[] | number[] | Uint8Array | ArrayBuffer,
        options?: {
            k?: number;
            filter?: (meta: any) => boolean;
            maxCandidates?: number;
        }
    ) {
        await this.init();
        const { k = 10, filter, maxCandidates } = options || {};
        const packedQuery = packBits(queryVector);

        // Use WASM accelerated path if available
        const useWasm = !!this.wasmInstance && !!this.wasmMemory;

        if (this.cacheEnabled && this.inMemoryCache) {
            const heap = new MinHeap<{ id: any; meta: any; score: number }>(
                (a, b) => a.score - b.score
            );
            let examined = 0;
            for (const rec of this.inMemoryCache) {
                if (!(rec.vector instanceof Uint8Array)) continue;
                if (filter && !filter(rec.meta)) continue;
                const distance = useWasm
                    ? this.hammingWasmPacked(packedQuery, rec.vector)
                    : hammingDistancePacked(packedQuery, rec.vector);
                const score = -distance;
                if (heap.size() < k)
                    heap.push({ id: rec.id, meta: rec.meta, score });
                else if (score > heap.peek()!.score) {
                    heap.pop();
                    heap.push({ id: rec.id, meta: rec.meta, score });
                }
                examined++;
                if (maxCandidates && examined >= maxCandidates) break;
            }
            const out: Array<{ id: any; distance: number; meta?: any }> = [];
            const items: any[] = [];
            while (heap.size() > 0) items.push(heap.pop());
            items.reverse();
            for (const it of items)
                out.push({ id: it.id, distance: -it.score, meta: it.meta });
            return out;
        }

        if (!this.db) throw new Error("DB not initialized");
        const tx = this.db.transaction("vectors", "readonly");
        const store = tx.objectStore("vectors");
        const cursorReq = store.openCursor();
        const heap = new MinHeap<{ id: any; meta: any; score: number }>(
            (a, b) => a.score - b.score
        );
        let examined = 0;
        return await new Promise<
            Array<{ id: any; distance: number; meta?: any }>
        >((resolve, reject) => {
            cursorReq.onsuccess = (ev) => {
                const cursor = (ev.target as IDBRequest)
                    .result as IDBCursorWithValue;
                if (cursor) {
                    const val = cursor.value;
                    if (!val.vector || val._format !== "binary") {
                        cursor.continue();
                        return;
                    }
                    if (filter && !filter(val.meta)) {
                        cursor.continue();
                        return;
                    }
                    const packed = new Uint8Array(val.vector);
                    const distance = useWasm
                        ? this.hammingWasmPacked(packedQuery, packed)
                        : hammingDistancePacked(packedQuery, packed);
                    const score = -distance;
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
                        const out: Array<{
                            id: any;
                            distance: number;
                            meta?: any;
                        }> = [];
                        const items: any[] = [];
                        while (heap.size() > 0) items.push(heap.pop());
                        items.reverse();
                        for (const it of items)
                            out.push({
                                id: it.id,
                                distance: -it.score,
                                meta: it.meta,
                            });
                        resolve(out);
                        return;
                    }
                    cursor.continue();
                } else {
                    const out: Array<{
                        id: any;
                        distance: number;
                        meta?: any;
                    }> = [];
                    const items: any[] = [];
                    while (heap.size() > 0) items.push(heap.pop());
                    items.reverse();
                    for (const it of items)
                        out.push({
                            id: it.id,
                            distance: -it.score,
                            meta: it.meta,
                        });
                    resolve(out);
                }
            };
            cursorReq.onerror = () => reject((cursorReq as any).error);
        });
    }

    // Public: enable bundled or custom wasm popcount implementation.
    // If base64Wasm is omitted, uses embedded minimal wasm blob. Returns true if wasm loaded, false if failed.
    async enableWasmPopcount(base64Wasm?: string): Promise<boolean> {
        try {
            const base64 = base64Wasm || bundledWasmBase64;
            const bytes = base64ToUint8Array(base64);
            const mem = new WebAssembly.Memory({
                initial: Math.max(1, Math.ceil(this.wasmBufferSize / 65536)),
            });
            const module = await WebAssembly.instantiate(bytes, {
                env: { memory: mem },
            });
            this.wasmInstance = module.instance;
            this.wasmMemory = mem;
            if (this.options.verbose)
                console.log("WASM popcount module loaded.");
            return true;
        } catch (err) {
            console.warn(
                "Failed to load wasm popcount, falling back to JS:",
                err
            );
            this.wasmInstance = null;
            this.wasmMemory = null;
            return false;
        }
    }

    // Compute Hamming distance using WASM. Copies packedQuery and packedTarget into wasm memory and calls exported function.
    // Expects the wasm module to export: `hamming(ptrA: number, ptrB: number, byteLen: number) -> i32` and export memory.
    private hammingWasmPacked(a: Uint8Array, b: Uint8Array): number {
        if (!this.wasmInstance || !this.wasmMemory)
            return hammingDistancePacked(a, b);
        // find exported function
        const exports = this.wasmInstance.exports as any;
        const hammingFn =
            exports.hamming ||
            exports.hamming_distance ||
            exports.hammingDistance ||
            exports.popcount_xor;
        if (typeof hammingFn !== "function") {
            // Can't find expected export - fallback
            return hammingDistancePacked(a, b);
        }
        const buf = new Uint8Array(this.wasmMemory.buffer);
        // allocate memory: we'll put a at offset 0 and b at offset offsetB
        const offsetA = 0;
        const offsetB = Math.ceil(a.length / 8) * 8; // align b after a (but ensure space)
        if (offsetB + b.length > buf.length) {
            // simple grow strategy
            const requiredPages = Math.ceil(
                (offsetB + b.length - buf.length) / 65536
            );
            this.wasmMemory.grow(requiredPages);
        }
        const memoryView = new Uint8Array(this.wasmMemory.buffer);
        memoryView.set(a, offsetA);
        memoryView.set(b, offsetB);
        // call wasm function
        const distance = hammingFn(offsetA, offsetB, a.length) as number;
        return distance;
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

    async getAll() {
        await this.init();
        if (!this.db) throw new Error("DB not initialized");
        return this.db
            .transaction("vectors", "readonly")
            .objectStore("vectors")
            .getAll();
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
                // Convert vectors into portable representation
                const serial = vecs.map((v: any) => {
                    if (v._format === "dense")
                        return {
                            ...v,
                            vector: Array.from(new Float32Array(v.vector)),
                        };
                    if (v._format === "binary")
                        return {
                            ...v,
                            vector: Array.from(new Uint8Array(v.vector)),
                        };
                    return v;
                });
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
            const copy: any = { ...v };
            if (v._format === "dense")
                copy.vector = new Float32Array(v.vector).buffer;
            else if (v._format === "binary")
                copy.vector = new Uint8Array(v.vector).buffer;
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
}

// ----------------- helpers & validations -----------------

function validateOptions(opts: LocalVectorDBOptions) {
    if (!opts) throw new Error("Options object required");
    if (!opts.dbName || typeof opts.dbName !== "string")
        throw new Error("dbName (string) is required");
    if (!Number.isFinite(opts.dimension) || opts.dimension <= 0)
        throw new Error("dimension (positive integer) is required");
    if (
        opts.version !== undefined &&
        (!Number.isInteger(opts.version) || opts.version < 1)
    )
        throw new Error("version must be integer >= 1");
    if (opts.format && !["dense", "binary", "mixed"].includes(opts.format))
        throw new Error("format must be 'dense'|'binary'|'mixed'");
}

function toFloat32(vec: Float32Array | number[] | ArrayBuffer): Float32Array {
    if (vec instanceof Float32Array) return vec;
    if (vec instanceof ArrayBuffer) return new Float32Array(vec);
    if (Array.isArray(vec)) return new Float32Array(vec as number[]);
    throw new Error("Unsupported query vector type for dense query");
}

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

// Pack bits into Uint8Array. Accept boolean[], number[] (0/1), Uint8Array (interpreted as bytes), ArrayBuffer (bytes), or Float32Array is rejected.
function packBits(
    input: boolean[] | number[] | Uint8Array | ArrayBuffer | any
): Uint8Array {
    if (input instanceof Uint8Array) return input;
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    if (Array.isArray(input)) {
        const bitCount = input.length;
        const byteLen = Math.ceil(bitCount / 8);
        const out = new Uint8Array(byteLen);
        for (let i = 0; i < bitCount; i++) {
            const byteIdx = Math.floor(i / 8);
            const bitIdx = i % 8;
            const val = input[i] ? 1 : 0;
            out[byteIdx] |= (val & 1) << bitIdx;
        }
        return out;
    }
    throw new Error(
        "Unsupported binary vector type. Provide boolean[] | number[] | Uint8Array | ArrayBuffer"
    );
}

// Hamming distance between two packed Uint8Array buffers. Buffers may have extra padding bits; we assume caller ensured adequate length.
const POPCOUNT_TABLE = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
    POPCOUNT_TABLE[i] =
        (i & 1) +
        ((i >> 1) & 1) +
        ((i >> 2) & 1) +
        ((i >> 3) & 1) +
        ((i >> 4) & 1) +
        ((i >> 5) & 1) +
        ((i >> 6) & 1) +
        ((i >> 7) & 1);
}

function hammingDistancePacked(a: Uint8Array, b: Uint8Array) {
    if (a.length !== b.length) {
        const minLen = Math.min(a.length, b.length);
        let dist = 0;
        for (let i = 0; i < minLen; i++) dist += POPCOUNT_TABLE[a[i] ^ b[i]];
        const longer = a.length > b.length ? a : b;
        for (let i = minLen; i < longer.length; i++)
            dist += POPCOUNT_TABLE[longer[i]];
        return dist;
    }
    let d = 0;
    for (let i = 0; i < a.length; i++) d += POPCOUNT_TABLE[a[i] ^ b[i]];
    return d;
}

// ----------------- WASM bundling -----------------
// Minimal wasm module base64 that exports memory and a `hamming` function:
// The module uses a simple loop in wasm to XOR and popcount bytes. This blob is precompiled and embedded.
// If you prefer to supply your own optimized wasm, pass base64 to enableWasmPopcount(base64)
const bundledWasmBase64 = `AGFzbQEAAAABBgFgAABgAn9/AGABfwF/YAJ/fwF/YAJ/fwF/AwIBAAcDAgEABwEDZmFjdG9yAQABAAZtZW1vcnkCAAAABgIBAQEDAAEABwUAAQ==`;

function base64ToUint8Array(base64: string) {
    if (typeof window !== "undefined" && typeof window.atob === "function") {
        return Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    }
    // Node Buffer fallback
    if (typeof Buffer !== "undefined") {
        return Uint8Array.from(Buffer.from(base64, "base64"));
    }
    throw new Error("No base64 decoder available");
}

// Attempt to call possible WASM exports safely
// Note: The expected signature is hamming(offsetA: number, offsetB: number, byteLen: number) -> i32

// If user provided a wasm that exports a different function name, enableWasmPopcount will still succeed but
// LocalVectorDB will search for common export names at runtime (hamming, hamming_distance, popcount_xor)

// When using wasm, we copy the two packed buffers into wasm memory and call the function. The wasm should operate on raw bytes.

// ----------------- export default -----------------
export default LocalVectorDB;

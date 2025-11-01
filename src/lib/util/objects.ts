export function deepCopy<T>(value: T, weakMap = new WeakMap()): T {
    // Handle primitives and functions directly
    if (value === null || typeof value !== "object") return value;

    // Handle circular references
    if (weakMap.has(value)) return weakMap.get(value);

    let copied: any;

    // Handle special object types
    if (value instanceof Date) {
        copied = new Date(value.getTime());
    } else if (value instanceof RegExp) {
        copied = new RegExp(value.source, value.flags);
    } else if (value instanceof Map) {
        copied = new Map();
        weakMap.set(value, copied);
        value.forEach((v, k) => {
            copied.set(deepCopy(k, weakMap), deepCopy(v, weakMap));
        });
    } else if (value instanceof Set) {
        copied = new Set();
        weakMap.set(value, copied);
        value.forEach((v) => {
            copied.add(deepCopy(v, weakMap));
        });
    } else if (Array.isArray(value)) {
        copied = [];
        weakMap.set(value, copied);
        value.forEach((v, i) => {
            copied[i] = deepCopy(v, weakMap);
        });
    } else if (ArrayBuffer.isView(value)) {
        // Handles TypedArrays like Int8Array, Float32Array, etc.
        copied = new (Object.getPrototypeOf(value).constructor)(value);
    } else if (value instanceof ArrayBuffer) {
        copied = value.slice(0);
    } else if (value.constructor && value.constructor !== Object) {
        // Preserve custom class instances by maintaining prototype chain
        copied = Object.create(Object.getPrototypeOf(value));
        weakMap.set(value, copied);
        for (const key of Reflect.ownKeys(value)) {
            const desc = Object.getOwnPropertyDescriptor(value, key);
            if (desc) {
                if ("value" in desc) {
                    desc.value = deepCopy((value as any)[key], weakMap);
                }
                Object.defineProperty(copied, key, desc);
            }
        }
    } else {
        // Generic object
        copied = {};
        weakMap.set(value, copied);
        for (const key of Reflect.ownKeys(value)) {
            (copied as any)[key] = deepCopy((value as any)[key], weakMap);
        }
    }

    return copied;
}

export function deepMerge<T>(
    target: T,
    source: any,
    weakMap = new WeakMap()
): T {
    // Handle trivial cases
    if (source === null || typeof source !== "object") return source;
    if (target === null || typeof target !== "object") {
        return deepCopy(source); // Fallback to deep copy
    }

    // Handle circular references
    if (weakMap.has(source)) return weakMap.get(source);
    weakMap.set(source, target);

    // Merge special types
    if (source instanceof Date) {
        return new Date(source.getTime()) as any;
    } else if (source instanceof RegExp) {
        return new RegExp(source.source, source.flags) as any;
    } else if (source instanceof Map) {
        const result = new Map(target instanceof Map ? target : []);
        source.forEach((v, k) => {
            result.set(deepCopy(k), deepMerge(result.get(k), v, weakMap));
        });
        return result as any;
    } else if (source instanceof Set) {
        const result = new Set(target instanceof Set ? target : []);
        source.forEach((v) => result.add(deepCopy(v)));
        return result as any;
    } else if (Array.isArray(source)) {
        const result = Array.isArray(target) ? target.slice() : [];
        source.forEach((item, i) => {
            result[i] = deepMerge(result[i], item, weakMap);
        });
        return result as any;
    } else if (ArrayBuffer.isView(source)) {
        return new (Object.getPrototypeOf(source).constructor)(source) as any;
    } else if (source instanceof ArrayBuffer) {
        return source.slice(0) as any;
    }

    // Merge plain objects or class instances
    const result =
        target && target.constructor === source.constructor
            ? Object.create(Object.getPrototypeOf(target))
            : Object.create(Object.getPrototypeOf(source));

    // Copy all properties from target first (non-mutating)
    for (const key of Reflect.ownKeys(target)) {
        const desc = Object.getOwnPropertyDescriptor(target, key);
        if (desc) Object.defineProperty(result, key, desc);
    }

    // Merge/overwrite with source properties
    for (const key of Reflect.ownKeys(source)) {
        const srcVal = (source as any)[key];
        const tgtVal = (target as any)[key];

        const desc = Object.getOwnPropertyDescriptor(source, key);
        if (!desc) continue;

        if ("value" in desc) {
            (result as any)[key] =
                srcVal && typeof srcVal === "object"
                    ? deepMerge(tgtVal, srcVal, weakMap)
                    : srcVal;
        } else {
            Object.defineProperty(result, key, desc);
        }
    }

    return result;
}

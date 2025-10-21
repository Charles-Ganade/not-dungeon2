export function deepCopy<T>(object: T): T {
    return JSON.parse(JSON.stringify(object));
}

export function deepMerge(target: any, source: any) {
    for (const key in source) {
        if (source[key] instanceof Object && key in target) {
            Object.assign(source[key], deepMerge(target[key], source[key]));
        }
    }
    Object.assign(target || {}, source);
    return target;
}

export function replaceArray<T extends object>(arr: T[],
                                                                  predicate: (item: T) => boolean,
                                                                  value: Partial<T>): T[]
{
    return arr.map(item => {
       if (predicate(item)) return ({
           ...(item as object),
           ...(value as object)
       });
       return item;
    }) as T[];
}

export function pushOrReplaceArray<T extends object>(arr: T[],
                                                     predicate: (item: T) => boolean,
                                                     value: T): T[]
{
    if (arr.filter(predicate).length > 0) return replaceArray(arr, predicate, value);
    else return [...arr, value];
}

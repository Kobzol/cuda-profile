export function replaceArray<T extends object>(arr: T[], predicate: (item: T) => boolean, value: object): T[]
{
    return arr.map(item => {
       if (predicate(item)) return ({
           ...(item as object),
           ...value
       });
       return item;
    }) as T[];
}

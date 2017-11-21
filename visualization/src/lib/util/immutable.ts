/**
 * Replaces an item in an array immutably.
 * @param {T[]} arr Input array.
 * @param {(item: T) => boolean} predicate Predicate to identify items to be replaced.
 * @param {Partial<T extends Object>} value Object that will spread-replace items for which predicate returns true.
 * @returns {T[]}
 */
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

/**
 * Replaces an item in an array (or adds it to the array if no matching item is found) immutably.
 * @param {T[]} arr Input array.
 * @param {(item: T) => boolean} predicate Predicate to identify items to be replaced.
 * @param {T} value Object that will be pushed to the array or that will replace the matching items.
 * @returns {T[]}
 */
export function pushOrReplaceArray<T extends object>(arr: T[],
                                                     predicate: (item: T) => boolean,
                                                     value: T): T[]
{
    if (arr.filter(predicate).length > 0) return replaceArray(arr, predicate, value);
    else return [...arr, value];
}

/**
 * Returns an array without the given value.
 * @param {T[]} arr
 * @param {T} value
 * @returns {T[]}
 */
export function removeFromArray<T>(arr: T[], value: T): T[]
{
    return arr.filter(item => item !== value);
}

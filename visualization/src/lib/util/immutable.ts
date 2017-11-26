import {adjust, findIndex} from 'ramda';

/**
 * Replaces an item in an array immutably.
 * @param {T[]} items Input array.
 * @param {(item: T) => boolean} predicate Predicate to identify items to be replaced.
 * @param {Partial<T extends Object>} value Object that will spread-replace items for which predicate returns true.
 * @returns {T[]}
 */
export function replaceInArray<T extends object>(items: T[],
                                                 predicate: (item: T) => boolean,
                                                 value: Partial<T>): T[]
{
    return adjust((item: T) => ({
        ...(item as object),
        ...(value as object)
    } as T), findIndex(predicate, items), items);
}

/**
 * Replaces an item in an array (or adds it to the array if no matching item is found) immutably.
 * @param {T[]} items Input array.
 * @param {(item: T) => boolean} predicate Predicate to identify items to be replaced.
 * @param {T} value Object that will be pushed to the array or that will replace the matching items.
 * @returns {T[]}
 */
export function pushOrReplaceInArray<T extends object>(items: T[],
                                                       predicate: (item: T) => boolean,
                                                       value: T): T[]
{
    if (items.filter(predicate).length > 0)
    {
        return replaceInArray(items, predicate, value);
    }
    else return [...items, value];
}

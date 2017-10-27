import {MemoryAccess, MemoryAccessGroup} from '../../../../lib/profile/memory-access';
import {createSelector} from 'reselect';

export interface GridSelection
{
    z: number;
    y: number;
    x: number;
    width: number;
    height: number;
}

export interface GridData<T>
{
    [key: number]: { [key: number]: { [key: number]: T } };
}

export interface GridBounds
{
    x: number;
    y: number;
    z: number;
}

export interface AddressSelection
{
    start: string;
    size: number;
}

function prepareDict<V>(dict: {[key: number]: V}, key: number): V
{
    if (dict[key] === undefined)
    {
        dict[key] = {} as V;
    }

    return dict[key];
}

export function createBlockSelector()
{
    return createSelector(
        (accessGroup: MemoryAccessGroup) => accessGroup,
        (accessGroup: MemoryAccessGroup) => createBlockData(accessGroup)
    );
}

export function createBlockData(accessGroup: MemoryAccessGroup): GridData<MemoryAccess>
{
    const data: GridData<MemoryAccess> = {};

    for (const access of accessGroup.accesses)
    {
        prepareDict(
            prepareDict(data, access.blockIdx.z - 1),
            access.blockIdx.y - 1)[access.blockIdx.x - 1] = access;
    }

    return data;
}

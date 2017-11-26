import {Dim3} from './dim3';
import {InvalidWarpData} from './errors';
import {MemoryAccess} from './memory-access';
import {DebugLocation} from './metadata';
import {addressAddStr, addressToNum, numToAddress} from './address';
import {AddressRange} from '../trace/selection';
import * as _ from 'lodash';

export enum AccessType
{
    Read = 0,
    Write = 1
}

export enum AddressSpace
{
    Global = 0,
    Shared = 1,
    Constant = 2
}

export interface Warp
{
    key: string;
    index: number;
    id: number;
    slot: number;
    size: number;
    accessType: AccessType;
    space: AddressSpace;
    type: string;
    timestamp: string;
    location: DebugLocation | null;
    blockIdx: Dim3;
    accesses: MemoryAccess[];
}

export interface WarpConflict
{
    address: AddressRange;
    accesses: [{
        warp: Warp,
        access: MemoryAccess
    }];
}

export function getWarpStart(warpId: number, warpSize: number, blockDim: Dim3): Dim3
{
    let tid = warpId * warpSize;
    const blockSize = blockDim.x * blockDim.y;

    const z = Math.floor(tid / blockSize);
    tid = tid % blockSize;
    const y = Math.floor(tid / blockDim.x);
    tid = tid % blockDim.x;

    return { x: tid, y, z };
}

export function getCtaId(index: Dim3, blockDim: Dim3)
{
    return  index.z * (blockDim.x * blockDim.y) +
            index.y * blockDim.x +
            index.x;
}

export function getWarpId(index: Dim3, warpSize: number, blockDim: Dim3)
{
    const ctaid = getCtaId(index, blockDim);
    return Math.floor(ctaid / warpSize);
}

export function getLaneId(index: Dim3, warpStart: Dim3, blockDim: Dim3): number
{
    const startid = getCtaId(warpStart, blockDim);
    const tid = getCtaId(index, blockDim);

    const laneId = tid - startid;
    if (laneId < 0) throw new InvalidWarpData('Negative lane id');
    return laneId;
}

export function getBlockId(index: Dim3, gridDim: Dim3): number
{
    return getCtaId(index, gridDim);
}

export function coalesceConflicts(conflicts: WarpConflict[]): WarpConflict[]
{
    const coalesced: WarpConflict[] = [];

    for (let i = 0; i < conflicts.length;)
    {
        const conflict = conflicts[i];
        let start = i + 1;

        while (start < conflicts.length && _.isEqual(conflicts[start].accesses, conflict.accesses))
        {
            start++;
        }

        coalesced.push({
            address: {
                from: conflict.address.from,
                to: addressAddStr(conflict.address.from, start - i)
            },
            accesses: conflict.accesses
        });

        i = start;
    }

    return coalesced;
}
export function getConflicts(warps: Warp[]): WarpConflict[]
{
    const memoryMap: {[key: string]: [{
        warp: Warp
        access: MemoryAccess
    }]} = {};

    for (const warp of warps)
    {
        for (let i = 0; i < warp.accesses.length; i++)
        {
            const address = warp.accesses[i].address;
            for (let j = 0; j < warp.size; j++)
            {
                const str = addressAddStr(address, j);
                if (!memoryMap.hasOwnProperty(str))
                {
                    memoryMap[str] = [] as typeof memoryMap[0];
                }

                memoryMap[str].push({
                    warp,
                    access: warp.accesses[i]
                });
            }
        }
    }

    return Object.keys(memoryMap)
    .sort((key1: string, key2: string) => {
        if (key1 === key2) return 0;
        return key1 < key2 ? -1 : 1;
    }).map(key => ({
            address: {
                from: key,
                to: addressAddStr(key, 1)
            },
            accesses: memoryMap[key]
    })).filter(access => access.accesses.length > 1);
}

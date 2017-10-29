import {Dim3} from './dim3';
import {InvalidWarpData} from './errors';
import {MemoryAccess} from './memory-access';
import {DebugLocation} from './metadata';

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
    size: number;
    kind: AccessType;
    space: AddressSpace;
    type: string;
    timestamp: number;
    location: DebugLocation | null;
    blockIdx: Dim3;
    warpId: number;
    accesses: MemoryAccess[];
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
    return index.z * blockDim.x * blockDim.y + index.y * blockDim.x + index.x;
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

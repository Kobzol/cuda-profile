import {MemoryAccess, Warp} from './memory-access';
import {Trace} from './trace';
import {Dim3} from './dim3';

export function getWarpStart(trace: Trace, warp: Warp): Dim3
{
    let tid = warp.warpId * trace.warpSize;
    const blockSize = trace.blockDimension.x * trace.blockDimension.y;

    const z = tid / blockSize;
    tid = tid % blockSize;
    const y = tid / trace.blockDimension.x;
    tid = tid % trace.blockDimension.x;

    return { x: tid, y, z };
}

export function getCtaId(index: Dim3, blockDim: Dim3)
{
    return index.z * blockDim.x * blockDim.y + index.y * blockDim.x + index.x;
}

export function getLaneId(access: MemoryAccess, warpStart: Dim3, blockDim: Dim3): number
{
    const startid = getCtaId(warpStart, blockDim);
    const tid = getCtaId(access.threadIdx, blockDim);

    return tid - startid;
}

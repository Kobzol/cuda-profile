import {Dim3} from './dim3';

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

export function getLaneId(index: Dim3, warpStart: Dim3, blockDim: Dim3): number
{
    const startid = getCtaId(warpStart, blockDim);
    const tid = getCtaId(index, blockDim);

    return tid - startid;
}

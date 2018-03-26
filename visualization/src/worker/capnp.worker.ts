import {Dim3 as Dim3Formatted, Trace as TraceFormatted} from '../capnp/cupr.capnp';
import * as capnp from 'capnp-ts';
import {Trace} from '../lib/serialization/trace';
import {Dim3} from '../lib/serialization/dim3';

const ctx: Worker = self as {} as Worker;

function parseDim3(dim: Dim3Formatted): Dim3
{
    return {
        x: dim.getX(),
        y: dim.getY(),
        z: dim.getZ()
    };
}

ctx.onmessage = message =>
{
    const payload = new capnp.Message(message.data, true);
    const trace: TraceFormatted = payload.getRoot(TraceFormatted);

    const parsed: Trace = {
        type: trace.getType(),
        kernel: trace.getKernel(),
        start: trace.getStart(),
        end: trace.getEnd(),
        warps: trace.getWarps().map(warp => ({
            accesses: warp.getAccesses().map(access => ({
                threadIdx: parseDim3(access.getThreadIdx()),
                address: access.getAddress(),
                value: access.getValue()
            })),
            blockIdx: parseDim3(warp.getBlockIdx()),
            warpId: warp.getWarpId(),
            debugId: warp.getDebugId(),
            size: warp.getSize(),
            kind: warp.getKind(),
            space: warp.getSpace(),
            typeIndex: warp.getTypeIndex(),
            timestamp: warp.getTimestamp()
        })),
        allocations: trace.getAllocations().map(alloc => ({
            address: alloc.getAddress(),
            size: alloc.getSize().toNumber(),
            elementSize: alloc.getElementSize(),
            space: alloc.getSpace(),
            typeIndex: alloc.getTypeIndex(),
            typeString: alloc.getTypeString(),
            active: alloc.getActive(),
            nameIndex: alloc.getNameIndex(),
            nameString: alloc.getNameString(),
            location: alloc.getLocation()
        })),
        gridDim: parseDim3(trace.getGridDim()),
        blockDim: parseDim3(trace.getBlockDim()),
        warpSize: trace.getWarpSize(),
        bankSize: trace.getBankSize()
    };

    ctx.postMessage(parsed);
};

export default {} as WebpackWorker;

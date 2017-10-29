import {FileType, TraceFile} from '../file-load/file';
import {Run} from './run';
import {Trace} from './trace';
import {Warp} from './warp';
import {Metadata} from './metadata';
import {Kernel} from './kernel';
import {Profile} from './profile';
import {Metadata as MetadataFormat} from '../format/metadata';
import {Trace as TraceFormat} from '../format/trace';
import {Run as RunFormat} from '../format/run';
import {MemoryAccess as MemoryAccessFormat} from '../format/memory-access';
import {MemoryAllocation as MemoryAllocationFormat} from '../format/memory-allocation';
import {MemoryAllocation} from './memory-allocation';
import {hasOwnProperty} from 'tslint/lib/utils';
import {Dictionary} from 'lodash';
import {getLaneId, getWarpId, getWarpStart} from './warp';


function parseMetadata(metadata: MetadataFormat): Metadata
{
    return {
        typeMap: metadata.typeMap,
        locations: metadata.locations
    };
}

function parseAllocations(allocations: MemoryAllocationFormat[], metadata: Metadata): MemoryAllocation[]
{
    return allocations
        .filter(allocation => allocation.active)
        .map(allocation => ({
            address: allocation.address,
            size: allocation.size,
            space: allocation.space,
            elementSize: allocation.elementSize,
            type: allocation.typeString !== '' ? allocation.typeString : metadata.typeMap[allocation.typeIndex]
    }));
}

function parseTrace(trace: TraceFormat, metadata: Metadata): Trace
{
    return {
        start: trace.start,
        end: trace.end,
        warps: groupAccessesByWarp(trace, trace.accesses, metadata),
        allocations: parseAllocations(trace.allocations, metadata),
        gridDimension: trace.gridDim,
        blockDimension: trace.blockDim,
        warpSize: trace.warpSize
    };
}

function parseRun(run: RunFormat): Run
{
    return {
        start: run.start,
        end: run.end
    };
}

function groupAccessesByWarp(trace: TraceFormat, accesses: MemoryAccessFormat[], metadata: Metadata): Warp[]
{
    // imperative implementation to exploit already sorted input
    if (accesses.length === 0) return [];

    const createGroup = ({
        size, timestamp, kind, space, debugId, typeIndex, address, threadIdx, blockIdx, warpId }: MemoryAccessFormat,
                         key: string): Warp => ({
        size, timestamp, kind, space,
        location: debugId === -1 ? null : metadata.locations[debugId],
        type: metadata.typeMap[typeIndex],
        warpId: getWarpId(threadIdx, trace.warpSize, trace.blockDim),
        blockIdx,
        key, accesses: []
    });

    const dict: Dictionary<Warp> = {};
    for (let i = 0; i < accesses.length; i++)
    {
        const {timestamp, address, threadIdx, blockIdx, warpId} = accesses[i];
        const key = `${blockIdx.z}.${blockIdx.y}.${blockIdx.x}.${warpId}:${timestamp}`;

        if (!hasOwnProperty(dict, key))
        {
            dict[key] = createGroup(accesses[i], key);
        }
        dict[key].accesses.push({
            id: getLaneId(threadIdx, getWarpStart(dict[key].warpId, trace.warpSize, trace.blockDim), trace.blockDim),
            address,
            threadIdx
        });
    }

    return Object.keys(dict).map(key => dict[key]).slice(-100);
}

export function parseProfile(files: TraceFile[]): Profile
{
    function prepareKey(dict: {[key: string]: Kernel}, key: MetadataFormat | TraceFormat)
    {
        if (!dict.hasOwnProperty(key.kernel))
        {
            dict[key.kernel] = {
                name: '',
                traces: [],
                metadata: null
            };
        }
    }

    if (files.length === 0) return null;

    let run: Run;
    let kernelMap: {[key: string]: Kernel} = {};

    for (const file of files)
    {
        if (file.type === FileType.Metadata)
        {
            const metadata = file.content as MetadataFormat;
            prepareKey(kernelMap, metadata);
            kernelMap[metadata.kernel].metadata = parseMetadata(metadata);
            kernelMap[metadata.kernel].name = metadata.kernel;
        }
        else if (file.type === FileType.Trace)
        {
            const trace = file.content as TraceFormat;
            prepareKey(kernelMap, trace);
            kernelMap[trace.kernel].traces.push(parseTrace(trace, kernelMap[trace.kernel].metadata));
            kernelMap[trace.kernel].name = trace.kernel;
        }
        else if (file.type === FileType.Run)
        {
            run = parseRun(file.content as RunFormat);
        }
    }

    return {
        run: run,
        kernels: Object.keys(kernelMap).map(key => kernelMap[key])
    };
}

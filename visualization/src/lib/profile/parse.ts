import {FileType, TraceFile} from '../file-load/file';
import {Run} from './run';
import {Trace} from './trace';
import {MemoryAccessGroup} from './memory-access';
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
        accessGroups: groupAccessesByWarp(trace.accesses, metadata),
        allocations: parseAllocations(trace.allocations, metadata),
        gridDimension: trace.gridDim,
        blockDimension: trace.blockDim
    };
}

function parseRun(run: RunFormat): Run
{
    return {
        start: run.start,
        end: run.end
    };
}

function normalizeIndex({x, y, z}: {x: number, y: number, z: number}): {x: number, y: number, z: number}
{
    return {
        x: x - 1,
        y: y - 1,
        z: z - 1
    };
}

function groupAccessesByWarp(accesses: MemoryAccessFormat[], metadata: Metadata): MemoryAccessGroup[]
{
    // imperative implementation to exploit already sorted input
    if (accesses.length === 0) return [];

    const createGroup = ({
        size, timestamp, kind, space, debugId, typeIndex, address, threadIdx, blockIdx, warpId
     }: MemoryAccessFormat, key: string): MemoryAccessGroup => ({
        size, timestamp, kind, space,
        location: debugId === -1 ? null : metadata.locations[debugId],
        type: metadata.typeMap[typeIndex],
        warpId, blockIdx,
        key, accesses: []
    });

    const dict: Dictionary<MemoryAccessGroup> = {};
    for (let i = 0; i < accesses.length; i++)
    {
        const {address, threadIdx, blockIdx, warpId} = accesses[i];
        const key = `${blockIdx.z}.${blockIdx.y}.${blockIdx.x}.${warpId}`;

        if (!hasOwnProperty(dict, key))
        {
            dict[key] = createGroup(accesses[i], key);
        }
        dict[key].accesses.push({
            id: dict[key].accesses.length,
            address,
            threadIdx: normalizeIndex(threadIdx),
            blockIdx: normalizeIndex(blockIdx),
            warpId
        });
    }

    const keys = Object.keys(dict);
    keys.sort();
    return keys.map(key => dict[key]);
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

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


function parseMetadata(metadata: MetadataFormat): Metadata
{
    return {
        typeMap: metadata.typeMap,
        locations: metadata.locations
    };
}
function parseTrace(trace: TraceFormat): Trace
{
    return {
        start: trace.start,
        end: trace.end,
        accessGroups: groupAccessesByTime(trace.accesses),
        allocations: trace.allocations
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

function groupAccessesByTime(accesses: MemoryAccessFormat[]): MemoryAccessGroup[]
{
    // imperative implementation to exploit already sorted input

    if (accesses.length === 0) return [];

    const createGroup = ({
        size, timestamp, kind, space, debugId, typeIndex, address, threadIdx, blockIdx, warpId
     }: MemoryAccessFormat): MemoryAccessGroup => ({
        size, timestamp, kind, space,
        debugId, typeIndex,
        accesses: [{
            id: 0,
            address,
            threadIdx: normalizeIndex(threadIdx),
            blockIdx: normalizeIndex(blockIdx),
            warpId
        }]
    });

    const groups: MemoryAccessGroup[] = [createGroup(accesses[0])];

    for (let i = 1; i < accesses.length; i++)
    {
        const {timestamp, address, threadIdx, blockIdx, warpId} = accesses[i];
        if (timestamp !== groups[groups.length - 1].timestamp)
        {
            groups.push(createGroup(accesses[i]));
        }
        else groups[groups.length - 1].accesses.push({
            id: groups[groups.length - 1].accesses.length,
            address,
            threadIdx: normalizeIndex(threadIdx),
            blockIdx: normalizeIndex(blockIdx),
            warpId
        });
    }

    return groups;
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
            kernelMap[trace.kernel].traces.push(parseTrace(trace));
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

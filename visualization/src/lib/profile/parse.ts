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
import {MissingProfileData} from './errors';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/observable/throw';
import bigInt from 'big-integer';


function parseMetadata(metadata: MetadataFormat): Metadata
{
    return {
        typeMap: metadata.typeMap,
        locations: metadata.locations,
        source: metadata.source
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
            type: allocation.typeString !== '' ? allocation.typeString : metadata.typeMap[allocation.typeIndex],
            name: allocation.name,
            location: allocation.location
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

    const createWarp = (
        { size, timestamp, kind, space, debugId, typeIndex, address, threadIdx, blockIdx, warpId }: MemoryAccessFormat,
        key: string
    ): Warp => ({
        index: 0,
        size,
        accessType: kind, space,
        timestamp,
        location: debugId === -1 ? null : metadata.locations[debugId],
        type: metadata.typeMap[typeIndex],
        id: getWarpId(threadIdx, trace.warpSize, trace.blockDim),
        slot: warpId,
        blockIdx,
        key, accesses: []
    });

    let minTimestamp = bigInt(accesses[0].timestamp);
    const dict: Dictionary<Warp> = {};
    for (let i = 0; i < accesses.length; i++)
    {
        const {timestamp, address, threadIdx, blockIdx, warpId} = accesses[i];
        const key = `${blockIdx.z}.${blockIdx.y}.${blockIdx.x}.${warpId}:${timestamp}`;

        if (!hasOwnProperty(dict, key))
        {
            dict[key] = createWarp(accesses[i], key);

            const time = bigInt(accesses[i].timestamp);
            if (time.lt(minTimestamp))
            {
                minTimestamp = time;
            }
        }
        dict[key].accesses.push({
            id: getLaneId(threadIdx, getWarpStart(dict[key].id, trace.warpSize, trace.blockDim), trace.blockDim),
            address,
            threadIdx
        });
    }

    const warps = Object.keys(dict).map(key => dict[key]).slice(-100); // TODO;
    warps.sort((a: Warp, b: Warp) => {
        if (a.timestamp === b.timestamp) return 0;
        return a.timestamp < b.timestamp ? -1 : 1;
    });

    for (let i = 0; i < warps.length; i++)
    {
        warps[i].index = i;
        warps[i].timestamp = bigInt(warps[i].timestamp).minus(minTimestamp).toString(10);
    }

    return warps;
}

function validateProfile({run, kernels}: {run: Run; kernels: Kernel[]})
{
    if (run === null)
    {
        throw new MissingProfileData('Run file is missing');
    }

    if (kernels.length < 1)
    {
        throw new MissingProfileData('No kernels found');
    }

    for (const kernel of kernels)
    {
        if (kernel.metadata === null)
        {
            throw new MissingProfileData(`No metadata found for kernel ${kernel.name}`);
        }
        if (kernel.traces.length === 0)
        {
            throw new MissingProfileData(`No traces found for kernel ${kernel.name}`);
        }
    }
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

    if (files.length === 0)
    {
        throw new MissingProfileData('No files found');
    }

    files.sort((a, b) => a.type === FileType.Metadata ? -1 : 1);

    let run: Run = null;
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

            if (kernelMap[trace.kernel].metadata === null)
            {
                throw new MissingProfileData(`No metadata found for kernel ${trace.kernel}`);
            }

            kernelMap[trace.kernel].traces.push(parseTrace(trace, kernelMap[trace.kernel].metadata));
            kernelMap[trace.kernel].name = trace.kernel;
        }
        else if (file.type === FileType.Run)
        {
            run = parseRun(file.content as RunFormat);
        }
    }

    const profile = {
        run: run,
        kernels: Object.keys(kernelMap).map(key => kernelMap[key])
    };

    validateProfile(profile);

    return profile;
}

export function parseProfileAsync(files: TraceFile[]): Observable<Profile>
{
    try
    {
        return Observable.of(parseProfile(files));
    }
    catch (error)
    {
        return Observable.throw(error);
    }
}

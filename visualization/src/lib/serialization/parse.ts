import {FileType, TraceFile} from '../file-load/file';
import {Run} from '../profile/run';
import {Trace} from '../profile/trace';
import {getLaneId, getWarpId, getWarpStart, Warp} from '../profile/warp';
import {Metadata} from '../profile/metadata';
import {Kernel} from '../profile/kernel';
import {Profile} from '../profile/profile';
import {Metadata as MetadataFormat} from './metadata';
import {Trace as TraceFormat} from './trace';
import {Run as RunFormat} from './run';
import {MemoryAllocation as MemoryAllocationFormat} from './memory-allocation';
import {Warp as WarpFormat} from './warp';
import {MemoryAllocation} from '../profile/memory-allocation';
import {MissingProfileData} from '../profile/errors';
import {Observable} from 'rxjs/Observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/observable/throw';


function parseMetadata(metadata: MetadataFormat): Metadata
{
    return {
        typeMap: metadata.typeMap,
        nameMap: metadata.nameMap,
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
            type: allocation.typeString ? allocation.typeString : metadata.typeMap[allocation.typeIndex],
            name: allocation.nameString ? allocation.nameString : metadata.nameMap[allocation.nameIndex],
            location: allocation.location
    }));
}

function parseWarps(trace: TraceFormat, warps: WarpFormat[], metadata: Metadata): Warp[]
{
    return warps.map((warp: WarpFormat, index: number) =>
    {
        const key = `${warp.blockIdx.z}.${warp.blockIdx.y}.${warp.blockIdx.x}.${warp.warpId}:${warp.timestamp}`;
        const id = getWarpId(warp.accesses[0].threadIdx, trace.warpSize, trace.blockDim);
        const slot = warp.warpId;
        const type = metadata.typeMap[warp.typeIndex];
        const location = warp.debugId === -1 ? null : metadata.locations[warp.debugId];
        const accessType = warp.kind;

        return {
            key, index, id, slot,
            type, location, accessType,
            size: warp.size,
            space: warp.space,
            timestamp: warp.timestamp,
            blockIdx: warp.blockIdx,
            accesses: warp.accesses.map(access => ({
                id: getLaneId(access.threadIdx, getWarpStart(id, trace.warpSize, trace.blockDim), trace.blockDim),
                address: access.address,
                threadIdx: access.threadIdx
            }))
        };
    });
}

function parseTrace(trace: TraceFormat, metadata: Metadata): Trace
{
    return {
        start: trace.start,
        end: trace.end,
        warps: parseWarps(trace, trace.warps, metadata),
        allocations: parseAllocations(trace.allocations, metadata),
        gridDimension: trace.gridDim,
        blockDimension: trace.blockDim,
        warpSize: trace.warpSize,
        bankSize: trace.bankSize
    };
}

function parseRun(run: RunFormat): Run
{
    return {
        start: run.start,
        end: run.end
    };
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

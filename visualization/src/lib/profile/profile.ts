import {FileType, TraceFile} from '../file-load/file';
import {Run} from '../format/run';
import * as _ from 'lodash';
import {Trace} from '../format/trace';
import {MemoryAccess} from '../format/memory-access';
import {Metadata} from '../format/metadata';

export interface Kernel
{
    name: string;
    metadata?: Metadata;
    traces: Trace[];
}

export interface Profile
{
    run: Run;
    kernels: Kernel[];
}

function prepareKey(dict: {[key: string]: Kernel}, key: Metadata | Trace)
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

export function buildProfile(files: TraceFile[]): Profile
{
    if (files.length === 0) return null;

    let run: Run;
    let kernelMap: {[key: string]: Kernel} = {};

    for (const file of files)
    {
        if (file.type === FileType.Metadata)
        {
            const metadata = file.content as Metadata;
            prepareKey(kernelMap, metadata);
            kernelMap[metadata.kernel].metadata = metadata;
            kernelMap[metadata.kernel].name = metadata.kernel;
        }
        else if (file.type === FileType.Trace)
        {
            const trace = file.content as Trace;
            prepareKey(kernelMap, trace);
            kernelMap[trace.kernel].traces.push(trace);
            kernelMap[trace.kernel].name = trace.kernel;
        }
        else if (file.type === FileType.Run)
        {
            run = file.content as Run;
        }
    }

    return {
        run: run,
        kernels: Object.keys(kernelMap).map(key => kernelMap[key])
    };
}

export function groupAccessesByTime(accesses: MemoryAccess[]): MemoryAccess[][]
{
    const grouped = _.groupBy(accesses, access => access.timestamp);
    const sortedKeys = _.keys(grouped).sort();

    return sortedKeys.map(key => grouped[key]);
}

import {FileType, TraceFile} from '../file-load/file';
import {Metadata} from './metadata';
import {Trace} from './trace';

export interface Kernel
{
    metadata?: Metadata;
    traces: Trace[];
}

export function buildKernels(files: TraceFile[]): Kernel[]
{
    let kernelMap: {[key: string]: Kernel} = {};

    for (const file of files)
    {
        if (!kernelMap.hasOwnProperty(file.content.kernel))
        {
            kernelMap[file.content.kernel] = {
                traces: [],
                metadata: null
            };
        }

        if (file.type === FileType.Metadata)
        {
            kernelMap[file.content.kernel].metadata = (file.content as Metadata);
        }
        else kernelMap[file.content.kernel].traces.push((file.content as Trace));
    }

    let kernels: Kernel[] = [];
    for (const key of Object.keys(kernelMap))
    {
        kernels.push(kernelMap[key]);
    }

    return kernels;
}

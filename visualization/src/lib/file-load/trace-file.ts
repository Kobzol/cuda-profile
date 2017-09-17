import {Trace} from '../data/trace';
import {Metadata} from '../trace/metadata';

export enum FileType
{
    Trace = 0,
    Metadata = 1,
    Unknown = 2,
    Invalid = 3
}

export interface TraceFile
{
    name: string;
    loading: boolean;
    content: Trace | Metadata | null;
    type: FileType;
    error: number;
}

export interface FileLoadData
{
    type: FileType;
    content: Trace | Metadata;
}

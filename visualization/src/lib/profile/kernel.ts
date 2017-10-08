import {Metadata} from './metadata';
import {Trace} from './trace';

export interface Kernel
{
    name: string;
    metadata?: Metadata;
    traces: Trace[];
}

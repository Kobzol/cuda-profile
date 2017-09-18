import {actionCreatorFactory} from 'typescript-fsa';
import {TraceFile} from '../file-load/file';
import {Kernel} from './kernel';

const actionCreator = actionCreatorFactory('trace');

export const buildKernels = actionCreator.async<TraceFile[], Kernel[]>('build-kernels');

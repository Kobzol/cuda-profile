import {actionCreatorFactory} from 'typescript-fsa';
import {TraceSelection} from './selection';

const actionCreator = actionCreatorFactory('trace');

export const selectTrace = actionCreator<TraceSelection>('select-trace');
export const selectWarps = actionCreator<number[]>('select-warps');

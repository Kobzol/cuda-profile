import {actionCreatorFactory} from 'typescript-fsa';
import {TraceFile} from '../file-load/file';
import {Profile} from '../profile/profile';
import {TraceSelection} from './selection';

const actionCreator = actionCreatorFactory('trace');

export const buildProfile = actionCreator.async<TraceFile[], Profile>('build-profile');
export const selectTrace = actionCreator<TraceSelection>('select-trace');
export const selectWarps = actionCreator<number[]>('select-warps');

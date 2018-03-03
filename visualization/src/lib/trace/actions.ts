import {actionCreatorFactory} from 'typescript-fsa';
import {TraceSelection} from './selection';
import {Warp} from '../profile/warp';

const actionCreator = actionCreatorFactory('trace');

export const selectTrace = actionCreator<TraceSelection>('select-trace');
export const selectWarps = actionCreator<Warp[]>('select-warps');
export const deselectWarp = actionCreator<Warp>('deselect-warp');
export const clearWarpSelection = actionCreator('clear-warp-selection');

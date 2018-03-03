import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {selectWarps, selectTrace, deselectWarp, clearWarpSelection} from './actions';
import {TraceSelection} from './selection';
import {createSelector} from 'reselect';
import {GlobalState} from '../state/reducers';
import {Kernel} from '../profile/kernel';
import {Trace} from '../profile/trace';
import {ProfileState} from '../profile/reducer';
import {buildProfile} from '../profile/actions';
import {without} from 'ramda';

export interface TraceState
{
    selectedTrace?: TraceSelection;
    selectedWarps: number[];
}

const reducer = reducerWithInitialState<TraceState>({
    selectedTrace: null,
    selectedWarps: []
}).case(buildProfile.done, (state) => ({
    ...state,
    selectedTrace: null,
    selectedWarps: []
})).case(selectTrace, (state, payload) => ({
    ...state,
    selectedTrace: payload,
    selectedWarps: []
})).case(selectWarps, (state, payload) => ({
    ...state,
    selectedWarps: payload.map(warp => warp.index)
})).case(deselectWarp, (state, payload) => ({
    ...state,
    selectedWarps: without([payload.index], state.selectedWarps)
})).case(clearWarpSelection, (state) => ({
    ...state,
    selectedWarps: []
}));

export const selectedKernel = createSelector(
    (state: GlobalState) => state.trace,
    (state: GlobalState) => state.profile,
    (trace: TraceState, profile: ProfileState) =>
        trace.selectedTrace !== null ? profile.profile.kernels[trace.selectedTrace.kernel] : null
);
export const selectedTrace = createSelector(
    (state: GlobalState) => state.trace,
    selectedKernel,
    (state: TraceState, kernel: Kernel) => kernel !== null ? kernel.traces[state.selectedTrace.trace] : null
);
export const selectedWarps = createSelector(
    (state: GlobalState) => state.trace,
    selectedTrace,
    (state: TraceState, trace: Trace) => state.selectedWarps.map(warp => trace.warps[warp])
);

export const traceReducer = reducer;

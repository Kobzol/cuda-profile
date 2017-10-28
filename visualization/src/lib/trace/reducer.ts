import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildProfile, selectWarps, selectTrace} from './actions';
import {Profile} from '../profile/profile';
import {TraceSelection} from './trace-selection';
import {createSelector} from 'reselect';
import {AppState} from '../state/reducers';
import {Kernel} from '../profile/kernel';
import {Trace} from '../profile/trace';

export interface TraceState
{
    profile?: Profile;
    selectedTrace?: TraceSelection;
    selectedWarps: number[];
}

const reducer = reducerWithInitialState<TraceState>({
    profile: null,
    selectedTrace: null,
    selectedWarps: []
}).case(buildProfile.done, (state, payload) => ({
    ...state,
    profile: payload.result,
    selectedTrace: null,
    selectedWarps: []
})).case(selectTrace, (state, payload) => ({
    ...state,
    selectedTrace: payload,
    selectedWarps: []
})).case(selectWarps, (state, payload) => ({
    ...state,
    selectedWarps: payload
}));

export const selectedKernel = createSelector(
    (state: AppState) => state.trace,
    (state: TraceState) => state.selectedTrace !== null ? state.profile.kernels[state.selectedTrace.kernel] : null
);
export const selectedTrace = createSelector(
    (state: AppState) => state.trace,
    selectedKernel,
    (state: TraceState, kernel: Kernel) => kernel !== null ? kernel.traces[state.selectedTrace.trace] : null
);
export const selectedWarps = createSelector(
    (state: AppState) => state.trace,
    selectedTrace,
    (state: TraceState, trace: Trace) => state.selectedWarps.map(warp => trace.warps[warp])
);

export const traceReducer = reducer;

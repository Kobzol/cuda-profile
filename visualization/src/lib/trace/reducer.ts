import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildProfile, selectTrace} from './actions';
import {Profile} from './profile';
import {TraceSelection} from './trace-selection';
import {createSelector} from 'reselect';
import {AppState} from '../state/reducers';
import {Kernel} from './kernel';

export interface TraceState
{
    profile?: Profile;
    selectedTrace?: TraceSelection;
}

const reducer = reducerWithInitialState<TraceState>({
    profile: null,
    selectedTrace: null
}).case(buildProfile.done, (state, payload) => ({
    ...state,
    profile: payload.result,
    selectedTrace: null
})).case(selectTrace, (state, payload) => ({
    ...state,
    selectedTrace: payload
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

export const traceReducer = reducer;

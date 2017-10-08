import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildProfile, selectAccessGroup, selectTrace} from './actions';
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
    selectedAccessGroup?: number;
}

const reducer = reducerWithInitialState<TraceState>({
    profile: null,
    selectedTrace: null,
    selectedAccessGroup: null
}).case(buildProfile.done, (state, payload) => ({
    ...state,
    profile: payload.result,
    selectedTrace: null,
    selectedAccessGroup: null
})).case(selectTrace, (state, payload) => ({
    ...state,
    selectedTrace: payload,
    selectedAccessGroup: null
})).case(selectAccessGroup, (state, payload) => ({
    ...state,
    selectedAccessGroup: payload
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
export const selectedAccessGroup = createSelector(
    (state: AppState) => state.trace,
    selectedTrace,
    (state: TraceState, trace: Trace) => {
        if (state.selectedAccessGroup !== null &&
            state.selectedAccessGroup >= 0 &&
            state.selectedAccessGroup < trace.accessGroups.length)
        {
            return trace.accessGroups[state.selectedAccessGroup];
        }

        return null;
    }
);

export const traceReducer = reducer;

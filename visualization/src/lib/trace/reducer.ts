import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildProfile, selectTrace} from './actions';
import {Profile} from './profile';
import {TraceSelection} from './trace-selection';

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
    profile: payload.result
})).case(selectTrace, (state, payload) => ({
    ...state,
    selectedTrace: payload
}));

export const traceReducer = reducer;
